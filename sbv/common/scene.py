"""Load a scene into MuJoCo under the case's fixed physics and describe its logical objects.

Submission convention (docs/ENTRANTS.md, section 2): scene.xml at the package root, meters, Z up. Every direct child
body of <worldbody> is one logical object named <category>__<n>; its descendant bodies are its links. A free joint on
the root makes the object movable; hinge and slide joints below the root are its mechanisms. Geoms attached to
<worldbody> itself, and top-level bodies whose category is architectural (wall, floor, ...), are architecture, not
objects. Probe bodies (a ball, a mug, a load box, a two-finger gripper) can be added far away with gravity compensation;
a test switches compensation off when it places one, and reset() parks them again.
"""

import json, os, re, time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
import numpy as np, mujoco

PHYSICS = dict(
    mujoco=mujoco.__version__,
    timestep=0.002,
    integrator="implicitfast",
    cone="elliptic",
    impratio=10.0,
    gravity=[0.0, 0.0, -9.81],
    settle_s=5.0,
)
ARCH = {"floor", "wall", "ceiling", "room", "architecture", "window", "door_frame", "probe"}
FREE, SLIDE, HINGE = 0, 2, 3
BALL_R, MUG_R, MUG_H, LOAD_H, PAD = 0.01, 0.04, 0.05, 0.05, (0.005, 0.02, 0.025)
PARK = np.array([200.0, 200.0, 50.0])
INERT_DAMPING = (
    1e9  # N s/m and N m s/rad on an inert object's joints (Scene.set_inert): the implicit integrator holds it still
)


def _rounds():
    """Unit point sets that bound round primitives: 16-sided rings (pushed out so the polygon contains the circle) at
    several heights. Sphere and ellipsoid: latitude rings on the unit sphere. Cylinder: rings at z = +-1 (scaled by the
    half-length). Capsule: z holds the cap fraction, so z = +-(half-length + radius * |z|)."""
    a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    k = 1 / np.cos(np.pi / 16)
    ring = np.c_[np.cos(a) * k, np.sin(a) * k]
    lat = [(np.cos(t), np.sin(t)) for t in np.linspace(-np.pi / 2, np.pi / 2, 9)]
    sphere = np.vstack(
        [np.c_[ring * r, np.full(16, z)] for r, z in lat]
    )  # latitude rings; the poles are the r = 0 rings
    cyl = np.vstack([np.c_[ring, np.full(16, z)] for z in (-1.0, 1.0)])
    caps = np.vstack([np.c_[ring * r, np.full(16, z if z else 1e-9)] for r, z in lat])
    T = mujoco.mjtGeom
    return {
        int(T.mjGEOM_SPHERE): sphere,
        int(T.mjGEOM_ELLIPSOID): sphere,
        int(T.mjGEOM_CYLINDER): cyl,
        int(T.mjGEOM_CAPSULE): caps,
    }


ROUND = _rounds()


class Unstable(Exception):
    """The engine produced non-finite or exploding accelerations."""


class RuntimeBudget(Exception):
    """The CPU-time deadline passed between two steps (F1's runtime budget): this process's own seconds, so another job
    on the host cannot fail a room (a reference room under a load of 76 on 32 cores: 316 s wall clock, F1 failed, reward
    25)."""


def solver_threads(d):
    """SBV_SOLVER_THREADS (default 1): above 1, d gets MuJoCo's thread pool, which spreads collision detection and the
    solve of independent constraint islands over that many threads. The states stayed the same bit for bit in every
    measurement. Count processes times threads against the cores you have."""
    n = int(os.environ.get("SBV_SOLVER_THREADS", "1"))
    if n < 1:
        raise ValueError("SBV_SOLVER_THREADS must be a positive integer")
    if n > 1:
        mujoco.mju_threadpool(d, n)


CATEGORIES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "categories.txt")


def word_groups(path=CATEGORIES):
    """The public category words: main word -> its aliases. A group main/alias/alias in the file makes the aliases; a
    word may also stand alone on its topic's line."""
    with open(path) as f:
        words = [w for line in f if not line.lstrip().startswith("#") for w in line.split()]
    groups = {w.split("/")[0]: w.split("/")[1:] for w in words if "/" in w}
    alias = {a for al in groups.values() for a in al}
    return {**{w: [] for w in words if "/" not in w and w not in alias}, **groups}


ALIAS = {a: m for m, al in word_groups().items() for a in al}


def main_word(word):
    """The word an object is scored as: an alias's main word (bookshelf -> bookcase), any other word itself."""
    return ALIAS.get(word, word)


def word_topics(path=CATEGORIES):
    """Each word on a topic's line -> that topic, the comment line above it (seating, decor, kitchen and dining,
    ...)."""
    out, topic = {}, None
    with open(path) as f:
        for line in f:
            if line.lstrip().startswith("#"):
                topic = line.strip("# \n")
                continue
            out.update({w: topic for w in line.split() if "/" not in w})
    return out


TOPIC = word_topics()


def topic(word):
    """The kind of thing a word names: its main word's topic in the list (a pillow is bedding, a trinket decor); None
    for a word that is not on the list."""
    return TOPIC.get(main_word(word))


def category(name, vocabulary=()):
    """<category>__<n> names give their category exactly (cup_holder__0 is a cup_holder, not a cup); other names take
    the longest vocabulary word they start with (chair_left is a chair), else the name without a trailing number.
    An alias gives its main word (bookshelf__0 is a bookcase)."""
    n = name.lower()
    if "__" in n:
        return main_word(n.split("__")[0])
    hits = [c for c in vocabulary if n == c or n.startswith(c + "_") or (n.startswith(c) and n[len(c) :].isdigit())]
    if hits:
        return main_word(max(hits, key=len))
    return main_word(re.sub(r"[_\-]?\d+$", "", n) or n)


def is_arch(name):
    """Walls, floors and whole-room shells (SceneSmith calls its shell room_geometry_<room>) are architecture."""
    return category(name) in ARCH or name.lower().startswith("room_geometry")


_DEFAULT_OPT = mujoco.MjModel.from_xml_string("<mujoco/>").opt


def apply_physics(m, phys):
    """Every engine option is the case's or MuJoCo's default, whatever the scene file sets (doc C.2: the case fixes the
    solver and contact settings; a scene's own <option> cannot switch off friction loss or add air density)."""
    for f in dir(_DEFAULT_OPT):
        if f.startswith("_"):
            continue
        v = getattr(_DEFAULT_OPT, f)
        cur = getattr(m.opt, f)
        if isinstance(cur, np.ndarray):
            cur[:] = v
        else:
            setattr(m.opt, f, v)
    m.opt.timestep = phys["timestep"]
    m.opt.impratio = phys["impratio"]
    m.opt.gravity[:] = phys["gravity"]
    m.opt.integrator = getattr(mujoco.mjtIntegrator, "mjINT_" + phys["integrator"].upper())
    m.opt.cone = getattr(mujoco.mjtCone, "mjCONE_" + phys["cone"].upper())


def add_probes(spec, inertials=True):
    """Ball, mug and 5 kg load box as free bodies, plus a two-finger gripper whose wrist a mocap target pulls by a
    weld."""

    def free(name, k, typ, size, mass):
        b = spec.worldbody.add_body(name=name, pos=PARK + [3.0 * k, 0, 0], gravcomp=1)
        b.add_freejoint()
        b.add_geom(type=typ, size=size, mass=mass, friction=[0.5, 0.005, 0.0001])

    free("probe_ball", 0, mujoco.mjtGeom.mjGEOM_SPHERE, [BALL_R, 0, 0], 0.01)
    ball = spec.body("probe_ball").geoms[0]
    ball.priority = 1
    ball.solref = [0.004, 1.0]  # stiffest contact at 2 ms: a
    # dropped ball sinks 1.4 mm into a cup's base instead of 11 mm, so the reference sets no loose penetration tolerance
    free("probe_mug", 1, mujoco.mjtGeom.mjGEOM_CYLINDER, [MUG_R, MUG_H, 0], 0.3)
    free("probe_load", 2, mujoco.mjtGeom.mjGEOM_BOX, [LOAD_H, LOAD_H, LOAD_H], 5.0)
    spec.worldbody.add_body(name="probe_target", mocap=True, pos=PARK + [9, 0, 0])
    wrist = spec.worldbody.add_body(name="probe_wrist", pos=PARK + [9, 0, 0], gravcomp=1)
    wrist.add_freejoint()
    wrist.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0], mass=0.5, contype=0, conaffinity=0)
    eq = spec.add_equality()
    eq.type = mujoco.mjtEq.mjEQ_WELD
    eq.objtype = mujoco.mjtObj.mjOBJ_BODY
    eq.name1, eq.name2 = "probe_target", "probe_wrist"
    eq.data = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], float)
    for side in ("l", "r"):
        f = wrist.add_body(name=f"probe_finger_{side}", gravcomp=1)
        f.add_joint(
            name=f"probe_slide_{side}",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[1, 0, 0],
            range=[-0.1, 0.1],
            damping=20.0,
            limited=mujoco.mjtLimited.mjLIMITED_TRUE,
        )  # said outright: with <compiler autolimits="false"> a range alone does not compile
        f.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(PAD), mass=0.2, friction=[0.2, 0.005, 0.0001], condim=4)
    for b, (mass, ipos, iquat, inertia) in (
        probe_inertials() if inertials else {}
    ).items():  # explicit, whatever the copy's <compiler> says
        body = spec.body(b)
        body.explicitinertial = True
        body.mass, body.ipos, body.iquat, body.inertia = mass, ipos, iquat, inertia


PROBE_FIELDS = dict(
    geom=("contype", "conaffinity", "condim", "priority", "friction", "solmix", "solref", "solimp", "margin", "gap"),
    jnt=("stiffness", "range", "limited", "solref", "solimp", "margin"),
    dof=("armature", "damping", "frictionloss", "solref", "solimp"),
    eq=("solref", "solimp"),
)


def _probe_ids(m):
    """The probes' rows in a compiled model, each under a name that is the same in every room: {"geom": {key: id},
    ...}."""
    name = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
    root = lambda b: b if m.body_parentid[b] == 0 else root(m.body_parentid[b])
    mine = lambda b: name(root(b)).startswith("probe_")
    out = dict(geom={}, jnt={}, dof={}, eq={})
    count = {}
    for kind, n, body in (("geom", m.ngeom, m.geom_bodyid), ("jnt", m.njnt, m.jnt_bodyid), ("dof", m.nv, m.dof_bodyid)):
        for i in range(n):
            if mine(body[i]):
                k = count[kind, body[i]] = count.get((kind, body[i]), -1) + 1
                out[kind][f"{name(body[i])}#{k}"] = i
    for i in range(m.neq):
        if m.eq_objtype[i] == mujoco.mjtObj.mjOBJ_BODY and (mine(m.eq_obj1id[i]) or mine(m.eq_obj2id[i])):
            out["eq"][f"{name(m.eq_obj1id[i])}+{name(m.eq_obj2id[i])}"] = i
    return out


def probe_rows(m):
    """What a room's own <default> and <compiler> give the probes (PROBE_FIELDS), as plain lists, per probe row."""
    return {
        kind: {
            key: {f: np.asarray(getattr(m, f"{kind}_{f}")[i]).tolist() for f in PROBE_FIELDS[kind]}
            for key, i in ids.items()
        }
        for kind, ids in _probe_ids(m).items()
    }


def set_probe_rows(m, rows):
    """Give the probes of a compiled model these rows. True when anything changed (the caller then recomputes the
    model's constants). A model that already has them, the reference room's own, is left untouched bit for bit."""
    changed = False
    ids = _probe_ids(m)
    for kind, per in rows.items():
        for key, fields in per.items():
            if key not in ids[kind]:
                continue
            for f, v in fields.items():
                arr = getattr(m, f"{kind}_{f}")
                v = np.asarray(v, arr.dtype)
                if not np.array_equal(np.asarray(arr[ids[kind][key]]), v):
                    arr[ids[kind][key]] = v
                    changed = True
    if changed:  # what the compiler derives per body from its geoms: the broad phase reads these, not the geoms
        for b in {int(m.geom_bodyid[g]) for g in ids["geom"].values()}:
            gs = np.flatnonzero(m.geom_bodyid == b)
            m.body_contype[b] = int(np.bitwise_or.reduce(m.geom_contype[gs]))
            m.body_conaffinity[b] = int(np.bitwise_or.reduce(m.geom_conaffinity[gs]))
            m.body_margin[b] = float(m.geom_margin[gs].max())
    return changed


_STANDARD_PROBES = None


def standard_probes():
    """The probes with MuJoCo's own defaults, whatever a room's file sets: what calibration uses, and records in the
    case's physics, so a room's own <default> no longer decides how its probes touch it (a reference room whose default
    geom had no collision calibrated with probes that touched nothing, and lost every container, edge and grasp test). A
    case from before has no record and is scored with reference_probes(), as it was calibrated."""
    global _STANDARD_PROBES
    if _STANDARD_PROBES is None:
        spec = mujoco.MjSpec()
        add_probes(spec)
        _STANDARD_PROBES = probe_rows(spec.compile())
    return _STANDARD_PROBES


_REFERENCE_PROBES = {}


def reference_probes(xml):
    """The probes as the reference room's file makes them: probe_rows() of a model holding only that file's <compiler>
    and root <default> (includes followed) and the probes. The probes are added to a room's own file, so they take its
    root defaults (a default contype of 0 leaves the ball and the gripper touching nothing; a default condim, margin or
    joint armature changes how they touch). Calibration froze every probe test with the reference room's; scoring gives
    the copy's probes the same rows (Scene, phys["probes"]), so a copy's defaults never reach them and no case
    changes."""
    key = os.path.abspath(xml)
    if key not in _REFERENCE_PROBES:
        from .dependencies import xml_tree

        root = xml_tree(xml)
        top = ET.Element("mujoco")
        for el in root:
            if el.tag == "compiler":
                top.append(ET.Element("compiler", {k: v for k, v in el.attrib.items() if not k.endswith("dir")}))
            elif el.tag == "default":  # the root class only: the probes belong to no named class
                d = ET.SubElement(top, "default", {k: v for k, v in el.attrib.items()})
                for c in el:
                    if c.tag != "default":
                        d.append(
                            ET.Element(
                                c.tag,
                                {
                                    k: v
                                    for k, v in c.attrib.items()
                                    if k not in ("material", "mesh", "hfield", "class", "childclass")
                                },
                            )
                        )
        spec = mujoco.MjSpec.from_string(ET.tostring(top, encoding="unicode"))
        add_probes(spec)
        _REFERENCE_PROBES[key] = probe_rows(spec.compile())
    return _REFERENCE_PROBES[key]


_PROBE_INERTIALS = None


def probe_inertials():
    """{probe body: (mass, ipos, iquat, diagonal inertia)} as MuJoCo derives them from the probe geoms with its own
    defaults. Pinned on each probe body as an explicit inertial: a copy with <compiler inertiafromgeom="false"> (and a
    default geom mass of 0) otherwise left the probes massless and the model would not compile. The values are MuJoCo's
    own, so a room with ordinary settings compiles to the same numbers as before."""
    global _PROBE_INERTIALS
    if _PROBE_INERTIALS is None:
        s = mujoco.MjSpec()
        add_probes(s, inertials=False)
        m = s.compile()
        _PROBE_INERTIALS = {}
        for b in range(1, m.nbody):
            if m.body_mass[b] > 0:
                _PROBE_INERTIALS[m.body(b).name] = (
                    float(m.body_mass[b]),
                    m.body_ipos[b].copy(),
                    m.body_iquat[b].copy(),
                    m.body_inertia[b].copy(),
                )
    return _PROBE_INERTIALS


def flat_textures(spec):
    """Every texture becomes a 1-pixel builtin before compiling: its name, count and material links stay, so the model
    differs from the full one only in its tex_* arrays, but no picture is decoded. For every verifier but Appearance;
    the F1 gate still compiles the real pictures."""
    for t in spec.textures:
        t.file = ""
        t.cubefiles = [""] * 6
        t.builtin = mujoco.mjtBuiltin.mjBUILTIN_FLAT
        t.width = t.height = 1


def small_xml(xml, drop):
    """The scene file's text without the named top-level bodies, the meshes only they used and its keyframes, with file
    directories made absolute so it compiles from a string. None for a file with includes (left to the caller)."""
    root = ET.parse(xml).getroot()
    base = os.path.dirname(os.path.abspath(xml))
    if root.find(".//include") is not None:
        return None
    comp = root.find("compiler")
    if comp is None:
        comp = ET.SubElement(root, "compiler")
    for k in ("meshdir", "texturedir", "assetdir"):
        if comp.get(k) is not None:
            comp.set(k, os.path.join(base, comp.get(k)))
    if comp.get("assetdir") is None:
        comp.set("assetdir", base)
    wb = root.find("worldbody")
    gone = [b for b in wb.findall("body") if b.get("name") in drop]
    named = lambda els: {e.get("name") for b in els for e in b.iter() if e.get("name")}
    lost, kept = named(gone), named(x for x in wb if x not in gone)
    for b in gone:
        wb.remove(b)
    for sec in [x for t in ("contact", "equality", "tendon", "actuator", "sensor") for x in root.findall(t)]:
        for e in list(sec):  # entries naming only left-out parts go; one tying a kept part to a
            refs = {v for x in e.iter() for v in x.attrib.values()}  # left-out one would change the physics: refuse
            if refs & lost:
                if refs & kept:
                    return None
                sec.remove(e)
    used = {g.get("mesh") for g in root.iter("geom")} | {
        g.get("mesh") for d in root.iter("default") for g in d.iter("geom")
    }
    for asset in root.iter("asset"):
        for x in [
            x
            for x in asset.findall("mesh")
            if x.get("name", os.path.splitext(os.path.basename(x.get("file", "")))[0]) not in used
        ]:
            asset.remove(x)
    for kf in root.findall("keyframe"):
        root.remove(kf)
    return ET.tostring(root, encoding="unicode")


_SLEEP = int(mujoco.mjtEnableBit.mjENBL_SLEEP)


@contextmanager
def sleep_off(m):
    """Sleep switched off for work on scratch data (poses, sweeps), which never steps: with sleep on, MuJoCo 3.14 builds
    data with garbage sleep state at random and poses data built with sleep off wrongly (geoms metres off). Rollouts
    keep sleep as isolation set it (functionality/sleep.py)."""
    f = int(m.opt.enableflags)
    m.opt.enableflags = f & ~_SLEEP
    try:
        yield
    finally:
        m.opt.enableflags = f


# Scenes compiled for the workers forked next (sbv/common/workers.py): key -> [scene, pools using it, pid, files]
_SHARED = {}


def _key(xml, phys=None, vocabulary=(), probes=False, textures=True):
    return (
        os.path.abspath(xml),
        json.dumps(dict(PHYSICS, **(phys or {})), sort_keys=True, default=str),
        tuple(vocabulary),
        bool(probes),
        bool(textures),
    )


def _files(xml):  # the scene's folder as it is now, so an edited file is noticed
    from .dependencies import model_files

    return tuple(sorted(model_files(xml).items()))


def share(args, kw):
    """Compile the Scene of these constructor arguments here, once, for the workers forked next: they take it with
    make() and share its model pages copy-on-write instead of each compiling its own. Returns the key, for unshare()."""
    k = _key(*args, **kw)
    files = _files(args[0])
    if k in _SHARED and _SHARED[k][3] != files:
        del _SHARED[k]  # a file changed since: never hand out the old compile
    if k not in _SHARED:  # no thread pool on the template: its threads would not exist in
        n = os.environ.get("SBV_SOLVER_THREADS")  # the forked workers, which crashed (8 threads: SIGSEGV)
        os.environ["SBV_SOLVER_THREADS"] = "1"
        try:
            _SHARED[k] = [Scene(*args, **kw), 0, os.getpid(), files]
        finally:
            if n is None:
                os.environ.pop("SBV_SOLVER_THREADS")
            else:
                os.environ["SBV_SOLVER_THREADS"] = n
    _SHARED[k][1] += 1
    return k


def unshare(k):
    if k not in _SHARED:
        return
    _SHARED[k][1] -= 1
    if _SHARED[k][1] <= 0:
        del _SHARED[k]


def make(*args, **kw):
    """Scene(*args, **kw). In a worker forked while its parent shared a Scene of these arguments, that Scene with a new
    MjData of its own: the same compile, so the same model bit for bit, its pages copied only where the worker writes.
    Taken once, so two Scenes of one process never share a model; never in the process that compiled it."""
    k = _key(*args, **kw)
    if k not in _SHARED or _SHARED[k][2] == os.getpid():
        return Scene(*args, **kw)
    t = _SHARED.pop(k)[0]
    t.d = mujoco.MjData(t.m)
    solver_threads(t.d)
    mujoco.mj_forward(t.m, t.d)
    t.inherited = True
    return t


ARENA = 1 << 30  # bytes of MuJoCo arena per simulation, the same for the reference room and every copy


def fixed_size(spec):
    """The verifiers' own <size>, as <option> is theirs: a copy's njmax / nconmax / memory set the arena every worker
    allocates, so it decided whether the copy could be scored at all. Every scene gets ARENA instead, whatever its file
    says."""
    spec.memory, spec.njmax, spec.nconmax, spec.nstack = ARENA, -1, -1, -1
    return spec


class Scene:
    def __init__(self, xml, phys=None, vocabulary=(), probes=False, textures=True, text=None):
        """text: the scene file's text to compile instead of reading xml (Scene.subset)."""
        self.xml, self.phys, self.textures, self.vocabulary = (
            xml,
            dict(PHYSICS, **(phys or {})),
            textures,
            tuple(vocabulary),
        )
        spec = mujoco.MjSpec.from_file(xml) if text is None else mujoco.MjSpec.from_string(text)
        if not textures:
            flat_textures(spec)
        if probes:
            add_probes(spec)
        fixed_size(spec)
        self.m = m = spec.compile()
        apply_physics(m, self.phys)
        self.d = d = mujoco.MjData(m)
        solver_threads(d)
        if probes and self.phys.get("probes") and set_probe_rows(m, self.phys["probes"]):
            mujoco.mj_setConst(m, d)  # the reference room's probes, not this file's (reference_probes)
        m.geom_rgba[m.geom_rgba[:, 3] == 0, 3] = (
            1e-3  # mj_ray skips fully transparent geoms; collision proxies often are
        )
        m.mat_rgba[m.mat_rgba[:, 3] == 0, 3] = 1e-3  # and a geom with a material is judged by the material's alpha
        self.name = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
        self.owner = np.zeros(m.nbody, int)
        for b in range(1, m.nbody):
            self.owner[b] = b if m.body_parentid[b] == 0 else self.owner[m.body_parentid[b]]
        self.objects, self.probe = {}, {}
        has_geom = set(self.owner[m.geom_bodyid[g]] for g in range(m.ngeom))
        for b in range(1, m.nbody):
            if m.body_parentid[b] != 0:
                continue
            n = self.name(b)
            if n.startswith("probe_"):
                self.probe[n[6:]] = b
            elif not is_arch(n) and b in has_geom:
                self.objects[n] = b  # a geom-less root (camera rig, light) is no object
        self.cat = {n: category(n, vocabulary) for n in self.objects}
        self.links = {n: [k for k in range(m.nbody) if self.owner[k] == b] for n, b in self.objects.items()}
        self.ogeoms = {
            n: [g for g in range(m.ngeom) if self.owner[m.geom_bodyid[g]] == b] for n, b in self.objects.items()
        }
        self.joints = {
            n: [j for j in range(m.njnt) if self.owner[m.jnt_bodyid[j]] == b and m.jnt_type[j] in (SLIDE, HINGE)]
            for n, b in self.objects.items()
        }
        self.role = {
            n: ("free" if self.free_dof(n) is not None else "fixed") + ("+joints" if self.joints[n] else "")
            for n in self.objects
        }
        # The whole room's objects in order, each with free root and movable: random draws walk this list, so an object
        # gets the same numbers in a small model (Scene.subset keeps the whole room's list) as in the whole room.
        self.draw_order = [
            (n, self.free_dof(n) is not None, self.free_dof(n) is not None or bool(self.joints[n]))
            for n in self.objects
        ]
        self.set_inert(self.phys.get("inert", ()))  # objects that blow up the copy (f1_gate.blame), if any
        if probes:
            self.slide = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"probe_slide_{s}") for s in "lr"}
            self.finger = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"probe_finger_{s}") for s in "lr"}
            self.mocap = int(m.body_mocapid[self.probe["target"]])
        mujoco.mj_forward(m, d)
        self.q0 = d.qpos.copy()
        self.start = self.q0.copy()
        self.on_step = None  # called after every step (isolation, functionality/sleep.py)

    # ---- state -------------------------------------------------------------------------------------------------
    def free_dof(self, n):
        """(qpos address, dof address) of the object's root free joint, or None for a fixed root."""
        b, m = self.objects[n], self.m
        j = m.body_jntadr[b]
        return (m.jnt_qposadr[j], m.jnt_dofadr[j]) if j >= 0 and m.jnt_type[j] == FREE else None

    def set_inert(self, names):
        """The object gate's way to score a copy without the objects that blow it up (F1.4): each named object stays
        where the copy put it and takes no part in the physics. Its geoms touch nothing, and a huge damping holds every
        one of its joints still, so neither its contacts nor its own springs and masses can blow up the rest. A Scene
        built with phys["inert"] does this itself, so workers and small models get the same."""
        m = self.m
        for n in set(names) & set(self.objects):  # a small model may have left the object out
            m.geom_contype[self.ogeoms[n]] = 0
            m.geom_conaffinity[self.ogeoms[n]] = 0
            m.dof_damping[np.isin(m.dof_bodyid, self.links[n])] = INERT_DAMPING

    def reset(self, qpos=None):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[:] = self.start if qpos is None else qpos
        for b in self.probe.values():
            m.body_gravcomp[b] = 1.0
        mujoco.mj_forward(m, d)

    def step(self, seconds, monitor=None, deadline=None):
        """deadline: a time.process_time() value; past it, RuntimeBudget is raised before the next step."""
        m, d, hook = self.m, self.d, self.on_step
        for k in range(int(round(seconds / m.opt.timestep))):
            if deadline is not None and time.process_time() > deadline:
                raise RuntimeBudget()
            if monitor is not None:
                monitor.before_step()
            mujoco.mj_step(m, d)
            if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number or not np.isfinite(d.qacc).all():
                raise Unstable("unstable simulation")
            if hook is not None:
                hook()
            if monitor is not None:
                monitor.tick()

    def settle(self):
        from sbv.common.gravity_cache import cached  # SBV_GRAVITY_CACHE: this model settled before

        self.start = self.q0.copy()
        self.reset()
        self.start = np.array(
            cached(self, "settle", lambda: (self.step(self.phys["settle_s"]), self.d.qpos.tolist())[1], self.q0)
        )

    def zone_objects(self, names, radius):
        """The objects within radius of names: the gap between bounding-sphere boxes at the settled start."""
        m, d = self.m, self.d
        self.reset()
        box = {}
        for k, gs in self.ogeoms.items():
            if gs:
                c, r = d.geom_xpos[gs], m.geom_rbound[gs][:, None]
                box[k] = ((c - r).min(0), (c + r).max(0))
        gap = lambda a, b: float(np.linalg.norm(np.maximum(0, np.maximum(a[0] - b[1], b[0] - a[1]))))
        return {
            k
            for k in self.objects
            if k not in box or any(n in box and gap(box[k], box[n]) <= radius for n in names) or k in names
        }

    def subset(self, names, radius):
        """A small model of the room for a local run: only the objects within radius of names, the architecture and the
         probes, at their settled start; every other object is left out, so the solver never steps it
        . None when it cannot be built (something else in the file names a left-out body):
         then run the same zone with Scene.focus on the whole room."""
        names = [n for n in names if n in self.objects]
        if radius is None or not names:
            return None
        keep = self.zone_objects(names, radius)
        try:
            text = small_xml(self.xml, set(self.objects) - keep)
            sub = (
                None
                if text is None
                else Scene(
                    self.xml, self.phys, self.vocabulary, probes=bool(self.probe), textures=self.textures, text=text
                )
            )
        except Exception:
            return None
        if sub is None:
            return None
        if set(sub.objects) != keep:
            return None
        for full, part in (
            (self.start, sub.start),
            (self.q0, sub.q0),
        ):  # body by body, joint by joint: same order in both
            for n in keep:
                for bf, bs in zip(self.links[n], sub.links[n]):
                    for jf, js in zip(
                        range(self.m.body_jntadr[bf], self.m.body_jntadr[bf] + self.m.body_jntnum[bf]),
                        range(sub.m.body_jntadr[bs], sub.m.body_jntadr[bs] + sub.m.body_jntnum[bs]),
                    ):
                        w = {0: 7, 1: 4}.get(int(self.m.jnt_type[jf]), 1)
                        a, b = self.m.jnt_qposadr[jf], sub.m.jnt_qposadr[js]
                        part[b : b + w] = full[a : a + w]
        sub.draw_order = self.draw_order
        sub.reset()
        return sub

    @contextmanager
    def focus(self, names, radius):
        """Local runs: only the objects within `radius` of `names` (gap between bounding-sphere boxes at the settled
        start) move and collide; every other object floats where it rests and touches nothing. A test on `names` then
        steps a handful of objects instead of the whole room, with the same outcome.
        radius None, or no named object: the whole room, as before."""
        names = [n for n in names if n in self.objects]
        if radius is None or not names:
            yield
            return
        m, d = self.m, self.d
        saved = (m.geom_contype.copy(), m.geom_conaffinity.copy(), m.body_gravcomp.copy())
        self.reset()
        box = {}
        for k, gs in self.ogeoms.items():
            if gs:
                c, r = d.geom_xpos[gs], m.geom_rbound[gs][:, None]
                box[k] = ((c - r).min(0), (c + r).max(0))
        gap = lambda a, b: float(np.linalg.norm(np.maximum(0, np.maximum(a[0] - b[1], b[0] - a[1]))))
        for k in self.objects:
            if k in box and all(n not in box or gap(box[k], box[n]) > radius for n in names):
                m.geom_contype[self.ogeoms[k]] = 0
                m.geom_conaffinity[self.ogeoms[k]] = 0
                m.body_gravcomp[self.links[k]] = 1.0
        try:
            yield
        finally:
            m.geom_contype[:], m.geom_conaffinity[:], m.body_gravcomp[:] = saved

    def jittered(self, seed, sigma=0.001, base=None):
        """The settled start (or base) with every free root moved by up to sigma in x and y; seed 0 leaves it
        unchanged."""
        q = (self.start if base is None else base).copy()
        if seed:
            rng = np.random.default_rng(seed)
            for n, free, _ in self.draw_order:
                if not free:
                    continue
                dx = rng.uniform(-sigma, sigma, 2)
                f = self.free_dof(n) if n in self.objects else None
                if f is not None:
                    q[f[0] : f[0] + 2] += dx
        return q

    def place(self, probe, pos, quat=(1, 0, 0, 0)):
        m, d = self.m, self.d
        b = self.probe[probe]
        j = m.body_jntadr[b]
        a, v = m.jnt_qposadr[j], m.jnt_dofadr[j]
        d.qpos[a : a + 3] = pos
        d.qpos[a + 3 : a + 7] = quat
        d.qvel[v : v + 6] = 0
        m.body_gravcomp[b] = 0.0
        mujoco.mj_forward(m, d)

    # ---- geometry ----------------------------------------------------------------------------------------------
    def visible(self, gs):
        """The geoms a camera shows, by the MuJoCo viewer's convention: group below 3 and not transparent; all of them
        when none qualifies. Used by both the renders and the G2 shape samples."""
        gs = list(gs)
        v = [g for g in gs if self.m.geom_group[g] < 3 and self.m.geom_rgba[g, 3] > 0.01]
        return v or gs

    def geom_points(self, g):
        """World-frame points that bound a geom: mesh vertices; for round primitives, rings around the surface (so a
        cylinder turned 45 degrees stays 10 cm wide, not the 14 cm of its box's corners); box corners otherwise."""
        m, d = self.m, self.d
        t, s = int(m.geom_type[g]), m.geom_size[g]
        T = mujoco.mjtGeom  # int: numpy ints fail `in` on enums
        if t == T.mjGEOM_MESH:
            i = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[i] : m.mesh_vertadr[i] + m.mesh_vertnum[i]]
        elif t in ROUND:
            v = ROUND[t] * (
                s[[0, 0, 0]]
                if t == T.mjGEOM_SPHERE
                else s[[0, 1, 2]]
                if t == T.mjGEOM_ELLIPSOID
                else np.array([s[0], s[0], 1.0])
            )
            if t == T.mjGEOM_CYLINDER:
                v[:, 2] *= s[1]
            if t == T.mjGEOM_CAPSULE:
                v[:, 2] = np.sign(v[:, 2]) * (s[1] + s[0] * np.abs(v[:, 2]))
        else:
            c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
            v = c + h * np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
        return v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]

    def aabb(self, n, geoms=None):
        pts = np.vstack([self.geom_points(g) for g in (self.ogeoms[n] if geoms is None else geoms)])
        return pts.min(0), pts.max(0)

    def center(self, n):
        lo, hi = self.aabb(n)
        return (lo + hi) / 2

    def collides(self, g):
        return bool(self.m.geom_contype[g] or self.m.geom_conaffinity[g])

    def subtree(self, body):
        m = self.m
        out = {body}
        for k in range(body + 1, m.nbody):
            if m.body_parentid[k] in out:
                out.add(k)
        return out
