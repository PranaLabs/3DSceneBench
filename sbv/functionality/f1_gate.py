"""F1 simulation validity (doc Appendix C.2): one execution gate per scene, the AND of package and resources, units
and frames, executable definitions, and load-and-step. It checks executability, not correctness."""

import os, subprocess, sys, time
import xml.etree.ElementTree as ET
import numpy as np, mujoco
from sbv.common.scene import Scene, Unstable, RuntimeBudget
from sbv.common.gravity_cache import cached

FILE_TAGS = {"mesh", "texture", "hfield", "skin", "include", "model"}
BLEW_UP = "blows up the simulation (F1.4); the rest of the copy is scored with this object inert"


def package(xml):
    """Every file the MJCF names (following <include>) exists, by a relative path that stays inside the package."""
    if os.path.islink(xml):
        return False, ["scene.xml must be a regular package file, not a symlink"]
    root_dir = os.path.dirname(os.path.realpath(xml))
    errors = []
    visited = set()
    dirs = dict(meshdir="", texturedir="", assetdir="")

    def walk(path):
        resolved = os.path.realpath(path)
        if resolved in visited:
            return
        visited.add(resolved)
        try:
            tree = ET.parse(path)
        except (ET.ParseError, OSError) as e:
            errors.append(f"cannot parse {os.path.basename(path)}: {e}")
            return
        for comp in tree.getroot().iter(
            "compiler"
        ):  # one setting for the whole model, as MuJoCo reads it: a later <compiler>, or one in an
            dirs.update(
                {k: comp.get(k) for k in dirs if comp.get(k)}
            )  # included file read before this one, counts for every file
        for b_ in tree.getroot().findall("worldbody/body"):
            if (b_.get("name") or "").startswith("probe_"):
                errors.append(f"body name {b_.get('name')}: names starting with probe_ are the verifiers' own")
        for el in tree.getroot().iter():
            outside = lambda v: os.path.isabs(v) or not (
                os.path.realpath(os.path.join(root_dir, v)) + os.sep
            ).startswith(root_dir + os.sep)
            for (
                k,
                v,
            ) in el.attrib.items():  # any other path: a cube texture's fileright .. fileback, a file on another element, a <compiler>'s folders
                other = (k.startswith("file") and not (k == "file" and el.tag in FILE_TAGS)) or (
                    el.tag == "compiler" and k in ("meshdir", "texturedir", "assetdir")
                )
                if other and v and outside(v):
                    errors.append(f"{el.tag} {k} leaves the package: {v}")
            f = el.get("file")
            if not f or el.tag not in FILE_TAGS:
                continue
            sub = dirs["meshdir"] if el.tag in ("mesh", "skin") else dirs["texturedir"] if el.tag == "texture" else ""
            sub = sub or (dirs["assetdir"] if el.tag != "include" else "")
            p = os.path.realpath(os.path.join(root_dir, sub, f))  # follow symlinks before checking the package boundary
            if os.path.isabs(f) or not p.startswith(root_dir + os.sep):
                errors.append(f"{el.tag} path leaves the package: {f}")
            elif not os.path.exists(p):
                errors.append(f"missing {el.tag} file: {f}")
            elif el.tag == "include":
                walk(p)

    if not os.path.exists(xml):
        return False, [f"no scene file at {xml}"]
    walk(xml)
    return not errors, errors


def object_values(scene):
    """F1.3 by owner: ({object: [bad values]}, [bad values of the architecture]). A bad value fails only the object that
    owns it (the object gate); one on the walls or floor fails the whole copy."""
    m = scene.m
    per, arch = {}, []

    def put(body, what):
        n = scene.name(int(scene.owner[body])) if body else "world"
        (per.setdefault(n, []) if n in scene.objects else arch).append(what)

    for arr, owner, label in (
        (m.body_mass, range(m.nbody), "mass"),
        (m.body_inertia, range(m.nbody), "inertia"),
        (m.geom_size, m.geom_bodyid, "geom size"),
        (m.geom_friction, m.geom_bodyid, "friction"),
        (m.dof_damping, m.dof_bodyid, "damping"),
        (m.dof_frictionloss, m.dof_bodyid, "friction loss"),
    ):
        if not len(arr):
            continue  # nothing of that kind (a room without joints has no dofs)
        a = np.asarray(arr).reshape(len(arr), -1)
        for i in np.flatnonzero(~np.isfinite(a).all(1) | (a < 0).any(1)):
            put(int(owner[i]), f"{label} not finite or negative")
    for j in range(m.njnt):
        if not np.isfinite(m.jnt_range[j]).all() or (m.jnt_limited[j] and not m.jnt_range[j][0] < m.jnt_range[j][1]):
            put(int(m.jnt_bodyid[j]), f"joint {j} range not finite or empty")
    return per, arch


def constraints(scene):
    """What holds each object besides contacts: equality constraints (welds, connects) touching its bodies, gravity
    compensation, and contact excludes. The object gate fails a copy object that has more than its reference object."""
    m = scene.m
    out = {n: dict(equality=0, gravcomp=0.0, exclude=0) for n in scene.objects}
    to_body = {
        mujoco.mjtObj.mjOBJ_BODY: lambda i: i,
        mujoco.mjtObj.mjOBJ_JOINT: lambda i: m.jnt_bodyid[i],
        mujoco.mjtObj.mjOBJ_GEOM: lambda i: m.geom_bodyid[i],
        mujoco.mjtObj.mjOBJ_SITE: lambda i: m.site_bodyid[i],
    }

    def owners(bodies):
        return {scene.name(int(scene.owner[b])) for b in bodies if b > 0} & set(out)

    for e in range(m.neq):
        f = to_body.get(int(m.eq_objtype[e]))
        if f is None:
            continue
        sides = [int(f(i)) for i in (m.eq_obj1id[e], m.eq_obj2id[e]) if i >= 0]  # -1: the world
        for n in owners(sides):
            out[n]["equality"] += 1
    for sig in m.exclude_signature:
        for n in owners([int(sig) >> 16, int(sig) & 0xFFFF]):
            out[n]["exclude"] += 1
    for n in out:
        out[n]["gravcomp"] = round(float(max(m.body_gravcomp[b] for b in scene.links[n])), 4)
    return out


def units_frames(scene):
    """Meters and Z up, judged from the geometry: the objects fit a room (median size 2 cm to 5 m, scene under 100 m),
    most objects stand at or above z = 0, and if the scene has planes, one of them faces +z (a floor). A single sunken
    object is a placement error for G3, not a units error."""
    m, d = scene.m, scene.d
    scene.reset(scene.q0)
    bad = []
    if not scene.objects:
        return False, ["no objects (top-level bodies) in the scene"]
    sizes = [float(np.max(np.subtract(*scene.aabb(n)[::-1]))) for n in scene.objects]
    lows = [scene.aabb(n)[0][2] for n in scene.objects]
    allp = np.vstack([np.vstack(scene.aabb(n)) for n in scene.objects])
    if not 0.02 <= float(np.median(sizes)) <= 5.0:
        bad.append(f"median object size {np.median(sizes):.3g} m: wrong units?")
    if float(np.max(allp.max(0) - allp.min(0))) > 100:
        bad.append("scene larger than 100 m: wrong units?")
    if float(np.median(lows)) < -0.05:
        bad.append(f"most objects stand below z = 0 (median bottom {np.median(lows):.3f} m): Z not up?")
    planes = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
    if planes and not any(d.geom_xmat[g].reshape(3, 3)[2, 2] > 0.99 for g in planes):
        bad.append("no plane faces +z (no floor): Z not up?")
    return not bad, bad


class GraderKilled(RuntimeError):
    """The compile check's child process was killed with SIGKILL, as the out-of-memory killer does: the grader's
    failure, never the copy's. evaluate lets it through, so no report is written and the run gets no reward
    (pipeline.reward_file)."""


def compile_error(xml):
    """The file compiled as submitted, pictures included, in a child process: a broken texture still fails the gate, and
    the decoded pictures never stay in the scorer's memory (on macOS a freed model's pages stay in the process). None if
    it compiles, else the error."""
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, mujoco\ntry: mujoco.MjSpec.from_file(sys.argv[1]).compile()\n"
            "except Exception as e: sys.exit(str(e))",
            xml,
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode == -9:
        raise GraderKilled("the compile check was killed (SIGKILL, out of memory?): a grader failure, rerun")
    return (
        None if r.returncode == 0 else (r.stderr.removesuffix("\n") or f"exit code {r.returncode}")
    )  # sys.exit adds one newline


def cpu_budget(budget_s):
    """The runtime budget in CPU seconds of this process: budget_s per solver thread (SBV_SOLVER_THREADS, default 1),
    since time.process_time() adds up every thread's time."""
    return budget_s * int(os.environ.get("SBV_SOLVER_THREADS", "1"))


def gate(xml, phys, vocabulary=(), budget_s=300.0, textures=True):
    """Returns (E_s, report, scene or None). The scene is loaded without probes, exactly as submitted; textures=False
    loads it with 1-pixel textures (Scene) after compile_error() has compiled the real ones."""
    rep = {}
    ok, rep["package"] = package(xml)
    if not ok:
        return False, rep, None
    try:
        err = None if textures else compile_error(xml)
        if err:
            raise ValueError(err)
        scene = Scene(xml, phys, vocabulary, textures=textures)
    except GraderKilled:
        raise
    except Exception as e:
        rep["definitions"] = [f"does not compile: {str(e)[:300]}"]
        return False, rep, None
    try:
        rep["object_gate"], rep["definitions"] = object_values(scene)
        ok_d = not rep["definitions"]
        ok_u, rep["units_frames"] = units_frames(scene)
    except Exception as e:  # never let a strange scene crash the scorer
        ok_d, ok_u, rep["units_frames"] = False, False, [f"cannot be checked: {type(e).__name__}: {str(e)[:200]}"]
    ok_s, rep["load_and_step"], rep["cost"] = cached(
        scene,
        "f1",
        lambda: load_and_step(scene, phys, budget_s),
        budget_s,
        keep=lambda r: r[2]["seconds"] < cpu_budget(budget_s) and not any("runtime budget" in w for w in r[1]),
    )
    # never keep a run over the budget: it measures the machine's load, not the room (the time budget stops one before
    # its first step)
    if ok_d and ok_u and rep["load_and_step"] == ["unstable simulation"]:  # the object gate: which objects blew it up?
        inert = blame(
            scene,
            set(rep["object_gate"]),
            phys["settle_s"] + 1.0,
            time.process_time() + cpu_budget(budget_s) - rep["cost"]["seconds"],
        )
        if inert:
            ok_s, rep["load_and_step"], rep["inert"] = True, [], inert
            for n in inert:
                rep["object_gate"].setdefault(n, []).append(BLEW_UP)
    return bool(ok_d and ok_u and ok_s), rep, scene


def suspects(scene, inert, seconds, deadline):
    """None if the scene, with the named objects inert (Scene.set_inert), steps `seconds` from its submitted start
    without blowing up; else every other object with joints, the largest acceleration at the blow-up first. MuJoCo's own
    reset after a bad acceleration is off meanwhile, so the bad values stay to be read. The scene is left as it was."""
    m = scene.m
    saved = (m.geom_contype.copy(), m.geom_conaffinity.copy(), m.dof_damping.copy(), int(m.opt.disableflags))
    try:
        scene.set_inert(inert)
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
        scene.reset(scene.q0)
        scene.step(seconds, deadline=deadline)
        return None
    except Unstable:
        acc, own = np.nan_to_num(np.abs(scene.d.qacc), nan=np.inf), scene.owner[m.dof_bodyid]
        top = {n: acc[own == b].max() for n, b in scene.objects.items() if n not in inert and (own == b).any()}
        return sorted(top, key=lambda n: -top[n])
    finally:
        m.geom_contype[:], m.geom_conaffinity[:], m.dof_damping[:], m.opt.disableflags = saved
        scene.reset(scene.q0)


def blame(scene, bad, seconds, deadline, most=3):
    """The objects that blow up the scene (F1.4), for the object gate: at most `most`, each of them needed, that let it
    step `seconds` when inert. At each blow-up the first suspect is made inert, until the scene steps: an object with a
    bad value of its own (F1.3, in `bad`; a fixed one too), else the largest acceleration. Then each goes back if the
    scene still steps without it. [] when that finds none in time: the copy fails F1.4, as before."""
    try:
        out, fast = [], suspects(scene, [], seconds, deadline)
        while fast is not None and len(out) < most:
            first = [n for n in fast if n in bad] + sorted(set(bad) - set(fast) - set(out)) + fast
            if not first:
                break  # nothing to blame: only architecture moves
            out.append(first[0])
            fast = suspects(scene, out, seconds, deadline)
        if fast is not None:
            return []
        for n in list(out):
            if len(out) > 1 and suspects(scene, [x for x in out if x != n], seconds, deadline) is None:
                out.remove(n)
        return out
    except RuntimeBudget:
        return []


def load_and_step(scene, phys, budget_s):
    """[passed, problems, cost]: the scene steps settle_s + 1 s without blowing up, within the runtime budget (CPU
    seconds of this process, scene.RuntimeBudget: budget_s per solver thread, cpu_budget)."""
    t0 = time.process_time()
    ok_s, why = True, []
    cpu = cpu_budget(budget_s)
    try:
        scene.reset(scene.q0)
        total = phys["settle_s"] + 1.0
        for _ in range(int(round(total / 0.5))):  # in half-second chunks; the budget is checked every step
            scene.step(0.5, deadline=t0 + cpu)
            if time.process_time() - t0 > cpu:
                raise RuntimeBudget()
    except RuntimeBudget:
        ok_s = False
        why = [f"over the runtime budget of {budget_s:.0f} s"]
    except Unstable as e:
        ok_s = False
        why = [str(e)]
    return [
        ok_s,
        why,
        dict(seconds=round(time.process_time() - t0, 3), sim_seconds=phys["settle_s"] + 1.0, geoms=int(scene.m.ngeom)),
    ]
