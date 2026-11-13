#!/usr/bin/env python
"""The entrants' format checker: does a copy load and run the way the verifiers need? (The F1 gate: files inside the
folder, compiles, plausible units, Z up, named objects, steps without blowing up.) It names each object whose word is an
alias with the word it is scored as, and, given the vocabulary, each word not on it. It also checks nonrigid.json when
the folder has one; with --nonrigid, a missing one is a problem. (the checker names the new ones.)

This file stands alone: MuJoCo, NumPy and (for mesh soft bodies) trimesh, nothing from sbv. It is the only program of
ours that ships to entrants, so it must never import the verifiers (tests/test_checker.py runs it with sbv absent).
Its constants mirror sbv/common/scene.py and sbv/nonrigid/schema.py; the same self-test keeps them equal.
Usage: python check_scene.py SCENE_DIR [--vocabulary words.json] [--reward reward.json] [--nonrigid] [--budget 300]
The vocabulary file is a list of words, or {"words": [...], "aliases": {"mug": "cup", ...}}."""

import argparse, json, os, subprocess, sys, time
import xml.etree.ElementTree as ET
import numpy as np, mujoco

PHYSICS = dict(
    timestep=0.002, integrator="implicitfast", cone="elliptic", impratio=10.0, gravity=[0.0, 0.0, -9.81], settle_s=5.0
)
ARCH = {"floor", "wall", "ceiling", "room", "architecture", "window", "door_frame", "probe"}
FILE_TAGS = {"mesh", "texture", "hfield", "skin", "include", "model"}
ARENA = 1 << 30
KINDS = ("fluid", "soft_body", "thin_shell")
OLD_KINDS = {"liquid": "fluid", "soft_solid": "soft_body", "cloth": "thin_shell"}
VISCOSITY = ("water", "oil", "syrup")
FABRIC = ("silk", "cotton", "denim", "wool_coat")
UNSTABLE = "unstable simulation"


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
            p = os.path.realpath(os.path.join(root_dir, sub, f))
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


def load(xml, phys):
    """The model under the verifiers' fixed settings: their arena size and every engine option, whatever the file
    says."""
    spec = mujoco.MjSpec.from_file(xml)
    spec.memory, spec.njmax, spec.nconmax, spec.nstack = ARENA, -1, -1, -1
    m = spec.compile()
    default = mujoco.MjModel.from_xml_string("<mujoco/>").opt
    for f in [f for f in dir(default) if not f.startswith("_")]:
        v, cur = getattr(default, f), getattr(m.opt, f)
        if isinstance(cur, np.ndarray):
            cur[:] = v
        else:
            setattr(m.opt, f, v)
    m.opt.timestep, m.opt.impratio, m.opt.gravity[:] = phys["timestep"], phys["impratio"], phys["gravity"]
    m.opt.integrator = getattr(mujoco.mjtIntegrator, "mjINT_" + phys["integrator"].upper())
    m.opt.cone = getattr(mujoco.mjtCone, "mjCONE_" + phys["cone"].upper())
    return m, mujoco.MjData(m)


def objects(m):
    """({object: root body}, owner root of every body): each direct child of <worldbody> with a geom, except probes and
    architecture. A name's word is what comes before __, else the name without a trailing number."""
    name = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
    owner = np.zeros(m.nbody, int)
    for b in range(1, m.nbody):
        owner[b] = b if m.body_parentid[b] == 0 else owner[m.body_parentid[b]]
    has_geom = set(owner[m.geom_bodyid[g]] for g in range(m.ngeom))
    out = {}
    for b in range(1, m.nbody):
        n = name(b)
        w = n.lower().split("__")[0] if "__" in n else n.lower().rstrip("0123456789").rstrip("_-")
        if (
            m.body_parentid[b] == 0
            and b in has_geom
            and w not in ARCH
            and not n.startswith("probe_")
            and not n.lower().startswith("room_geometry")
        ):
            out[n] = b
    return out, owner, name


def object_values(m, objs, owner, name):
    """({object: [bad values]}, [bad values of the architecture]): a bad value fails only the object that owns it."""
    per, arch = {}, []
    roots = {b: n for n, b in objs.items()}

    def put(body, what):
        n = roots.get(int(owner[body]) if body else 0)
        (per.setdefault(n, []) if n else arch).append(what)

    for arr, own, label in (
        (m.body_mass, range(m.nbody), "mass"),
        (m.body_inertia, range(m.nbody), "inertia"),
        (m.geom_size, m.geom_bodyid, "geom size"),
        (m.geom_friction, m.geom_bodyid, "friction"),
        (m.dof_damping, m.dof_bodyid, "damping"),
        (m.dof_frictionloss, m.dof_bodyid, "friction loss"),
    ):
        if not len(arr):
            continue
        a = np.asarray(arr).reshape(len(arr), -1)
        for i in np.flatnonzero(~np.isfinite(a).all(1) | (a < 0).any(1)):
            put(int(own[i]), f"{label} not finite or negative")
    for j in range(m.njnt):
        if not np.isfinite(m.jnt_range[j]).all() or (m.jnt_limited[j] and not m.jnt_range[j][0] < m.jnt_range[j][1]):
            put(int(m.jnt_bodyid[j]), f"joint {j} range not finite or empty")
    return per, arch


def aabb(m, d, geoms):
    pts = []
    for g in geoms:
        c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
        v = c + h * np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
        pts.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    pts = np.vstack(pts)
    return pts.min(0), pts.max(0)


def units_frames(m, d, objs, owner):
    """Meters and Z up, judged from the geometry: median object size 2 cm to 5 m, scene under 100 m, most objects at or
    above z = 0, and if the scene has planes, one faces +z."""
    mujoco.mj_forward(m, d)
    bad = []
    if not objs:
        return False, ["no objects (top-level bodies) in the scene"]
    boxes = [aabb(m, d, [g for g in range(m.ngeom) if owner[m.geom_bodyid[g]] == b]) for b in objs.values()]
    sizes = [float(np.max(hi - lo)) for lo, hi in boxes]
    lows = [lo[2] for lo, _ in boxes]
    allp = np.vstack([np.vstack(b) for b in boxes])
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


def load_and_step(m, d, objs, owner, phys, budget_s, bad=()):
    """[passed, problems, suspects]: the scene steps settle_s + 1 s from the file's start without blowing up, within
    budget_s CPU seconds. When it blows up, suspects are the objects with a bad value of their own, then those with the
    largest accelerations: the grader scores such a copy with the objects to blame made inert, losing only their
    verifiers."""
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    t0 = time.process_time()
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
    for _ in range(int(round((phys["settle_s"] + 1.0) / m.opt.timestep))):
        if time.process_time() - t0 > budget_s:
            return False, [f"over the runtime budget of {budget_s:.0f} s"], []
        mujoco.mj_step(m, d)
        if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number or not np.isfinite(d.qacc).all():
            acc, own = np.nan_to_num(np.abs(d.qacc), nan=np.inf), owner[m.dof_bodyid]
            top = {n: acc[own == b].max() for n, b in objs.items() if (own == b).any()}
            fast = sorted(top, key=lambda n: -top[n])
            return False, [UNSTABLE], ([n for n in fast if n in bad] + [n for n in fast if n not in bad])[:3]
    return True, [], []


def gate(xml, phys=PHYSICS, budget_s=300.0):
    """(ok, report, (model, data, objects) or None)."""
    rep = {}
    ok, rep["package"] = package(xml)
    if not ok:
        return False, rep, None
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
    )  # a broken texture fails here
    if r.returncode < 0:
        rep["definitions"] = [
            f"the compile was stopped by signal {-r.returncode} (9: most often out of memory in this check, not a fault of the scene): run the check again with more memory"
        ]
        return False, rep, None
    if r.returncode:
        rep["definitions"] = [
            f"does not compile: {(r.stderr.removesuffix(chr(10)) or f'exit code {r.returncode}')[:300]}"
        ]
        return False, rep, None
    m, d = load(xml, phys)
    objs, owner, name = objects(m)
    try:
        rep["object_gate"], rep["definitions"] = object_values(m, objs, owner, name)
        ok_d = not rep["definitions"]
        ok_u, rep["units_frames"] = units_frames(m, d, objs, owner)
    except Exception as e:
        ok_d, ok_u, rep["units_frames"] = False, False, [f"cannot be checked: {type(e).__name__}: {str(e)[:200]}"]
    ok_s, rep["load_and_step"], rep["blow_up_suspects"] = load_and_step(
        m, d, objs, owner, phys, budget_s, set(rep.get("object_gate", ()))
    )
    return bool(ok_d and ok_u and ok_s), rep, (m, d, objs)


def nonrigid_problems(side, bodies):
    """nonrigid.json's problems, as sbv/nonrigid/schema.py finds them, plus 'on' names not in scene.xml."""
    try:
        with open(side) as f:
            top = json.load(f)
    except (ValueError, RecursionError, UnicodeDecodeError) as e:
        return [], [f"nonrigid.json is not valid JSON: {type(e).__name__}: {str(e)[:200]}"]
    objs = top.get("objects") if isinstance(top, dict) else None
    if not isinstance(objs, list):
        return [], ["no 'objects' list"]
    problems, names, kept = [], set(), []
    for i, o in enumerate(objs):  # a report for any file, never a traceback: the file is whatever was written
        try:
            problems += entry_problems(o, i, names, side, bodies)
            kept.append(o)
        except Exception as e:
            problems.append(f"entry {i}: cannot be read ({type(e).__name__}: {str(e)[:120]})")
    return kept, problems


def entry_problems(o, i, names, side, bodies):
    problems = []
    if not isinstance(o, dict):
        return [f"entry {i}: not an object"]
    n, k, mat, s = o.get("name", ""), o.get("kind"), o.get("material", {}), o.get("shape", {})
    if not isinstance(n, str):
        return [f"entry {i}: its name is not text"]
    if not isinstance(mat, dict):
        problems.append(f"{n}: material is an object")
        mat = {}
    if not isinstance(s, dict) or not isinstance(s.get("type"), str):
        problems.append(f"{n}: shape is an object with a type")
        s = {}
    if "__" not in n or n in names:
        problems.append(f"{n!r}: names are <category>__<n> and unique")
    names.add(n)
    if k in OLD_KINDS:
        problems.append(f"{n}: kind {k!r} is now {OLD_KINDS[k]!r}")
    elif k not in KINDS:
        problems.append(f"{n}: kind must be one of {KINDS}")
    if not o.get("on") or not isinstance(o["on"], str):
        problems.append(f"{n}: 'on' must name a rigid body in scene.xml")
    elif o["on"] not in bodies | {"floor"}:
        problems.append(f"{n}: 'on' names {o['on']!r}, which is not an object in scene.xml")
    if k == "fluid" and mat.get("viscosity") not in VISCOSITY:
        problems.append(f"{n}: viscosity must be one of {list(VISCOSITY)}")
    if k == "thin_shell" and mat.get("fabric") not in FABRIC:
        problems.append(f"{n}: fabric must be one of {list(FABRIC)}")
    if k == "soft_body" and not all(
        isinstance(mat.get(x), (int, float)) and mat[x] > 0 for x in ("youngs_kpa", "density")
    ):
        problems.append(f"{n}: youngs_kpa and density must be positive numbers")
    if s.get("type") == "mesh":
        f = str(s.get("file", ""))
        p = os.path.join(os.path.dirname(side), f)
        if os.path.basename(f) != f or not f.endswith(".obj") or not os.path.exists(p):
            problems.append(f"{n}: a mesh shape names an .obj file next to nonrigid.json")
        elif k == "soft_body":
            import trimesh

            if not trimesh.load(p, force="mesh", process=False).is_watertight:
                problems.append(f"{n}: a soft body's mesh must be closed")
        if k == "thin_shell" and not (isinstance(s.get("rail"), list) and len(s["rail"]) == 3):
            problems.append(f"{n}: a thin shell's mesh also gives rail [x, y, z]")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--vocabulary")
    ap.add_argument("--reward")
    ap.add_argument("--nonrigid", "--soft", action="store_true")
    ap.add_argument("--budget", type=float, default=300.0)
    a = ap.parse_args(argv)
    v = json.load(open(a.vocabulary)) if a.vocabulary and os.path.exists(a.vocabulary) else []
    words, aliases = (v.get("words", []), v.get("aliases", {})) if isinstance(v, dict) else (v, {})
    path = os.path.join(a.scene, "scene.xml") if os.path.isdir(a.scene) else a.scene
    ok, rep, loaded = gate(path, budget_s=a.budget)
    objs = list(loaded[2]) if loaded else []
    unnamed = [n for n in objs if "__" not in n]
    named = sorted(set(objs) - set(unnamed))
    word = lambda n: n.split("__")[0].lower()
    out = dict(
        loads_and_runs=bool(ok),
        objects=len(objs),
        names_without_category=unnamed[:20],
        aliases=[f"{n} is scored as {aliases[word(n)]}" for n in named if word(n) in aliases],
        words_not_on_the_list=[n for n in named if words and word(n) not in words][:20],
        report=rep,
    )
    side = os.path.join(os.path.dirname(path), "nonrigid.json")
    if a.nonrigid or os.path.exists(side):
        old = os.path.join(os.path.dirname(path), "soft_liquid.json")
        sobjs, problems = (
            nonrigid_problems(side, set(objs))
            if os.path.exists(side)
            else (
                [],
                [
                    "no nonrigid.json"
                    + (" (soft_liquid.json is its old name: rename it)" if os.path.exists(old) else "")
                ],
            )
        )
        out["nonrigid"] = dict(objects=len(sobjs), problems=problems)
    print(json.dumps(out, indent=1, default=str))
    if a.reward:
        os.makedirs(os.path.dirname(os.path.abspath(a.reward)), exist_ok=True)
        r = {"format_ok": float(bool(ok) and not unnamed)}
        if a.nonrigid:
            r["nonrigid_ok"] = float(not out["nonrigid"]["problems"])
        with open(a.reward, "w") as f:
            json.dump(r, f)
    return 0 if ok and not unnamed and (not a.nonrigid or not out["nonrigid"]["problems"]) else 1


if __name__ == "__main__":
    sys.exit(main())
