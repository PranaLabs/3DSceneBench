#!/usr/bin/env python
"""Build a frozen case from a reference room (doc Appendix C.2 and C.5): geometry targets, the F2 bundles and F3
functions with their reference outcomes, after dropping every test that changes under 1 mm jitter and marking the ones
whose outcome depends on hidden weight or friction (hint_only: scored only in the physics-hint setting).
Usage: python -m sbv.calibrate REFERENCE (scene dir or .xml) --out case.json [--procs 8] [--draws 10] [--need 9]
       [--jitter 3] [--nav-start X Y] [--robot-tasks 4] [--push-tasks 2] [--tasks tasks.json] [--allow-no-tasks]
       [--functions functions.json] [--allow-no-functions]
"""

import argparse, ctypes, json, os, secrets, signal, sys, tempfile, time
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repository root
from sbv.common import case as K  # noqa: E402
from sbv.functionality import f2_physical as P, f2_calibrate as C, f3_calibrate as CF, f3_functions as Fn, sleep as SL  # noqa: E402
from sbv.geometry import g2_shape as Geo, g3_layout as Lay  # noqa: E402
from sbv.appearance import views as App  # noqa: E402
from sbv.common.scene import Scene, PHYSICS, category, make, standard_probes  # noqa: E402
from sbv.common.calibration_jobs import Journal, note  # noqa: E402
from sbv.common.calibration_inputs import rigid_identity as identity
from sbv.common import tolerances as Tol  # noqa: E402
from sbv.common.dependencies import operation
from sbv.common.workers import Workers  # noqa: E402
from sbv.functionality.f1_gate import gate, constraints  # noqa: E402
from sbv.functionality.units import run, judge, LOCAL_M, zone, RULES, FALLS  # noqa: E402

S = CTX = None
OPT = {}
LOAD = dict(
    probes=True, textures=False
)  # every room calibration loads: 1-pixel pictures (scene.flat_textures), which no
# result reads; the F1 gate still compiles the real ones once, in a child process


def _init(xml, phys, center, opt):
    global S, CTX, OPT
    S = make(xml, phys, **LOAD)
    S.settle()
    CTX = C.context(S, center)
    OPT = opt


def _local(names, job):
    """Run job on a small model of the room around names (Scene.subset), or, when one cannot be built, on the whole room
    with every object farther than local_m switched off (Scene.focus). The tests read the module's S, so S is
    swapped."""
    global S
    sub = S.subset(names, OPT.get("local_m")) if OPT.get("small") else None
    if sub is None:
        with S.focus(names, OPT.get("local_m")):
            return job()
    full, S = S, sub
    try:
        return job()
    finally:
        S = full


def _init_prepare(procs):
    global OPT
    OPT = dict(procs=procs)  # not in the prepare job's key: the job's result does not depend on it


def _init_records(xml, phys):
    """A worker of the prepare job, with its own copy of the room. On Linux it dies with the job, so a prepare job
    stopped at its timeout leaves no rollout running."""
    global S
    if sys.platform.startswith("linux"):
        ctypes.CDLL(None).prctl(1, int(signal.SIGKILL))  # PR_SET_PDEATHSIG
    S = Scene(xml, phys, **LOAD)


def _record_job(seed):
    keep = {} if seed == 0 else None
    return P.init_record(S, S.jittered(seed, base=S.q0), keep=keep), (keep or {}).get("q1")


def init_records(scene, jitter, procs, keep=None):
    """The nominal (seed 0) and jittered gravity rollouts of the whole room, in seed order. Each rollout starts with
    mj_resetData from a given state, so a fresh copy of the room gives the same record, bit for bit: up to procs at
    once. keep (a dict) receives the nominal run's state at settle_s as "q1" (P.init_record)."""
    if procs < 2 or jitter < 1:
        return [
            P.init_record(scene, scene.jittered(s, base=scene.q0), keep=keep if s == 0 else None)
            for s in range(jitter + 1)
        ]
    with Workers(min(procs, jitter + 1), _init_records, (scene.xml, scene.phys)) as pool:
        out = list(pool.map(_record_job, range(jitter + 1)))
    if keep is not None and out[0][1] is not None:
        keep["q1"] = out[0][1]
    return [r for r, _ in out]


def _filter(unit, want=None, finalized=False):
    """Freeze the reference outcome, then the jitter and picture-decidability checks. Returns (unit, reason)."""
    fn = P.trial if unit["kind"] in ("push", "tilt", "slide", "remove", "edge", "chain") else Fn.trial
    unit["rules"] = RULES  # one run-rule limit for every room (sbv/functionality/integrity.py)
    if not finalized:
        why = C.finalize(S, unit, want, run=fn)
        if why:
            return None, why
    runs = [run(S, unit, None, S.jittered(s)) for s in range(1, OPT["jitter"] + 1)]
    mons = [mo for _, ms in runs for mo in ms]
    if any(mo.crossed(mo.owners) for mo in mons):
        return None, "a jittered reference run crosses a barrier"
    if any(mo.report(mo.owners).get("sweep_unresolved") for mo in mons):
        return None, "a jittered reference motion check is unresolved"
    if unit["kind"] not in FALLS and any(not mo.ok_rules(mo.owners) for mo in mons):
        return (
            None,
            "the reference room keeps an object sunk more than 5 mm into another for 0.3 s or longer (run rule R1)",
        )
    pen = max([max(mo.pen.values() or [0.0]) for mo in mons] or [0.0])
    unit["tol"] = round(max(unit["tol"], 1.5 * pen + 0.002), 4)  # the tolerance covers every reference run
    if not all(
        judge(unit, o, ms)[k]["physics_hint"] for o, ms in runs for k in ("passed", "passed_reference_bar")
    ):  # at both bars
        return None, "outcome changes under 1 mm jitter"
    if unit["kind"] != "joint":  # decided at each pass bar: a wider bar is less sensitive to weight
        ok = C.decidable(S, unit, OPT["draws"], OPT["need"], run=fn)
        unit["hint_only"] = not ok[0]
        if len(ok) > 1:
            unit["hint_only_reference_bar"] = not ok[1]
    return unit, None


def _declared(uid, n, make):
    """(id, unit, None) for a function functions.json declares on container n, or (id, None, why the reference room
    loses it)."""
    if n in OPT["unsettled"]:
        return uid, None, "declared, but the container moves while the reference room settles"
    try:
        u, why = make()
    except P.Unstable:
        u, why = None, "declared, but the reference room's run is numerically unstable"
    return (u["id"],) + _filter(u, finalized=True) if u else (uid, None, why)


def parts(n, joints, declared):
    """Object n's calibration jobs, in the order its tests used to run in one job: its local tests, its declared holds,
    each declared opening, each joint (joints: their names)."""
    c = declared.get("containers", {}).get(n, {})
    return (
        ["tests"]
        + ["holds"] * bool(c.get("holds"))
        + [f"opening|{k}" for k in range(len(c.get("openings", [])))]
        + [f"joint|{j}" for j in joints]
    )


def _object_job(job):
    """One part of object n's calibration (parts). Every test resets the room to the settled start before it runs, so
    the results are those of one job per object, and the job timeout bounds one joint, not all of an object's joints."""
    n, part = job
    S.reset()  # Jobs must not depend on the previous worker task (or on cache hits).
    if part.startswith("joint|"):  # doors and drawers run on the whole room
        j = next(j for j in S.joints[n] if "joint|" + Fn.jname(S.m, j) == part)
        note("joint:" + Fn.jname(S.m, j))
        with SL.isolated(S, [n], OPT["restless"] if OPT.get("isolate") else None):  # what the door cannot reach sleeps
            u, why = CF.cal_joint(S, n, j)
            return [
                (f"joint|{n}|{Fn.jname(S.m, j)}", None, why) if u is None else (u["id"],) + _filter(u, finalized=True)
            ]
    out = []
    _local([n], lambda: _object_tests(n, part, out))  # local runs: only what lies within local_m of n is simulated
    return out


def _object_tests(n, part, out):
    if part == "tests" and n in OPT["unsettled"]:  # it moves while the reference settles: no state to perturb from
        out.append(
            (
                f"perturbations|{n}",
                None,
                "the object moves while the reference settles, so it gets no push, tilt, slide, "
                "edge, grasp or load verifier (its doors and drawers are still tested)",
            )
        )
    elif part == "tests":
        for f in (C.cal_push, C.cal_tilt, C.cal_slide, C.cal_edge):
            note(f.__name__)
            try:
                r = f(S, n, CTX)
            except P.Unstable:
                r = None
            if r:
                out.append((r[0]["id"],) + _filter(*r))
        for f in (CF.cal_grasp,):  # containers are checked only where functions.json declares them; load retired
            note(f.__name__)
            try:
                u = f(S, n)
            except P.Unstable:
                u = None
            if u:
                out.append((u["id"],) + _filter(u, finalized=True))
        on = sorted(w for s, w in CTX["pairs"] if s == n and w in S.objects and S.free_dof(w) is not None)
        if on and CF.turnable(S, n):  # F3.6: a small holder turned over lets go of what rests on it
            note("cal_turnover")
            try:
                u, why = CF.cal_turnover(S, n, on)
            except P.Unstable:
                u, why = None, "the reference room's run is numerically unstable"
            out.append((u["id"],) + _filter(u, finalized=True) if u else (f"turnover|{n}|{'+'.join(on)}", None, why))
    c = OPT["declared"].get("containers", {}).get(n, {})  # functions.json: what this container is declared to do
    if c and part != "tests":
        note("declared functions")
    if c.get("holds") and part == "holds":
        out.append(_declared(f"holds|{n}|{'+'.join(c['holds'])}", n, lambda: CF.cal_holds(S, n, c["holds"])))
    for k, op in enumerate(c.get("openings", [])):
        if part == f"opening|{k}":
            out.append(_declared(f"opening|{n}|{k}", n, lambda: CF.cal_opening(S, n, k, op)))
            if CF.turnable(S, n):
                out.append(_declared(f"turnover|{n}|opening{k}", n, lambda: CF.cal_turnover(S, n, k=k, op=op)))  # F3.6


def _event_job(args):
    S.reset()
    unit, want = args
    note(unit["kind"])
    return _local(zone(unit), lambda: [(unit["id"],) + _filter(unit, want)])


def _why(det):
    """A failed robot task's broken predicates, in words (f4_robot/plan.why; an activity names its failed step)."""
    from sbv.functionality.f4_robot.plan import why

    return why(det)


NUDGES = {"0.05": (1, 2, 3, 4, 5), "0.1": (1, 2, 3, 4, 5)}
# ask is run again with the free objects it needs moved up to 5 and 10 cm (five seeds each); one that fails more than
# one of the five at 5 cm is refused (NUDGE_PASS), the 10 cm share is kept as its margin.
NUDGE_PASS = 0.8  # at 5 cm, at least 4 of the 5 draws must pass


def _nudged(scene, unit, trial, names, sigma, seed):
    """Whether the task still succeeds in the reference room with each free object of names moved up to sigma metres in
    x and y (seeded) at the file's start: in the robot's world (edit {"nudge": ...}, pick_place.setup, which then
    settles as the unit's trials do) and in the floor maps (scene.q0, which they read); None when the run fails. Nothing
    else of the model changes."""
    rng = np.random.default_rng(seed)
    off, q0 = (
        {n: rng.uniform(-sigma, sigma, 2) for n in names},
        scene.q0.copy(),
    )  # the draws, as before: object by object
    try:
        for n, o in off.items():
            a = scene.free_dof(n)[0]
            scene.q0[a : a + 2] += o
        return trial(scene, unit, None, 0, dict(nudge=off))[0] == [True]
    except Exception:
        return None
    finally:
        scene.q0 = q0
        scene.reset()


def needs_and_nudges(scene, unit, det, trial):
    """After a robot task passed in the reference room (det: its nominal run's details): params.needed, what it touched
    or drove past within 1 m of the task (plan.passed_by), and params.nudge, the share of runs it still passes with
    those of them that are free moved up to 5 and 10 cm. Returns why the task is refused (it fails at 5 cm), or None.
    calibrate and add_tasks both call it, so they accept the same tasks."""
    p = unit["params"]
    p["needed"] = [n for n in det.get("passed_by", []) if n not in unit["slots"]]
    free = [n for n in p["needed"] if scene.free_dof(n) is not None]
    p["nudge"] = {k: None for k in NUDGES}
    if free:
        tries = {k: [_nudged(scene, unit, trial, free, float(k), s) for s in seeds] for k, seeds in NUDGES.items()}
        p["nudge"] = {k: (sum(map(bool, v)) / len(v) if None not in v else None) for k, v in tries.items()}
    if p["nudge"]["0.05"] is not None and p["nudge"]["0.05"] < NUDGE_PASS:  # more than one of the five draws failed
        return f"the task fails when the free objects it needs move up to 5 cm ({p['nudge']['0.05']:.0%} of runs pass)"
    return None


def _robot_job(unit):
    """A robot task counts only if it succeeds in the reference and in every jittered reference run, crosses no barrier
    and pushes nothing more than 1 cm into anything. It is hint-only if it fails in a plausible copy error (the target
    30% heavier and every friction +0.15, or 30% lighter and -0.15), like the F2 and F3 units' sensitivity check."""
    from sbv.functionality.f4_robot import plan as robot_plan
    from sbv.functionality.units import robot_trial

    trial = robot_trial(unit["kind"])
    S.reset()
    note(unit["kind"])
    unit["params"]["near"] = robot_plan.nearby(S, unit["slots"])
    runs = [trial(S, unit, None, 0)]
    det = runs[0][1][0].details
    if not det["success"]:
        return unit["id"], None, f"reference task fails: {_why(det)}"  # refused whatever its jittered runs do
    runs += [trial(S, unit, None, s) for s in range(1, OPT["jitter"] + 1)]
    if not all(o == [True] for o, _ in runs):
        return unit["id"], None, "task outcome changes under 1 mm jitter"
    if any(mo.crossed(mo.owners) for _, ms in runs for mo in ms):
        return unit["id"], None, "the reference task crosses a barrier"
    if any(mo.report(mo.owners).get("sweep_unresolved") for _, ms in runs for mo in ms):
        return unit["id"], None, "reference task motion check is unresolved"
    pen = max(max(getattr(mo, "pen", {}).values() or [0.0]) for _, ms in runs for mo in ms)
    if pen > 0.01:
        return unit["id"], None, f"the reference task pushes something {100 * pen:.1f} cm into another object"
    hint = not all(
        trial(S, unit, None, 0, e)[0] == [True] for e in (dict(mass=1.3, friction=0.15), dict(mass=0.7, friction=-0.15))
    )
    why = needs_and_nudges(
        S, unit, det, trial
    )  # what it needs (touched or driven past), and furniture nudged 5 and 10 cm
    if why:
        return unit["id"], None, why
    unit.update(expect=[True], tol=round(max(P.PEN_TOL, 1.5 * pen + 0.002), 4), hint_only=hint, rules=RULES)
    return unit["id"], unit, None


def task_kinds(units):
    """How many working robot tasks of each kind the room has. A working activity counts for the kind of each of its
    steps: a room whose only pick-and-place is a step of a long task still has a working pick-and-place, so its short
    tasks can all be force tasks on other objects."""
    from sbv.functionality.f4_robot.tasks import KINDS

    have = Counter(u["kind"] for u in units)
    have.update(
        KINDS.get(st.get("kind"), st.get("kind"))
        for u in units
        if u["kind"] == "robot_activity"
        for st in u["params"].get("steps", [])
    )
    return have


def _no_task_error(missing, excluded):
    """Why the reference has no working robot task of a kind: the message that stops calibration."""
    lines = []
    for k in missing:
        tried = [f"{e['id']}: {e['reason'][:160]}" for e in excluded if e["id"].split("|")[0] == k][:6]
        lines.append(
            f"no working {dict(robot='pick-and-place', robot_push='push')[k]} task ("
            + ("tried " + "; ".join(tried) if tried else "no object meets the requirements")
            + ")"
        )
    return (
        "the reference room has "
        + " and ".join(lines)
        + " (a step of a working activity counts too). Add task-ready objects "
        ", name tasks in tasks.json, or pass --allow-no-tasks."
    )


def room_bounds(scene):
    pts = np.vstack([np.vstack(scene.aabb(n)) for n in scene.objects])
    lo, hi = pts.min(0), pts.max(0)
    lo = np.array([lo[0] - 0.5, lo[1] - 0.5, -0.05])
    hi = np.array([hi[0] + 0.5, hi[1] + 0.5, max(hi[2] + 0.5, 2.5)])
    arch = [g for g in Lay.world_geoms(scene) if scene.m.geom_type[g] != 0]
    if arch:
        ap = np.vstack([scene.geom_points(g) for g in arch])
        lo[:2], hi[:2] = np.maximum(lo[:2], ap.min(0)[:2]), np.minimum(hi[:2], ap.max(0)[:2])
    return lo.round(3).tolist(), hi.round(3).tolist()


def _prepare(arg):
    ref, jitter, nav_start, mounts, functions, camera_seed = (
        arg[k] for k in ("ref", "jitter", "nav_start", "mounts", "functions", "camera_seed")
    )
    said = []

    def log(line):
        said.append(line)
        note(line)

    xml = K.scene_xml(os.path.abspath(ref))
    phys = dict(
        PHYSICS, probes=standard_probes()
    )  # the probes never take this room's <default> (scene.standard_probes)
    note("F1 gate and model loading")
    ok, rep, plain = gate(xml, phys, textures=False)
    if not ok:
        raise SystemExit(f"the reference fails the F1 gate: {rep}")
    if rep["object_gate"]:
        raise SystemExit(f"objects of the reference fail the object gate: {rep['object_gate']}")
    held = constraints(plain)  # what holds each reference object besides contacts (the object gate)
    vocab = sorted({category(n) for n in plain.objects})
    scene = Scene(xml, phys, vocab, **LOAD)
    scene.reset(scene.q0)
    from sbv.functionality import declared as D  # before the slow steps: a forgotten functions.json stops here

    if arg.get("require_functions", True) and (why := D.missing_file(xml, scene.objects, functions)):
        raise SystemExit(why)
    left = D.undeclared(D.load(xml, functions), scene)
    if left and (functions is not None or os.path.exists(D.path(xml))):
        log(
            f"warning: container-like objects functions.json neither declares nor lists under not_containers: {', '.join(left)}"
        )
    if isinstance(mounts, (str, os.PathLike)):
        with open(mounts) as f:
            mounts = json.load(f)
    mounts = [] if mounts is None else mounts
    if not isinstance(mounts, list):
        raise ValueError("mounts must be a list of {a: object slot, b: world or fixed slot}")
    for mount in mounts:
        if (
            not isinstance(mount, dict)
            or set(mount) != {"a", "b"}
            or not all(isinstance(n, str) for n in mount.values())
            or mount["a"] == mount["b"]
        ):
            raise ValueError("each mount needs distinct a and b slots")
        for n in (mount["a"], mount["b"]):
            if n != "world" and (n not in scene.objects or scene.m.body_weldid[scene.objects[n]] != 0):
                raise ValueError(f"mount endpoint {n!r} must be a fixed reference object or world")
        if mount["a"] == "world":
            raise ValueError("a mount's a endpoint must be an object")
    if mounts:
        from sbv.geometry.support import SupportIndex

        parents, paths = SupportIndex(scene, 0.01, 0.002).graph(
            mounts
        )  # placement's support_tol and penetration_tol below
        for mount in mounts:
            if mount["b"] not in parents[mount["a"]] or mount["a"] not in paths:
                raise ValueError(f"mount {mount} needs a touching fixed interface and a support path to architecture")
    bounds = room_bounds(scene)
    center = [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2]
    note("surface samples and inventory")
    samples = {n: Geo.surface_samples(scene, n) for n in scene.objects}
    objects = []
    for n in sorted(scene.objects):
        b = scene.objects[n]
        lo, hi = scene.aabb(n)
        gs = [g for g in scene.ogeoms[n] if scene.collides(g)]
        objects.append(
            dict(
                slot=n,
                category=scene.cat[n],
                role=scene.role[n],
                center=((lo + hi) / 2).round(4).tolist(),
                dims=np.round(Geo.oriented_dims(samples[n]), 4).tolist(),
                yaw=round(Geo.yaw(scene, n), 2),
                mass=round(float(scene.m.body_subtreemass[b]), 4),
                friction=round(float(np.mean(scene.m.geom_friction[gs, 0])) if gs else 0.0, 4),
                com_rel=(scene.d.subtree_com[b] - (lo + hi) / 2).round(4).tolist(),
                height=round(float(hi[2] - lo[2]), 4),
                constraints=held.get(n),
            )
        )
    note("initialization and jitter")  # the init records first: the nominal one passes through the settled start
    keep = {}
    recs = init_records(scene, jitter, OPT.get("procs", 1), keep=keep)
    note("settling and support relationships")
    if "q1" in keep:
        scene.start = keep["q1"]  # bit for bit what scene.settle() gives (P.init_record), so no second settle
    else:
        scene.settle()  # its first settle_s seconds were unstable: settle() fails the same way
    ctx = C.context(scene, center)
    # geometry targets
    floor = [w for s, w in ctx["pairs"] if s == "world"]
    start = list(nav_start) if nav_start else None
    nav = dict(base_radius=0.3, bounds=bounds, routes=[dict(goal=w, approach=0.8) for w in sorted(floor)])
    if start is None:
        start = Lay.navigation_start(scene, bounds, nav["base_radius"], center)
    else:
        supported = Lay.navigation_start(scene, bounds, nav["base_radius"], start)
        if np.linalg.norm(np.asarray(supported) - start) > Lay.CELL:
            raise ValueError(
                "--nav-start is not on supported, clear floor; choose a point inside the valid floor region"
            )
    nav["start"] = start
    navres = Lay.navigation(nav, scene, {n: n for n in scene.objects})
    nav["routes"] = [dict(goal=r["goal"], approach=r["approach"]) for r in navres["routes"] if r["passed"]]
    note("relations and placement")
    same = {n: n for n in scene.objects}
    place = dict(support_tol=0.01, penetration_tol=0.002, bounds_tol=0.02, bounds=bounds, mounts=mounts)
    scene.reset(scene.q0)
    rels = Lay.relations(
        [dict(type="on", a=w, b=s) for s, w in ctx["pairs"] if s != "world"],
        scene,
        same,
        place["support_tol"],
        place["penetration_tol"],
    )
    place["mounts"] = Lay.mounts(place, scene)
    found = place["mounts"][len(mounts) :]  # --mounts, then the fixed objects that touch
    if found:
        log(f"mounted (fixed, touching, no support path): {', '.join(x['a'] + ' on ' + x['b'] for x in found)}")
    rows = Lay.placement(place, objects, scene, same)["objects"]
    place["exempt"] = sorted(o["object"] for o in rows if not o["passed"])
    if place["exempt"]:
        log(f"warning: exempt from placement, as the reference fails it: {', '.join(place['exempt'])}")
    geometry = dict(
        inventory=dict(required=dict(Counter(o["category"] for o in objects)), permitted_extras=[]),
        count_constraints=[],
        shape=dict(tau=0.03, samples=3000),
        pose_size=[dict(slot=o["slot"], position_tol=0.10, size_bars=list(Geo.SIZE_BARS)) for o in objects],
        relations=[{k: v for k, v in r.items() if k != "passed"} for r in rels["relations"] if r["passed"]],
        placement=place,
        navigation=nav,
        reference_fails=dict(relations=[r for r in rels["relations"] if not r["passed"]], placement=place["exempt"]),
    )
    # functionality units
    excluded, units = [], []
    for n in sorted(scene.objects):
        pen = (
            max(r[n]["integrity"]["max_penetration"] for r in recs if not r[n]["unstable"])
            if not all(r[n]["unstable"] for r in recs)
            else 0.0
        )
        u = dict(
            id=f"init|{n}",
            kind="init",
            slots=[n],
            expect=dict(role=scene.role[n], joints_still=recs[0][n]["joints_still"], mass_range=None),
            tol=round(max(P.PEN_TOL, 1.5 * pen + 0.002), 4),
            hint_only=False,
            rules=RULES,
        )
        bad = [i for i, r in enumerate(recs) if not P.init_passes(u, r[n])]
        if not bad:
            units.append(u)
        else:
            r = recs[bad[0]][n]
            why = [k for k in ("mass", "inertia", "com", "contact", "still", "joints_still") if not r.get(k)]
            excluded.append(
                dict(
                    id=u["id"],
                    reason=f"reference fails its own initialization in run {bad[0]} (0 = nominal): {why} {r['integrity']}",
                )
            )
    unsettled = {
        n for n in scene.objects if any(r[n]["unstable"] or not r[n].get("still", True) for r in recs)
    }  # moved, not merely failed
    declared = D.load(xml, functions)
    bad = D.check(declared, scene)
    if bad:
        raise ValueError("functions.json: " + "; ".join(bad))
    seed = secrets.randbits(31) if camera_seed is None else camera_seed
    return dict(
        xml=xml,
        phys=phys,
        center=center,
        bounds=bounds,
        objects=objects,
        geometry=geometry,
        units=units,
        excluded=excluded,
        unsettled=sorted(unsettled),
        restless=SL.restless(recs),
        nav=nav,
        vocab=vocab,
        declared=declared,
        log=said,
        camera_seed=seed,
        joints={n: [Fn.jname(scene.m, j) for j in scene.joints[n]] for n in scene.objects},
    )


def _resources(arg):
    from sbv.functionality.f1_gate import package, compile_error

    xml = K.scene_xml(arg["ref"])
    ok, why = package(xml)
    if not ok:
        raise SystemExit("reference resources: " + str(why))
    why = compile_error(xml)
    if why:
        raise SystemExit("reference compile: " + why)
    return dict(valid=True)


def resources(ref, checkpoint=None, resume=True):
    from pathlib import Path
    from sbv.common.phase_cache import cached
    from sbv.common.dependencies import model_files, runtime, SCOPES
    from sbv.common.code_dependencies import native

    if checkpoint is None:
        return _resources(dict(ref=ref))
    return cached(
        Path(checkpoint).parent,
        "resources",
        lambda: dict(
            files=model_files(K.scene_xml(ref)), runtime=runtime(packages=SCOPES["rigid"]), code=native("resources")
        ),
        lambda: _resources(dict(ref=ref)),
        resume=resume,
    )


@operation
def build(
    ref,
    procs=8,
    draws=10,
    need=9,
    jitter=3,
    nav_start=None,
    log=print,
    robot_tasks=4,
    push_tasks=2,
    isolate_robot=True,
    tasks=None,
    require_tasks=True,
    mounts=None,
    functions=None,
    camera_seed=None,
    checkpoint=None,
    resume=True,
    job_timeout=3600,
    local_m=LOCAL_M,
    small_models=True,
    isolate=True,
    tolerances=None,
    require_functions=True,
):
    """Resume completed calibration jobs; incomplete execution never yields a partial case. tolerances: a task judge's
    policy (sbv/common/tolerances.py); only an empty one is accepted, and it changes nothing."""
    Tol.require_supported(tolerances)
    from pathlib import Path

    resources(ref, checkpoint, resume)
    if isinstance(mounts, (str, os.PathLike)):
        with open(mounts) as f:
            mounts = json.load(f)
    task_spec = function_spec = None
    if tasks:
        with open(tasks) as f:
            task_spec = json.load(f)
    if functions:
        with open(functions) as f:
            function_spec = json.load(f)
    settings = dict(
        draws=draws,
        need=need,
        jitter=jitter,
        nav_start=nav_start,
        robot_tasks=robot_tasks,
        isolate_robot=isolate_robot,
        push_tasks=push_tasks,
        tasks=task_spec,
        require_tasks=require_tasks,
        mounts=mounts,
        functions=function_spec,
        camera_seed=camera_seed,
        local_m=local_m,
        small_models=small_models,
        isolate=isolate,
    )
    if not require_functions:
        settings["require_functions"] = False  # only when off, so earlier checkpoints keep their identity
    folder = os.path.realpath(ref if os.path.isdir(ref) else os.path.dirname(ref))
    if checkpoint and os.path.commonpath([folder, os.path.realpath(checkpoint)]) == folder:
        raise ValueError("checkpoint directory must be outside the reference room")
    signature = identity(ref, settings)
    with tempfile.TemporaryDirectory(prefix="sbv_calibration_") as temporary:
        with Journal(checkpoint or temporary, signature, log, job_timeout, resume) as journal:
            try:
                case = _build_jobs(
                    ref,
                    procs,
                    draws,
                    need,
                    jitter,
                    nav_start,
                    log,
                    robot_tasks,
                    push_tasks,
                    tasks,
                    require_tasks,
                    mounts,
                    functions,
                    camera_seed,
                    journal,
                    local_m,
                    small_models,
                    isolate,
                    require_functions,
                )
            except BaseException:
                try:
                    unchanged = identity(ref, settings) == signature
                except Exception:
                    unchanged = False
                if not unchanged:
                    journal.invalidate(created_only=True)
                raise
            try:
                unchanged = identity(ref, settings) == signature
            except Exception:
                unchanged = False
            if not unchanged:
                journal.invalidate(created_only=True)
                raise RuntimeError("calibration inputs changed during the run; results cannot be frozen")
            from sbv.common.dependencies import published

            published(
                None,
                check=lambda: identity(ref, settings) == signature,
                invalidate=lambda: journal.invalidate(created_only=True),
            )
            return case


def _build_jobs(
    ref,
    procs,
    draws,
    need,
    jitter,
    nav_start,
    log,
    robot_tasks,
    push_tasks,
    tasks,
    require_tasks,
    mounts,
    functions,
    camera_seed,
    journal,
    local_m=LOCAL_M,
    small_models=True,
    isolate=True,
    require_functions=True,
):
    t0 = time.time()
    small_models = bool(small_models and local_m)
    prepared = journal.jobs(
        "prepare",
        _prepare,
        [
            (
                "room",
                dict(
                    ref=ref,
                    jitter=jitter,
                    nav_start=nav_start,
                    mounts=mounts,
                    functions=functions,
                    camera_seed=camera_seed,
                    **({} if require_functions else dict(require_functions=False)),
                ),
            )
        ],
        1,
        _init_prepare,
        (procs,),
        key_args=lambda a: dict(a, ref="<room>"),
    )[0]
    for line in prepared["log"]:
        log(line)
    xml, phys, center, bounds, objects, geometry, units, excluded, unsettled, nav, vocab = (
        prepared[k]
        for k in (
            "xml",
            "phys",
            "center",
            "bounds",
            "objects",
            "geometry",
            "units",
            "excluded",
            "unsettled",
            "nav",
            "vocab",
        )
    )
    xml = K.scene_xml(os.path.abspath(ref))  # retained prepare metadata may come from another staging directory
    unsettled = set(unsettled)
    opt = dict(
        draws=draws,
        need=need,
        jitter=jitter,
        unsettled=unsettled,
        declared=prepared["declared"],
        local_m=local_m,
        small=small_models,
        isolate=isolate,
        restless=prepared["restless"],
    )
    model = ((xml, phys), LOAD)  # the Scene _init makes: each pool compiles it once (sbv/common/workers.py)
    jobs = [
        (f"{o['slot']}|{p}", (o["slot"], p))
        for o in objects
        for p in parts(o["slot"], prepared["joints"][o["slot"]], prepared["declared"])
    ]
    res = [
        r
        for rs in journal.jobs("objects", _object_job, jobs, procs, _init, (xml, phys, center, opt), model)
        for r in rs
    ]  # one job per part
    scene = Scene(xml, phys, **LOAD)
    scene.settle()
    ctx = C.context(scene, center)
    scene.reset()  # the settled start, not the last jittered init run
    events = []  # support removal and push chains retired
    res += [
        r
        for rs in journal.jobs(
            "events",
            _event_job,
            [(u["id"], (u, want)) for u, want in events],
            procs,
            _init,
            (xml, phys, center, opt),
            model,
        )
        for r in rs
    ]
    from sbv.functionality.f4_robot import pick_place as robot, push as robot_push, plan as robot_plan, tasks as T

    spec = T.load(xml, tasks)
    if (robot_tasks or push_tasks or spec) and robot_plan.installed():
        cap, kept, rs = dict(robot=robot_tasks, robot_push=push_tasks), Counter(), nav["start"] + [0.0]
        named, bad = T.named(scene, ctx, spec, rs, bounds)
        res += [(i, None, w) for i, w in bad]
        ids = {u["id"] for u in named}
        found = (robot.candidates(scene, ctx, rs, bounds) if robot_tasks else []) + (
            robot_push.candidates(scene, ctx, rs, bounds) if push_tasks else []
        )
        for u in (named + found) if journal.signature["settings"]["isolate_robot"] else []:
            u["params"]["isolate"] = True  # the task runs isolated (f4_robot/plan.World.settle), here and in scoring
        for u in (named + found) if robot_plan.requested() else []:
            u["params"]["freeze"] = (
                robot_plan.FROZEN
            )  # frozen trials (SBV_ROBOT_FREEZE=1), here and in scoring (plan.mode)
        for uid, u, why in journal.jobs(
            "robot",
            _robot_job,
            [(u["id"], u) for u in named + [u for u in found if u["id"] not in ids]],
            procs,
            _init,
            (xml, phys, center, opt),
            model,
        ):  # named tasks first, never capped
            if u is None and uid in ids:
                why = "named task: " + why
            if u is None or u.get("named") or kept[u["kind"]] < cap[u["kind"]]:
                res.append((uid, u, why))
                kept[u["kind"] if u else ""] += 1
    elif robot_tasks or push_tasks or spec:
        msg = f"no F4 tasks: the RB-Y1 model is not at {robot_plan.rby1().ROBOT_DIR} (run scripts/get_robot.sh, or set SBV_ROBOT_DIR)"
        if require_tasks:
            raise SystemExit(msg + ", or pass --allow-no-tasks")
        log(msg)
    for uid, u, why in res:
        if u:
            units.append(u)
        elif why:
            excluded.append(dict(id=uid, reason=why))
    have = task_kinds(units)
    missing = [k for k, c in (("robot", robot_tasks), ("robot_push", push_tasks)) if c and not have[k]]
    if require_tasks and missing:
        raise SystemExit(_no_task_error(missing, excluded))
    seed = prepared["camera_seed"]  # private: kept only in the case, never in this repository
    from sbv.appearance.heldout import update_case, supplied

    own = supplied(xml)  # the room's own input_cameras.json and heldout_cameras.json, checked on this room
    if own:
        appearance = dict(
            resolution=512, samples=64, light_z=App.light_height(scene, bounds), conditions=App.CONDITIONS
        )
    else:
        appearance = dict(
            resolution=512,
            samples=64,
            camera_seed=seed,
            camera_jitter=App.JITTER,
            cameras=App.cameras(scene, bounds, res=512, seed=seed),
            light_z=App.light_height(scene, bounds),
            input_cameras=App.cameras(scene, bounds, height=1.6, res=512, kind="input"),
            conditions=App.CONDITIONS,
        )
    appearance = update_case(dict(appearance=appearance, room=dict(bounds=bounds)), scene, own)["appearance"]
    from sbv.appearance import extra as Extra  # the extra pictures, chosen after the held-out cameras

    appearance = Extra.add(appearance, scene, bounds, xml)
    from sbv.common.dependencies import model_files

    case = dict(
        schema=K.SCHEMA,
        reference=os.path.relpath(xml),
        reference_sha256=K.file_sha256(xml),
        reference_files=model_files(xml),
        physics=phys,
        appearance=appearance,
        vocabulary=vocab,
        room=dict(bounds=bounds, center=center),
        objects=objects,
        support_pairs=ctx["pairs"],
        geometry=geometry,
        functionality=dict(
            units=units,
            excluded=excluded,
            repeats=[0],
            local_m=local_m,
            small_models=small_models,
            run_rules=RULES,
            isolate=isolate,
        ),  # scored the same way: local, isolated
        calibration=dict(
            jitter_seeds=jitter, sensitivity_draws=draws, sensitivity_need=need, seconds=round(time.time() - t0, 1)
        ),
    )
    log(
        f"{len(units)} units ({Counter(u['kind'] for u in units)}), {len(excluded)} excluded, "
        f"{sum(u.get('hint_only', False) for u in units)} hint-only, {case['calibration']['seconds']} s"
    )
    return case


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--need", type=int, default=9)
    ap.add_argument("--jitter", type=int, default=3)
    ap.add_argument("--nav-start", type=float, nargs=2)
    ap.add_argument("--robot-tasks", type=int, default=4, help="F4 pick-and-place tasks to keep (0: none)")
    ap.add_argument("--push-tasks", type=int, default=2, help="F4 contact-rich push tasks to keep (0: none)")
    ap.add_argument(
        "--no-isolate-robot", action="store_true", help="robot tasks on the whole room awake (default: isolated)"
    )
    ap.add_argument("--tasks", help="robot tasks the room's builder names (default: tasks.json next to the scene)")
    ap.add_argument(
        "--mounts",
        help="more fixed mounting interfaces as JSON [{a: slot, b: world or slot}], beside the touching ones calibration finds",
    )
    ap.add_argument(
        "--functions", help="what the room's containers are for (default: functions.json next to the scene)"
    )
    ap.add_argument(
        "--allow-no-functions",
        action="store_true",
        help="calibrate a room with container-like objects but no functions.json",
    )
    ap.add_argument(
        "--camera-seed",
        type=int,
        help="seed of the held-out cameras' jitter (default: a fresh random one, recorded in the case)",
    )
    ap.add_argument(
        "--allow-no-tasks",
        action="store_true",
        help="write the case even if the room has no working pick-and-place or push task",
    )
    ap.add_argument("--checkpoint", help="persistent job results (default: OUT.calibration)")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--job-timeout", type=float, default=3600, help="wall seconds per job, including worker setup")
    ap.add_argument(
        "--local-m", type=float, default=LOCAL_M, help="radius of local test runs in metres (0: the whole room)"
    )
    ap.add_argument(
        "--no-small-models", action="store_true", help="run local tests on the whole room with far objects switched off"
    )
    ap.add_argument(
        "--no-isolate",
        action="store_true",
        help="doors and drawers on the whole room awake (default: isolated, functionality/sleep.py)",
    )
    a = ap.parse_args()
    case = build(
        a.reference,
        a.procs,
        a.draws,
        a.need,
        a.jitter,
        a.nav_start,
        robot_tasks=a.robot_tasks,
        push_tasks=a.push_tasks,
        tasks=a.tasks,
        require_tasks=not a.allow_no_tasks,
        mounts=a.mounts,
        isolate_robot=not a.no_isolate_robot,
        functions=a.functions,
        camera_seed=a.camera_seed,
        checkpoint=a.checkpoint or a.out + ".calibration",
        resume=not a.no_resume,
        job_timeout=a.job_timeout,
        local_m=a.local_m or None,
        small_models=not a.no_small_models,
        isolate=not a.no_isolate,
        require_functions=not a.allow_no_functions,
    )
    case["reference"] = os.path.relpath(
        K.scene_xml(os.path.abspath(a.reference)), os.path.dirname(os.path.abspath(a.out))
    )
    K.save(case, a.out)
    print("wrote", a.out)
