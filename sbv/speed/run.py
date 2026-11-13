"""Running a room's robot trials in worker processes, recording each trial's judged quantities and where its time
went. Workers are set up as calibrate sets up its own (calibrate._init), so a recorded trial is the trial function
calibrate and the grader call.

Engine modes:
- "plain": every robot trial settles the whole room (no params.freeze, no settle cache): the engine as on main.
- "l1": frozen trials (params.freeze "once") and a settle cache directory shared by the workers (and the gravity cache).

Jobs are registered by kind (JOBS, @job); a worker runs _job(("kind", ...)).
"""

import collections, json, os, time
import multiprocessing as mp
import numpy as np

ROBOT = {"robot", "robot_push", "robot_shelve", "robot_activity"}
_W = dict(stage=collections.defaultdict(float), world=None, probe=None)
JOBS = {}


def job(kind):
    """Register a worker job kind."""

    def reg(f):
        JOBS[kind] = f
        return f

    return reg


def jsonable(x):
    """Plain JSON values (numpy scalars and arrays, sets and tuples made lists)."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def mode_env(mode, cache):
    """Environment of a worker in an engine mode. cache: the run's cache folder (settles are shared through it)."""
    if mode == "plain":
        return dict(SBV_ROBOT_FREEZE="0", SBV_ROBOT_SETTLE_CACHE=None, SBV_GRAVITY_CACHE=os.path.join(cache, "gravity"))
    return dict(
        SBV_ROBOT_FREEZE="1",
        SBV_ROBOT_SETTLE_CACHE=os.path.join(cache, "robot_settle"),
        SBV_GRAVITY_CACHE=os.path.join(cache, "gravity"),
    )


def as_mode(unit, mode, extra=None):
    """A copy of a robot unit set for an engine mode: frozen (params.freeze "once") or not; extra: more params for the
    frozen mode."""
    u = json.loads(json.dumps(unit))
    p = u["params"]
    p.pop("freeze", None)
    if mode == "l1":
        from sbv.functionality.f4_robot.plan import FROZEN

        p["freeze"] = FROZEN
        p.update(extra or {})
    return u


def _timed(cls, name, key):
    f = getattr(cls, name)

    def g(self, *a, **k):
        t = time.time()
        try:
            return f(self, *a, **k)
        finally:
            _W["stage"][key] += time.time() - t

    setattr(cls, name, g)


class StepProbe:
    """Samples the simulation while a trial runs: every `every` steps, the awake trees, contacts and the step's wall
    time (profile only: it wraps mj_step in the robot modules)."""

    def __init__(self, every=25):
        self.every, self.samples, self.n = every, [], 0

    def install(self):
        import mujoco
        from sbv.functionality.f4_robot import plan, pick_place

        probe, real = self, mujoco.mj_step

        class Proxy:
            def __getattr__(self, a):
                return getattr(mujoco, a)

            @staticmethod
            def mj_step(m, d, *a):
                t = time.perf_counter()
                real(m, d, *a)
                probe.n += 1
                if probe.n % probe.every == 0:
                    probe.samples.append(
                        (round(float(d.time), 4), int(d.ntree_awake), int(d.ncon), round(time.perf_counter() - t, 6))
                    )

        for mod in (plan, pick_place):
            mod.mujoco = Proxy()

    def take(self):
        s, self.samples, self.n = self.samples, [], 0
        return s


def _init(xml, phys, center, env, jitter=3, local_m=0.3, probe=False):
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    os.environ.setdefault("MUJOCO_GL", "disable")
    from sbv import calibrate as CB
    from sbv.functionality.f4_robot import plan as RP

    RP._KEPT.clear()
    CB._init(
        xml,
        phys,
        center,
        dict(draws=10, need=9, jitter=jitter, local_m=local_m, small=True, isolate=True, unsettled=set(), restless=[]),
    )
    init = RP.World.__init__

    def world_init(self, *a, **k):
        t = time.time()
        init(self, *a, **k)
        _W["stage"]["build_s"] += time.time() - t
        _W["world"] = self

    RP.World.__init__ = world_init
    _timed(RP.World, "settle", "settle_s")
    _timed(RP.World, "restore", "restore_s")
    _timed(RP.World, "restless", "restless_s")
    for name in ("floor_map", "yaw_maps"):
        f = getattr(RP, name)

        def g(*a, _f=f, **k):
            t = time.time()
            try:
                return _f(*a, **k)
            finally:
                _W["stage"]["floor_s"] += time.time() - t

        setattr(RP, name, g)
    if probe:
        _W["probe"] = StepProbe()
        _W["probe"].install()


def _record(unit, label, res, seconds):
    from sbv.functionality.f4_robot import plan as RP

    outcome, mons = res
    w = _W["world"]
    st = {k: round(v, 3) for k, v in _W["stage"].items()}
    rec = dict(trial=label, ok=outcome == [True], seconds=round(seconds, 2), stages=st)
    st["task_s"] = round(max(0.0, seconds - sum(v for k, v in st.items() if k != "task_s")), 3)
    if mons:
        mo = mons[0]
        det = mo.details or {}
        rec.update(
            success=det.get("success"),
            why=RP.why(det) if not det.get("success") else None,
            stage_points=list(RP.task_points(unit, det)),
            pen=round(float(max(getattr(mo, "pen", {}).values() or [0.0])), 6),
            crossed=bool(mo.crossed(mo.owners)),
            unresolved=bool(mo.report(mo.owners).get("sweep_unresolved")),
            ok_rules=bool(mo.ok_rules(mo.owners)),
            passed_by=det.get("passed_by"),
            details=jsonable(det),
        )
    if w is not None:
        rec.update(sim_s=round(float(w.d.time), 3), ntree=int(w.m.ntree), awake_end=int(w.d.ntree_awake))
    if _W["probe"] is not None:
        rec["samples"] = _W["probe"].take()
    return jsonable(rec)


def _near(unit):
    from sbv import calibrate as CB
    from sbv.functionality.f4_robot import plan as RP

    u = json.loads(json.dumps(unit))
    if "near" not in u["params"]:
        CB.S.reset()
        u["params"]["near"] = RP.nearby(CB.S, u["slots"])
    return u


@job("trials")
def _trials(unit, specs):
    """Robot trials [(seed, edit)] of a unit, each recorded (judged quantities, stage times)."""
    from sbv import calibrate as CB
    from sbv.functionality import units as U

    t0 = time.time()
    out = []
    trial = U.robot_trial(unit["kind"])
    u = _near(unit)
    for seed, edit in specs:
        CB.S.reset()
        _W["stage"].clear()
        _W["world"] = None
        t = time.time()
        res = trial(CB.S, u, None, seed, edit) if edit else trial(CB.S, u, None, seed)
        out.append(
            _record(u, f"edit:{0 if edit.get('mass') == 1.3 else 1}" if edit else f"seed:{seed}", res, time.time() - t)
        )
    return dict(job="trials", id=unit["id"], slots=unit["slots"], seconds=round(time.time() - t0, 1), trials=out)


@job("zone")
def _zone(unit, seed):
    """profile: a changed trial's zone settle at the task's own start, and the restless bodies of that start."""
    from sbv import calibrate as CB
    from sbv.functionality.f4_robot import plan as RP
    from sbv.speed.profile import zone_settle

    t0 = time.time()
    p = unit["params"]
    S = CB.S
    xs, ys, occ = RP.floor_map(S, p["bounds"])
    free = RP.yaw_maps(occ)
    s0 = RP.nearest_free(xs, ys, free, p["start"])
    if s0 is None:
        return dict(job="zone", id=unit["id"], error="no free floor at the start")
    w = RP.World(S.xml, (s0[0], s0[1], 0.0), S.textures, S.phys)
    t = time.time()
    restless = w.restless()
    rt = time.time() - t
    z = zone_settle(w, unit, restless, seed)
    z["restless_s"] = round(rt, 2)
    return jsonable(dict(job="zone", id=unit["id"], zone=z, seconds=round(time.time() - t0, 1)))


@job("nudge")
def _nudge(unit, needed, sigma, seed):
    """calibrate's nudge check, one draw, as the unit's mode runs it (both move the objects in the reference room)."""
    from sbv import calibrate as CB
    from sbv.functionality import units as U

    t0 = time.time()
    S = CB.S
    free = [n for n in needed if S.free_dof(n) is not None]
    trial = U.robot_trial(unit["kind"])
    u = _near(unit)
    u["params"]["needed"] = list(needed)
    S.reset()
    ok = CB._nudged(S, u, trial, free, sigma, seed) if free else "no free object needed"
    return dict(
        job="nudge", id=unit["id"], sigma=sigma, seed=seed, free=free, passed=ok, seconds=round(time.time() - t0, 1)
    )


def _job(job_):
    """One job in a worker: JOBS[kind](*args)."""
    kind, t0 = job_[0], time.time()
    try:
        return JOBS[kind](*job_[1:])
    except Exception as e:  # recorded, never silently dropped: a crashed job fails the gate that needs it
        import traceback

        return dict(
            job=kind,
            id=(job_[1].get("id") if len(job_) > 1 and isinstance(job_[1], dict) else None),
            error=f"{type(e).__name__}: {e}",
            trace=traceback.format_exc()[-2000:],
            seconds=round(time.time() - t0, 1),
        )


def pool(xml, case, jobs, procs, mode, cache, log=print, probe=False, local_m=None):
    """Run jobs (see _job) on the room xml under the case's physics in `procs` workers of engine `mode`; returns the
    results in job order and the wall seconds. Workers are forked, so each pays its own whole-room settle once (shared
    through the gravity cache)."""
    phys, center = case["physics"], case["room"]["center"]
    lm = case.get("functionality", {}).get("local_m", 0.3) if local_m is None else local_m
    jitter = case.get("calibration", {}).get("jitter_seeds", 3)
    t0 = time.time()
    out = [None] * len(jobs)
    ctx = mp.get_context("fork")
    with ctx.Pool(
        max(1, min(procs, len(jobs))),
        _init,
        (xml, phys, center, mode_env(mode, cache), jitter, lm, probe),
        maxtasksperchild=None,
    ) as p:
        for i, r in p.imap_unordered(_indexed, list(enumerate(jobs))):
            out[i] = r
            log(
                f"  [{mode}] {r.get('job')} {r.get('id', r.get('seed', ''))} {r.get('verdict', '')} {r.get('seconds')} s"
                + (f" ERROR {r['error']}" if r.get("error") else "")
            )
    return out, round(time.time() - t0, 1)


def _indexed(a):
    return a[0], _job(a[1])


def named_units(xml, case, isolate=True):
    """The room's robot tasks as calibrate builds them (tasks.T.named on the settled room: the tasks.json ones), with
    the case's accepted robot units (their recorded params: near, needed, nudge) in place of the same id."""
    from sbv.common import case as K
    from sbv.common.scene import Scene
    from sbv.functionality import f2_calibrate as C
    from sbv.functionality.f4_robot import tasks as T

    scene = Scene(xml, case["physics"], probes=True, textures=False)
    scene.settle()
    ctx = C.context(scene, case["room"]["center"])  # as calibrate (LOAD)
    spec = T.load(xml, None)
    rs = list(case["geometry"]["navigation"]["start"]) + [0.0]
    named, bad = T.named(scene, ctx, spec, rs, case["room"]["bounds"])
    for u in named:
        if isolate:
            u["params"]["isolate"] = True
    have = {u["id"]: u for u in case["functionality"]["units"] if u["kind"] in ROBOT}
    out = [have.get(u["id"], u) for u in named] + [u for i, u in have.items() if i not in {x["id"] for x in named}]
    return [json.loads(json.dumps(u)) for u in out], bad
