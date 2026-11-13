"""Planning for the F4 pick-and-place controller: the room plus the RB-Y1 in one model, a floor map for the base,
grid paths, and stand spots from which the arm reaches the needed fingertip poses. The robot model and its inverse
kinematics come from rby1.py (the MuJoCo Menagerie files: scripts/get_robot.sh)."""

import contextlib, fcntl, hashlib, json, math, os
from collections import deque
from pathlib import Path
import numpy as np, mujoco
from scipy import ndimage

CELL, NYAW = 0.05, 16  # 5 cm cells (layout.CELL), 16 headings
LATERAL, CARRY = (
    -0.15,
    np.array([0.40, -0.12, 1.12]),
)  # the right hand works 15 cm right of centre; carry pose in the base frame
FORWARD = (0.65, 0.72, 0.58, 0.80)
# Two opt-in speed switches for robot trials; unset, every trial settles the whole room.
CACHE_ENV = (
    "SBV_ROBOT_SETTLE_CACHE"  # a directory (every process) or "mem" (this one): whole-room settles kept (World.kept)
)
FREEZE_ENV = (
    "SBV_ROBOT_FREEZE"  # 1: calibrate and add_tasks record frozen trials (params.freeze); 0: off whatever the case says
)
FROZEN = "once"  # the one frozen mode params.freeze records (mode)
STATE, NEVER = mujoco.mjtState.mjSTATE_INTEGRATION, int(mujoco.mjtSleepPolicy.mjSLEEP_NEVER)
_KEPT = {}  # this process's kept settles, by key (World.kept)


def requested():
    """Whether calibrate and add_tasks record frozen trials in the robot units they build (SBV_ROBOT_FREEZE=1)."""
    return os.environ.get(FREEZE_ENV, "") == "1"


def mode(unit):
    """How a robot trial settles (pick_place.setup). None: the whole room for 5 s, every trial, as before. "once"
    (frozen trials): a trial that changes nothing for its task (no edit, no nudge: the nominal run, every copy's scored
    run and each jitter seed) starts from the whole-room settle of its model, start and seed, kept (World.restore), so
    it is the same, bit for bit, as before; a trial that changes something for its task (an edit, a nudge) applies its
    change at the file's start, as before, and then settles only the task's zone, the objects the whole-room settle
    moves and what touches them, the rest asleep from step 0 (World.freeze, World.restless). The case decides
    (params.freeze, which calibrate and add_tasks write when SBV_ROBOT_FREEZE=1), so a copy is scored the way its
    reference room was calibrated; SBV_ROBOT_FREEZE=0 only forces it off. "file", which an earlier draft of this switch
    recorded, reads as "once"."""
    if os.environ.get(FREEZE_ENV, "") == "0":
        return None
    return FROZEN if unit["params"].get("freeze") in ("once", "file") else None


def carry_point(base):
    """The hand's carry position (high and close to the body) in world coordinates for a base pose (x, y, yaw)."""
    c, s = math.cos(base[2]), math.sin(base[2])
    return [base[0] + c * CARRY[0] - s * CARRY[1], base[1] + s * CARRY[0] + c * CARRY[1], CARRY[2]]


def rby1():
    from sbv.functionality.f4_robot import rby1 as R

    return R


def installed():
    """Whether the RB-Y1 model files are present."""
    return os.path.exists(rby1().ROBOT_DIR)


class World:
    """Room plus robot; exposes what the integrity Monitor needs (m, d, owner, name, collides) and the room's
    objects."""

    base = None  # restore(): the settled room's (x0, x1, qpos), which settle_steps returns while nothing changed
    settled = False  # stepped or restored once (sbv/add_tasks.py shares such a world: restore leaves it as it is)
    edited = False  # edit_physics changed the model: a restored world settles again

    def __init__(self, xml, start, textures=True, phys=None):
        """phys: the case's physics (Scene.phys), with the copy's inert objects if the object gate named any."""
        from sbv.common.scene import apply_physics, PHYSICS, solver_threads

        rr = rby1()
        self.m, self.d = rr.build(xml, start, textures=textures)
        m = self.m
        self.phys = dict(PHYSICS, **(phys or {}))
        apply_physics(m, self.phys)  # the case's physics, whatever the submitted file sets, then the robot's own
        m.opt.impratio, m.opt.solver, m.opt.iterations = (
            100.0,
            mujoco.mjtSolver.mjSOL_NEWTON,
            rr.ITERATIONS,
        )  # solver settings
        solver_threads(self.d)
        mujoco.mj_forward(m, self.d)
        self.rb = rr.Robot(m, self.d)
        self.name = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
        self.owner = np.zeros(m.nbody, int)
        for b in range(1, m.nbody):
            self.owner[b] = b if m.body_parentid[b] == 0 else self.owner[m.body_parentid[b]]
        self.robot = {b for b in range(1, m.nbody) if self.name(self.owner[b]).startswith(rr.P)}
        from sbv.common.scene import is_arch

        has_geom = set(self.owner[m.geom_bodyid[g]] for g in range(m.ngeom))
        self.objects = {
            self.name(b): b
            for b in range(1, m.nbody)
            if m.body_parentid[b] == 0 and b not in self.robot and not is_arch(self.name(b)) and b in has_geom
        }
        self.robot_roots = {int(self.owner[b]) for b in self.robot}
        from sbv.common.scene import (
            INERT_DAMPING,
        )  # the object gate (f1_gate.blame): an object that blows the copy up stays

        for n in set(self.phys.get("inert", ())) & set(
            self.objects
        ):  # where it is and takes no part here either (Scene.set_inert)
            b = self.objects[n]
            gs = [g for g in range(m.ngeom) if self.owner[m.geom_bodyid[g]] == b]
            m.geom_contype[gs] = 0
            m.geom_conaffinity[gs] = 0
            m.dof_damping[self.owner[m.dof_bodyid] == b] = INERT_DAMPING
        mujoco.mj_forward(m, self.d)

    def settle(self, keep=None, awake=None):
        """The room and the parked robot settle for 5 s, as Scene does. keep (object names): the task then runs isolated
        : every contact island holding no robot part, none of keep and nothing that moved in the
         settle as the init records judge it (1 cm in all, 1 mm in its last second) sleeps from the next step, at rest,
         and wakes at the step an awake body touches it. Nothing leaves the model; the robot never sleeps. awake (object
         names; frozen trials, mode): the settle itself moves only these, the robot and what touches them (freeze)."""
        x0, x1 = self.settle_steps() if awake is None else self.settle_steps(awake)
        if keep is not None:
            self.isolate(keep, x0, x1)

    def settle_steps(self, awake=None):
        """The 5 s settle itself; returns the bodies' positions before it and 1 s before its end, for isolate. It does
        not depend on the task, so trials of one room, start and jitter seed can share it (sbv/add_tasks.py, and
        kept()). A world restored to the settled room (restore) that nothing changed since is settled already: no
        step."""
        if self.base is not None and not self.edited and np.array_equal(self.d.qpos, self.base[2]):
            return self.base[:2]
        if awake is None:
            return self.kept()
        saved = self.freeze(awake)
        try:
            return self.steps()
        finally:
            self.m.tree_sleep_policy[:] = saved  # isolate then decides what sleeps, as after a whole-room settle

    def steps(self):
        m, d = self.m, self.d
        n = int(self.phys["settle_s"] / m.opt.timestep)
        x0 = x1 = d.xipos.copy()
        self.last = None
        for k in range(n):
            if k == n - int(round(1.0 / m.opt.timestep)):
                x1 = d.xipos.copy()
            if k == n - 1:
                self.last = np.empty(mujoco.mj_stateSize(m, STATE))
                mujoco.mj_getState(m, d, self.last, STATE)
            mujoco.mj_step(m, d)
        self.settled = True
        return x0, x1

    def kept(self, memo=False):
        """steps(), or the same settle kept before for this compiled model (every array but the pictures, as
        gravity_cache.key: room, assets, robot, physics, edits) and the state it starts from (start, jitter), with the
        code and library versions. SBV_ROBOT_SETTLE_CACHE: a directory, shared by every process and computed once (a
        lock), or "mem", this process only; memo: this process even if unset. The state one step before the end is kept
        and that step replayed, so MjData is what stepping leaves, bit for bit (tests/test_settle_cache.py)."""
        where = os.environ.get(CACHE_ENV, "off")
        disk = where not in ("", "off", "mem")
        if not (disk or memo or where == "mem"):
            return self.steps()
        from sbv.common.gravity_cache import key

        m, d = self.m, self.d
        pre = np.empty(mujoco.mj_stateSize(m, STATE))
        mujoco.mj_getState(m, d, pre, STATE)
        k = key(self, "robot_settle", hashlib.sha256(pre.tobytes()).hexdigest())
        path = Path(where) / f"robot_settle-{k}.json"
        with contextlib.ExitStack() as held:
            if k not in _KEPT and disk:
                path.parent.mkdir(parents=True, exist_ok=True)
                lock = held.enter_context(open(path.with_suffix(".lock"), "w"))
                fcntl.flock(lock, fcntl.LOCK_EX)  # one process settles; the others wait, then read it
                try:
                    _KEPT[k] = {a: np.asarray(b, float) for a, b in json.loads(path.read_text()).items()}
                except (OSError, ValueError):
                    pass
            if k in _KEPT:  # replay the last step from the state kept before it
                v = _KEPT[k]
                mujoco.mj_setState(m, d, v["state"], STATE)
                mujoco.mj_step(m, d)
                self.settled = True
                return v["x0"], v["x1"]
            x0, x1 = self.steps()
            if self.last is None:
                return x0, x1  # no step to replay (settle_s 0): nothing to keep
            if len(_KEPT) >= 8:
                _KEPT.pop(next(iter(_KEPT)))
            _KEPT[k] = v = dict(state=self.last, x0=x0, x1=x1)
            if disk:
                try:
                    tmp = path.with_suffix(f".{os.getpid()}.tmp")
                    tmp.write_text(json.dumps({a: b.tolist() for a, b in v.items()}, allow_nan=False))
                    os.replace(tmp, path)  # atomic: a reader never sees half a file
                except (OSError, ValueError):
                    pass  # unwritable storage or a non-finite state: not on disk
            return x0, x1

    def restore(self):
        """A frozen unchanged trial's start (mode): the room and the parked robot as the whole-room settle of this model
        and start leaves them, unjittered and unedited, settled once and kept (kept); settle_steps then has nothing to
        do. A world settled already stays as it is."""
        if self.settled:
            return
        x0, x1 = self.kept(memo=True)
        self.base = (x0, x1, self.d.qpos.copy())

    def restless(self):
        """The objects the whole-room settle of this model and start moves, by isolate's rule (1 cm in all, or 1 mm in
        its last second): a changed frozen trial keeps them awake in its settle, so a room that is not at rest away from
        the task (a copy's crate falling somewhere) settles there as it did before. Reads the kept settle (kept, memo:
        the one restore shares), then puts this world back at its start, unsettled."""
        m, d = self.m, self.d
        pre = np.empty(mujoco.mj_stateSize(m, STATE))
        mujoco.mj_getState(m, d, pre, STATE)
        x0, x1 = self.kept(memo=True)
        moved = self.moved(x0, x1)
        out = {n for n, b in self.objects.items() if moved[self.owner == b].any()}
        mujoco.mj_setState(m, d, pre, STATE)
        mujoco.mj_forward(m, d)
        self.settled = False
        self.last = None
        return out

    def freeze(self, names):
        """A frozen trial's settle (mode): the robot, the named objects and whatever shares a contact island with them
        move; everything else sleeps from step 0 at its place, and wakes at the step an awake body touches it (sleep).
        Returns the sleep policies to put back."""
        m = self.m
        awake = np.zeros(m.ntree, bool)
        for b in range(1, m.nbody):
            if m.body_treeid[b] >= 0 and (b in self.robot or self.name(self.owner[b]) in names):
                awake[m.body_treeid[b]] = True
        return self.sleep(awake)

    def moved(self, x0, x1):
        """Per body: whether it moved in a settle that started at x0 and was at x1 1 s before its end (the init records'
        still rule: 1 cm in all, 1 mm in the last second)."""
        from sbv.functionality.f2_physical import STILL_MOVE, STILL_DRIFT

        d = self.d
        return (np.linalg.norm(d.xipos - x0, axis=1) > STILL_MOVE) | (
            np.linalg.norm(d.xipos - x1, axis=1) > STILL_DRIFT
        )

    def isolate(self, keep, x0, x1):
        m = self.m
        moved = self.moved(x0, x1)
        awake = np.zeros(m.ntree, bool)
        for b in range(1, m.nbody):
            if m.body_treeid[b] >= 0 and (b in self.robot or moved[b] or self.name(self.owner[b]) in keep):
                awake[m.body_treeid[b]] = True
        self.sleep(awake)

    def sleep(self, awake):
        """Every tree outside awake (a mask over trees) and outside their contact islands sleeps from the next step, at
        rest; the awake ones never do. A tree asleep already stays as it is. Returns the policies before."""
        m, d = self.m, self.d
        saved = m.tree_sleep_policy.copy()
        from sbv.functionality import sleep as SL

        if SL.AWAKE:
            return saved  # the whole room awake (sleep.awake): nothing sleeps
        m.tree_sleep_policy[awake] = NEVER
        m.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)
        m.opt.sleep_tolerance = 1e-4
        mujoco.mj_forward(m, d)
        isl = d.tree_island[: m.ntree]  # 1e-4 m/s, as sleep.py
        ready = ~awake & ((isl < 0) | ~np.isin(isl, isl[awake & (isl >= 0)])) & (d.tree_asleep[: m.ntree] < 0)
        d.tree_asleep[: m.ntree][ready] = -1
        d.qvel[ready[m.dof_treeid]] = 0.0
        mujoco.mj_forward(m, d)
        return saved

    def collides(self, g):
        return bool(self.m.geom_contype[g] or self.m.geom_conaffinity[g])

    def geoms(self, n):
        b = self.objects[n]
        return [g for g in range(self.m.ngeom) if self.owner[self.m.geom_bodyid[g]] == b]

    def aabb(self, n):
        from sbv.common.scene import Scene

        p = np.vstack([Scene.geom_points(self, g) for g in self.geoms(n)])
        return p.min(0), p.max(0)  # mesh vertices, not loose boxes

    def jitter(self, seed, sigma=0.001):
        if not seed:
            return
        rng = np.random.default_rng(seed)
        m, d = self.m, self.d
        for b in self.objects.values():
            j = m.body_jntadr[b]
            if j >= 0 and m.jnt_type[j] == 0:
                d.qpos[m.jnt_qposadr[j] : m.jnt_qposadr[j] + 2] += rng.uniform(-sigma, sigma, 2)
        mujoco.mj_forward(m, d)

    def move(self, offsets):
        """A nudge (calibrate._nudged, frozen trials): each named free object moved by (dx, dy) where it stands."""
        m, d = self.m, self.d
        for n, dxy in offsets.items():
            a = m.jnt_qposadr[m.body_jntadr[self.objects[n]]]
            d.qpos[a : a + 2] += dxy
        mujoco.mj_forward(m, d)


def nearby(scene, slots, margin=0.3):
    """Other objects whose boxes come within margin of a task's objects: the surroundings the task is done in. A copy
    that leaves one out does an easier task, so the task fails there (calibrate records them in params["near"])."""
    boxes = [scene.aabb(s) for s in slots]
    out = []
    for n in scene.objects:
        if n in slots:
            continue
        lo, hi = scene.aabb(n)
        if any((lo[:2] <= bh[:2] + margin).all() and (hi[:2] >= bl[:2] - margin).all() for bl, bh in boxes):
            out.append(n)
    return sorted(out)


def watched(w, unit, corr, tb):
    """Objects a robot task may not move: the task's other objects and its surroundings (params["near"], recorded at
    calibration), in the copy's names. An object elsewhere that drifts on its own is an at-rest failure, scored by F2,
    not by this task."""
    near = unit["params"].get("near")
    tbs = tb if isinstance(tb, set) else {tb}  # the task's own bodies: one, or an activity's
    if near is None:
        return {n: b for n, b in w.objects.items() if b not in tbs}  # a case from before near: the whole room
    names = set(unit["slots"]) | set(near)
    names = {n if corr is None else corr.get(n) for n in names}
    return {n: w.objects[n] for n in names if n in w.objects and w.objects[n] not in tbs}


def why(det):
    """A failed robot task's broken predicates, in words."""
    if det.get("reason"):
        return det["reason"]
    out = [
        f"ends {det['miss_cm']} cm from the mark" if not det.get("at_mark", True) else "",
        "does not end in the goal region" if not det.get("in_goal", True) else "",
        f"tipped {det.get('tilt_deg')} degrees" if not det.get("upright", True) else "",
        "still moving" if not det.get("still", True) else "",
        f"spine {det.get('spine_deg')} degrees from the bay's open side" if not det.get("spine_out", True) else "",
        "still in the hand" if not det.get("released", True) else "",
        f"took {det.get('seconds')} s" if not det.get("on_time", True) else "",
    ]
    out += [f"moved {n} {100 * v:.1f} cm" for n, v in (det.get("others_moved") or {}).items()]
    return "; ".join(x for x in out if x) or "failed"


def touches(w, tb):
    """Whether any part of the robot touches body tb now."""
    m, d = w.m, w.d
    return any(
        (w.owner[m.geom_bodyid[c.geom1]] == tb and m.geom_bodyid[c.geom2] in w.robot)
        or (w.owner[m.geom_bodyid[c.geom2]] == tb and m.geom_bodyid[c.geom1] in w.robot)
        for c in d.contact[: d.ncon]
    )


STAGES = dict(
    pick_place=(
        "reach the object",
        "grasp",
        "lift",
        "reach the goal",
        "set it down upright in the goal",
        "let go, still, nothing else moved",
    ),
    push=("reach the push path", "touch it", "at the mark", "upright", "let go, still, nothing else moved"),
    open=("find the handle", "reach the handle", "half way", "the whole way", "let go, still, nothing else moved"),
    press=("reach the part", "pressed down", "back up, nothing else pressed, let go"),
    shelve=(
        "reach the book",
        "push the spine out",
        "pinch the spine",
        "reach the bay",
        "stand it in the bay, upright, spine out",
        "let go, still, nothing else moved",
    ),
)  # shelving in six (before, one stage); no "start" stage:
# starting needs only the objects and some floor, which every copy with the objects has (geometry already scores that)
STAGES["close"] = STAGES["open"]


def stages(kind, det):
    """[(stage, passed)] of one move from its details (perform()'s det), in order. A move that succeeded passes every
    stage."""
    ok, r = bool(det.get("success")), det.get("reason") or ""
    settle = bool(det.get("released") and det.get("still") and not det.get("others_moved") and det.get("on_time"))
    v = dict(
        pick_place=[
            "reaches the object" not in r,
            det.get("grasp"),
            det.get("lift"),
            "in_goal" in det,
            det.get("in_goal") and det.get("upright"),
            settle,
        ],
        push=[
            "push path" not in r,
            det.get("push_force_N", 0.0) > 0.05,
            det.get("at_mark"),
            det.get("upright"),
            settle,
        ],
        open=[
            "moved" in det or "no reachable base pose" in r,
            "moved" in det,
            "moved" in det and (det["moved"] >= 0.5 * det.get("amount", 0.0) or det.get("done")),
            det.get("done"),
            settle,
        ],
        press=["down" in det, det.get("pressed"), ok],
        shelve=[
            "lie flat" not in r and "reaches the push" not in r,
            "overhangs" not in r,
            "book's spine" not in r,
            "reaches the bay" not in r,
            bool(det.get("in_goal") and det.get("upright") and det.get("spine_out")),
            settle,
        ],
    )[{"close": "open"}.get(kind, kind)]
    return [(s, True if ok else bool(x)) for s, x in zip(STAGES[kind], v)]


ACTIVITY_CAP = 12.0  # an activity is worth at most 12 points, two pick-move-places.


def capped(got, total):
    """(points earned, points possible) of an activity within ACTIVITY_CAP: its checkpoints scaled down to fit."""
    return (got * ACTIVITY_CAP / total, ACTIVITY_CAP) if total > ACTIVITY_CAP else (got, total)


def task_points(unit, det):
    """(checkpoints passed, checkpoints in all) of a robot task: its stages passed in order, a stage counting only when
    every stage before it passed; an activity adds up its steps, the failed step counting its leading stages and the
    steps after it none, at most ACTIVITY_CAP points. A task that never started passes none."""
    lead = lambda st: next((i for i, (_, x) in enumerate(st) if not x), len(st))
    if unit["kind"] != "robot_activity":
        kind = dict(robot="pick_place", robot_push="push", robot_shelve="shelve")[unit["kind"]]
        return (lead(stages(kind, det)) if det.get("started") else 0), len(STAGES[kind])
    done = det.get("steps", []) if det.get("started") else []
    kinds = [
        (s or {}).get("kind") or (done[i].get("kind") if i < len(done) else None)
        for i, s in enumerate(unit["params"]["steps"])
    ]
    total = sum(len(STAGES[k]) if k in STAGES else 1 for k in kinds)
    return capped(sum(lead(stages(s["kind"], s)) for s in done if s.get("kind") in STAGES), total)


def task_score(unit, det):
    """A robot task's score, 0 to 1: the share of its checkpoints passed in order (task_points()). A task that never
    started (an object or its surroundings missing, no floor to start on) scores 0."""
    if not det.get("started"):
        return 0.0
    got, total = task_points(unit, det)  # an activity: its checkpoints passed over all its checkpoints
    return got / total if total else 0.0


PASS_M, PASS_LOW = (
    0.56,
    0.5,
)  # passed by the base: its box within 0.56 m of the base's path (half its 0.69 x 0.60 m diagonal
# and 10 cm) and its bottom under 0.5 m, so it stands where the base drives
NEAR_M = 1.0  # driven past counts only within 1 m of the task's own objects
AWAKE_M = NEAR_M  # a frozen trial's settle moves what lies this near the task's objects (around)


def around(w, names, margin=None):
    """The objects whose boxes (each geom's bounding sphere, as Scene.focus) come within margin (AWAKE_M) of a named
    object's box now (a 3-D gap), the named ones included: the task's zone, which a changed frozen trial settles
    (World.freeze). nearby and passed_by measure gaps in x and y only, so pick_place.setup adds params.near and
    params.needed to it by name rather than rely on this radius; one the task touches farther off wakes on contact."""
    m, d = w.m, w.d
    margin = AWAKE_M if margin is None else margin
    go = w.owner[m.geom_bodyid]
    r = m.geom_rbound[:, None]
    lo, hi = np.full((m.nbody, 3), np.inf), np.full((m.nbody, 3), -np.inf)
    np.minimum.at(lo, go, d.geom_xpos - r)
    np.maximum.at(hi, go, d.geom_xpos + r)
    ks = [w.objects[n] for n in names if n in w.objects]
    gap = lambda a, b: float(np.linalg.norm(np.maximum(0, np.maximum(lo[a] - hi[b], lo[b] - hi[a]))))
    return set(names) | {n for n, b in w.objects.items() if any(gap(b, k) <= margin for k in ks)}


def boxes(w, names):
    """The boxes of the named objects now: a task's objects before it runs, for passed_by."""
    return [w.aabb(n) for n in names if n in w.objects]


def passed_by(w, ex, slots=(), before=()):
    """The room's objects a task needs: those the robot touched, and those it drove past (PASS_M, PASS_LOW) within
    NEAR_M of the task's own objects (slots), where they were before the run (before: boxes(w, slots) then) or where
    they are now, in the run just done. Only these can cancel the task in a copy that lacks them. Before, every object
    within 30 cm of the task's objects did, so a missing picture on the wall above the console cancelled three tasks;
    and with everything driven past, one side table missing in the middle of the room cancelled all four tasks of a
    copy, although the drive itself is not scored."""
    roots = {b: n for n, b in w.objects.items()}
    out = {roots[b] for b in ex.touched if b in roots}
    pts = np.array(ex.track or [ex.base[:2]], float)
    boxes_ = list(before) + boxes(w, slots)
    gap = lambda lo, hi, bl, bh: float(np.linalg.norm(np.maximum(0, np.maximum(bl[:2] - hi[:2], lo[:2] - bh[:2]))))
    for n in w.objects:
        lo, hi = w.aabb(n)
        if (
            lo[2] < PASS_LOW
            and (np.linalg.norm(np.maximum(0, np.maximum(lo[:2] - pts, pts - hi[:2])), axis=1) <= PASS_M).any()
            and (not boxes_ or min(gap(lo, hi, bl, bh) for bl, bh in boxes_) <= NEAR_M)
        ):
            out.add(n)
    return sorted(out)


def missing_near(unit, corr):
    """The reason a task cannot be compared in a copy, or None: an object it needs is missing (params "needed": what the
    robot touched or drove past in the reference room; a case from before keeps "near", everything within 30 cm)."""
    p = unit["params"]
    gone = [s for s in p.get("needed", p.get("near", [])) if corr is not None and corr.get(s) is None]
    return f"objects around the task are missing in the copy: {', '.join(gone)}" if gone else None


def edit_physics(world, body, edit):
    """A plausible copy error, for the hint-only check: the target's mass scaled by edit["mass"] and the friction of
    every geom in the room shifted by edit["friction"] (the robot's own geoms untouched)."""
    m = world.m
    links = [b for b in range(m.nbody) if world.owner[b] == body]
    m.body_mass[links] *= edit["mass"]
    m.body_inertia[links] *= edit["mass"]
    for g in range(m.ngeom):
        if m.geom_bodyid[g] not in world.robot:
            m.geom_friction[g, 0] = max(0.05, m.geom_friction[g, 0] + edit["friction"])
    d = world.d
    state = (d.qpos.copy(), d.qvel.copy(), d.ctrl.copy())
    mujoco.mj_setConst(m, d)  # recomputes the model's constants, but also resets qpos to qpos0: restore it
    d.qpos[:], d.qvel[:], d.ctrl[:] = state
    mujoco.mj_forward(m, d)
    world.edited = True


def floor_map(scene, bounds):
    """Cells blocked by obstacles OR missing ground, before oriented-base erosion."""
    from sbv.geometry.g3_layout import occupancy, ground_mask

    xs, ys, occ = occupancy(scene, bounds)
    return xs, ys, occ | ~ground_mask(scene, xs, ys)


def yaw_maps(occ):
    """free[k, i, j]: the base's box (0.69 m by 0.60 m plus a 4 cm margin) at heading k * 22.5 degrees, centred on cell
    (i, j), covers no occupied cell and stays on the map."""
    out = np.zeros((NYAW,) + occ.shape, bool)
    r = int(math.ceil(0.5 / CELL))
    for k in range(NYAW):
        c, s = math.cos(2 * math.pi * k / NYAW), math.sin(2 * math.pi * k / NYAW)
        K = np.zeros((2 * r + 1, 2 * r + 1))
        for a in range(-r, r + 1):
            for b in range(-r, r + 1):
                u, v = c * a * CELL + s * b * CELL, -s * a * CELL + c * b * CELL  # cell offset in the base frame
                K[a + r, b + r] = -0.40 <= u <= 0.37 and -0.34 <= v <= 0.34
        out[k] = ndimage.correlate(occ.astype(int), K, mode="constant", cval=1) == 0
    return out


def distances(free, src):
    """Breadth-first step counts over (heading, cell) states from src: slide to any of the 8 neighbouring cells at the
    same heading, or turn in place by one heading step; -1 where unreachable."""
    dist = np.full(free.shape, -1, int)
    if not free[src]:
        return dist
    dist[src] = 0
    q = deque([src])
    K, I, J = free.shape
    while q:
        k, i, j = q.popleft()
        n = dist[k, i, j] + 1
        for a, b, c in [(k, i + di, j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj] + [
            ((k + 1) % K, i, j),
            ((k - 1) % K, i, j),
        ]:
            if 0 <= b < I and 0 <= c < J and free[a, b, c] and dist[a, b, c] < 0:
                dist[a, b, c] = n
                q.append((a, b, c))
    return dist


def path(xs, ys, dist, goal):
    """States from the source to goal, walked back downhill, as base poses; runs of slides at one heading are merged."""
    K, I, J = dist.shape
    st = [goal]
    while dist[st[-1]] > 0:
        k, i, j = st[-1]
        nb = [(k, i + di, j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj] + [
            ((k + 1) % K, i, j),
            ((k - 1) % K, i, j),
        ]
        st.append(
            min(
                (s for s in nb if 0 <= s[1] < I and 0 <= s[2] < J and 0 <= dist[s] == dist[st[-1]] - 1), key=lambda s: s
            )
        )
    st = st[::-1]
    out = [st[0]]
    for a, b, c in zip(st, st[1:], st[2:]):
        if not (a[0] == b[0] == c[0] and (b[1] - a[1], b[2] - a[2]) == (c[1] - b[1], c[2] - b[2])):
            out.append(b)
    out.append(st[-1]) if len(st) > 1 else None
    return [(float(xs[i]), float(ys[j]), 2 * math.pi * k / K) for k, i, j in out]


def state_of(xs, ys, base):
    i = int(np.clip(round((base[0] - xs[0]) / CELL), 0, len(xs) - 1))
    j = int(np.clip(round((base[1] - ys[0]) / CELL), 0, len(ys) - 1))
    return int(round(base[2] / (2 * math.pi / NYAW))) % NYAW, i, j


def nearest_free(xs, ys, free, p, within=0.5):
    """The free state at heading 0 nearest p, if one lies within `within` metres; else None."""
    cells = np.argwhere(free[0])
    if not len(cells):
        return None
    pts = np.c_[xs[cells[:, 0]], ys[cells[:, 1]]]
    k = int(np.linalg.norm(pts - np.asarray(p[:2]), axis=1).argmin())
    return [float(pts[k, 0]), float(pts[k, 1])] if np.linalg.norm(pts[k] - np.asarray(p[:2])) <= within else None


def yawed(rot, angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ rot


def stand(world, xs, ys, free, dist, target, poses, rot, turn_with_base=None):
    """The nearest reachable base pose (on the grid, at one of the 16 headings) from which every fingertip pose in
    `poses` is reached within 1 cm. Returns (base (x, y, yaw), joint angles per pose, state, hand orientation) or
    None."""
    cands = []
    for k in range(NYAW):
        yaw = 2 * math.pi * k / NYAW
        R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        for f in FORWARD:
            st = state_of(xs, ys, (*(np.asarray(target[:2]) - R @ [f, LATERAL]), yaw))
            if free[st] and dist[st] >= 0:
                cands.append((dist[st], f, st))
    for _, _, st in sorted(set(cands))[:40]:
        base = (float(xs[st[1]]), float(ys[st[2]]), 2 * math.pi * st[0] / NYAW)
        if turn_with_base is not None:
            options = [yawed(rot, base[2] - turn_with_base)]  # a held object turns with the body
        else:
            options = [
                rot,
                rot @ np.diag([-1.0, -1.0, 1.0]),
            ]  # an empty two-finger hand works the same turned half round its own axis
        for r in options:
            q, qs = np.zeros(13), []
            for p in poses:
                q, err = world.rb.ik(base, np.asarray(p, float), r, q, iters=600)
                if err > 0.01:
                    break
                qs.append(q)
            else:
                return base, qs, st, r
    return None
