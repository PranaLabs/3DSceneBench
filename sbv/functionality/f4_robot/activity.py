"""F4.5 activity (one per named task): a room's own long-horizon task, several moves done in order in one unbroken run.
The robot keeps its place, its arm and whatever it changed between the steps; nothing is reset or moved by hand. Each
step is one move the rulebook has (pick_place, push, shelve, open, close, press), with the fields its own task kind
takes (tasks.py); the step list, in the reference room's names, is frozen in the unit at calibration. A step that fails
ends the activity with that step's reason; the activity passes only if every step passes in order, with the same checks
each move makes on its own (object where it should be, released, still, within the move's deadline), the objects around
the task moved no more than 2 cm over the whole run, no tunnelling, penetration within the unit's tolerance."""

from sbv.functionality.integrity import Monitor
from sbv.functionality.f4_robot import plan as RP, pick_place, push, shelve, open_close, press
from sbv.functionality.f4_robot.pick_place import Executor, NoRollout, setup

MOVES = dict(
    pick_place=pick_place, push=push, shelve=shelve, open=open_close, close=open_close, press=press
)  # step kind -> perform(w, ex, xs, ys, free, step, others)
NAMES = ("object", "goal")  # the fields of a step that name objects


def run_task(xml, unit, corr=None, seed=0, room_scene=None, edit=None):
    """Run the steps in one world; returns (details, Monitor or None). details["steps"] has each step's own details."""
    p = unit["params"]
    names = {s: (s if corr is None else corr.get(s)) for s in unit["slots"]}
    gone = [s for s, n in names.items() if n is None]
    if gone:
        return dict(success=False, reason=f"missing in the copy: {', '.join(gone)}"), None
    if RP.missing_near(unit, corr):
        return dict(success=False, reason=RP.missing_near(unit, corr)), None
    r = setup(xml, unit, seed, room_scene, edit, [names[s] for s in unit["slots"]], corr)
    if isinstance(r, str):
        return dict(success=False, reason=r), None
    w, xs, ys, free, start = r
    tbs = {w.objects[n] for n in names.values()}
    others = {n: w.d.xipos[b].copy() for n, b in RP.watched(w, unit, corr, tbs).items()}
    mon = Monitor(w, sorted(tbs), ignore=w.robot_roots)
    ex = Executor(w, start, mon)
    steps = []
    b0 = RP.boxes(w, unit["slots"])
    for k, st in enumerate(p["steps"]):
        st = dict(st, **{key: names[st[key]] for key in NAMES if key in st})  # the copy's names for this step
        det = MOVES[st["kind"]].perform(w, ex, xs, ys, free, st, others)
        steps.append(dict(kind=st["kind"], **det))
        if not det["success"]:
            return dict(
                success=False,
                started=True,
                steps=steps,
                failed_step=k + 1,
                seconds=round(ex.t, 1),
                reason=f"step {k + 1} ({st['kind']}): {RP.why(det)}",
            ), (mon if ex.t else None)
        if st["kind"] in ("open", "close"):
            xs, ys, free = floor_after(room_scene, w, p["bounds"])  # a doorway opened or shut
    return dict(
        success=True,
        started=True,
        steps=steps,
        seconds=round(ex.t, 1),
        **({"passed_by": RP.passed_by(w, ex, unit["slots"], b0)} if corr is None else {}),
    ), mon


def trial(scene, unit, corr=None, seed=0, edit=None):
    det, mon = run_task(scene.xml, unit, corr, seed, scene, edit)
    if mon is None:
        mon = NoRollout(det)
    mon.details = det
    return [det["success"]], [mon]


def floor_after(room_scene, w, bounds):
    """The floor maps with every door and drawer where the robot left them, so the next step's path can go through a
    doorway the robot opened. Taken on the room without the robot (its parts set from the robot's world by joint name),
    whose own start is put back after."""
    import mujoco

    m, keep = room_scene.m, room_scene.q0.copy()
    q = keep.copy()
    for j in range(m.njnt):
        if m.jnt_type[j] in (open_close.SLIDE, open_close.HINGE):
            jw = mujoco.mj_name2id(
                w.m, mujoco.mjtObj.mjOBJ_JOINT, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            )
            if jw >= 0:
                q[m.jnt_qposadr[j]] = w.d.qpos[w.m.jnt_qposadr[jw]]
    room_scene.q0 = q
    try:
        xs, ys, occ = RP.floor_map(room_scene, bounds)
        return xs, ys, RP.yaw_maps(occ)
    finally:
        room_scene.q0 = keep
        room_scene.reset(keep)
