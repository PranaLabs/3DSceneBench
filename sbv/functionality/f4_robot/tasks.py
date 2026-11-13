"""Robot tasks a room's builder names (doc Appendix C.3), in tasks.json next to the reference scene (or calibrate
--tasks):
    {"pick_place": [{"object": "cup__0", "goal": "cabinet__0"}], "push": [{"object": "cup__1"}]}
A pick-and-place task may add "spot": [dx, dy], where on the goal's top to set the object (metres from the top's
centre); a push may add "direction": [x, y]. A shelving task, {"object": "book__19", "goal": "bookcase__0", "bay": [[x0,
y0, z0], [x1, y1, z1]], "spine": [x, y], "front": [x, y]}, names the bay (a box in room coordinates, its floor the
shelf's top), the way the book's spine faces as it lies, and the bay's open side (shelve.py). A shelving task without
them, or with a bay that is not a box, is refused with the reason, like a task that names a missing object (calibrate
proposes no shelving tasks of its own). Any task may add "words": how the entrants' instruction says it, in plain words
and without object ids, and "start": [x, y], where the robot starts for this task (the free floor
nearest it, within 0.5 m; default: the room's start), for a task done from floor the room's start cannot reach, such as
behind a counter. An activity, {"words": "...", "steps": [{"kind": "pick_place", "object": ..., "goal": ...}, {"kind":
"push", "object": ...}, ...]}, is a long-horizon task: its steps are tasks of the kinds above, done in order in one
unbroken run (activity.py). It needs words and at least one step; a step that cannot be set up refuses the whole
activity with the step's number and reason. Two more step kinds exist only inside an activity: {"kind": "open",
"object": "cabinet__0", "part": "drawer"} and "close" (open_close.py). "part" names the joint or its body, or a whole
word of either; it may be left out when the object has one moving part. "amount" (metres or radians, optional) is how
far to open. The part needs a handle or knob. A third, {"kind": "press", "object": "piano__0", "part":
"piano_key_060_slide"}, presses a key or button: a part that slides, found the same way (press.py).
Calibrate runs every named task in the reference like those, and keeps it only if it succeeds there and under jitter;
named tasks are kept even beyond the --robot-tasks and --push-tasks caps. A "package" family holds tasks the room brings
in its own code (sbv/package_tasks, F4.7); the robot tasks here leave it alone."""

import json, os
import numpy as np


def load(xml, given=None):
    path = given or os.path.join(os.path.dirname(os.path.abspath(xml)), "tasks.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def said(t):
    """The builder's own words for a task, kept in its unit for the entrants' instruction (make_tasks.room_tasks)."""
    return dict(words=str(t["words"]).strip()) if t.get("words") else {}


def shelve_problem(t):
    """Why a shelving task cannot be set up from its own fields, or None: it needs bay (a box, [[x0, y0, z0], [x1, y1,
    z1]] with each high corner above the low one), spine and front (non-zero [x, y])."""
    gone = [k for k in ("bay", "spine", "front") if k not in t]
    if gone:
        return f"a shelving task needs {', '.join(gone)}"
    b, s, f = (np.asarray(t[k], float) for k in ("bay", "spine", "front"))
    if b.shape != (2, 3) or not (b[1] > b[0]).all():
        return "bay must be a box [[x0, y0, z0], [x1, y1, z1]] with x1 > x0, y1 > y0 and z1 > z0"
    if s.shape != (2,) or f.shape != (2,) or not np.linalg.norm(s) or not np.linalg.norm(f):
        return "spine and front must be non-zero directions [x, y]"
    return None


def resolve(scene, ctx, kind, t, problem):
    """One task or step of a kind, from its own fields: (the objects it names, its parameters, None), or (.., ..,
    why)."""
    from sbv.functionality.f4_robot import pick_place as robot, push as robot_push

    o, g = t.get("object"), t.get("goal")
    why = problem(o)
    if kind == "pick_place":
        why = why or (f"no object named {g}" if g not in scene.objects else None)
        rel = None if why else t.get("spot") or robot.goal_spot(scene, g)
        return (
            [o, g],
            dict(rel=rel and list(rel)),
            why or (None if rel else f"{g} has no clear 20 cm square on its top"),
        )
    if kind == "push":
        u = None if why else t.get("direction") or robot_push.push_direction(scene, ctx, o)
        if u is not None:
            u = np.asarray(u, float) / np.linalg.norm(u)
        return (
            [o],
            dict(dir=None if u is None else u.round(4).tolist()),
            why or (None if u is not None else "no clear straight 20 cm strip of its top along any of its sides"),
        )
    if kind == "shelve":
        why = why or (f"no object named {g}" if g not in scene.objects else None) or shelve_problem(t)
        return [o, g], dict(bay=t.get("bay"), spine=t.get("spine"), front=t.get("front")), why
    if kind in ("open", "close"):
        from sbv.functionality.f4_robot import open_close as OC

        if o not in scene.objects:
            return [o], {}, f"no object named {o}"
        j, why = OC.find_joint(scene, o, t.get("part"))
        if j is None:
            return [o], {}, why
        ref, u = OC.record(scene, o, j)
        h, why = OC.handle(scene, o, j, u)
        return [o], dict(ref=ref, amount=t.get("amount")), why
    if kind == "press":  # a key or button: a part that slides (press.py)
        from sbv.functionality.f4_robot import open_close as OC

        if o not in scene.objects:
            return [o], {}, f"no object named {o}"
        j, why = OC.find_joint(scene, o, t.get("part"))
        if j is None:
            return [o], {}, why
        if scene.m.jnt_type[j] != OC.SLIDE:
            return [o], {}, f"{OC.jname(scene.m, j)} does not slide, so it cannot be pressed"
        return [o], dict(ref=OC.record(scene, o, j)[0]), None
    return [], {}, f"unknown step kind {kind!r}: one of {', '.join(KINDS)}, open, close, press"


KINDS = {"pick_place": "robot", "push": "robot_push", "shelve": "robot_shelve"}  # tasks.json key -> unit kind


def named(scene, ctx, spec, start, bounds):
    """Units for the named tasks, and (id, reason) for each that cannot even be set up."""
    from sbv.functionality.f4_robot import plan as RP

    scene.reset()
    units, bad = [], []
    xs, ys, occ = RP.floor_map(scene, bounds)
    free = RP.yaw_maps(occ)
    room_s0 = RP.nearest_free(xs, ys, free, start, within=3.0)

    def problem(n):
        return (
            f"no object named {n}"
            if n not in scene.objects
            else f"{n} has no free joint, so it cannot move"
            if scene.free_dof(n) is None
            else None
        )

    def where(t):
        s0 = RP.nearest_free(xs, ys, free, list(t["start"]) + [0.0], within=0.5) if "start" in t else room_s0
        return s0, (
            None
            if s0
            else f"no free floor for the robot within {'0.5 m of the task' if 'start' in t else '3 m of the room'}'s start"
        )

    def keep(uid, kind, slots, params, t, why):
        s0, w2 = where(t)
        if why or w2:
            bad.append((uid, f"named task: {why or w2}"))
            return
        units.append(
            dict(
                id=uid,
                kind=kind,
                slots=slots,
                params=dict(params, start=[s0[0], s0[1], 0.0], bounds=bounds),
                named=True,
                **said(t),
            )
        )

    for key, kind in KINDS.items():
        for t in spec.get(key, []):
            slots, params, why = resolve(scene, ctx, key, t, problem)
            keep("|".join([kind] + [str(x) for x in slots]), kind, slots, params, t, why)
    for i, t in enumerate(spec.get("activity", [])):
        steps, slots, why = [], [], None if t.get("words") else "an activity needs words"
        for st in (t.get("steps") or []) if why is None else []:
            ss, params, why = resolve(scene, ctx, st.get("kind"), st, problem)
            if why:
                why = f"step {len(steps) + 1} ({st.get('kind')}): {why}"
                break
            steps.append(dict(st, **params))
            slots += [x for x in ss if x not in slots]
        keep(
            f"robot_activity|{i}",
            "robot_activity",
            slots,
            dict(steps=steps),
            t,
            why or (None if steps else "an activity needs at least one step"),
        )
    return units, bad
