"""F4 robot interaction (doc Appendix C.3): a general scripted pick-move-place controller for the RB-Y1. Declared
observations, read from the scene the robot runs in (privileged state, no camera): the target object's box and pose, the
goal surface's box, and the room's floor map. The controller drives to a stand spot, grasps from above across the
object's narrow side, lifts, carries the hand high, drives to a spot by the goal, sets the object down on a clear spot
of the goal's top that it reaches (the named one first), lets go and backs away. The task passes only if every predicate
holds: the object rests on the goal's top, anywhere on it, upright (under 15 degrees from its start), still, out of the
hand, no other object moved more than 2 cm, no tunnelling, penetration within the unit's tolerance (the hand's own
contacts with the object are allowed), within the deadline. The grasp closes across the object's narrow side, found from
its own footprint, so a copy that turns the object is gripped the same way."""

import math
import numpy as np, mujoco
from sbv.functionality.integrity import Monitor
from sbv.functionality.f4_robot import plan as RP
from sbv.geometry.g2_shape import oriented_box
from sbv.common.scene import Scene

OPEN, SHUT, SPEED, TURN, DEADLINE = -0.05, 0.0, 0.35, 0.8, 150.0
CARRY = 0.25  # m/s while the hand holds something.
# backing through the doorway on one of three nudged runs; at 0.25 it held
SPOT_STEP, SPOT_EDGE, SPOTS = 0.10, 0.03, 8  # the goal is anywhere the robot reaches on the goal's
# top: the named spot, then spots spread over it, 3 cm in from the edge, 8 at most


class Executor:
    def __init__(self, world, start, monitor):
        self.w, self.mon, self.t = world, monitor, 0.0
        self.base, self.q, self.grip = tuple(start), np.zeros(13), OPEN
        self.dt = world.m.opt.timestep
        self.touched, self.track, self.ticks = set(), [], 0  # for plan.passed_by
        self.robot_ids = np.array(sorted(world.robot))

    def seen(self, base):
        """Every 10 steps: the base's commanded place, and the bodies (their roots) the robot touches."""
        m, d, w, n = self.w.m, self.w.d, self.w, self.w.d.ncon
        self.track.append((float(base[0]), float(base[1])))
        if n:
            b1, b2 = m.geom_bodyid[d.contact.geom1[:n]], m.geom_bodyid[d.contact.geom2[:n]]
            r1, r2 = np.isin(b1, self.robot_ids), np.isin(b2, self.robot_ids)
            self.touched.update(w.owner[b2[r1 & ~r2]].tolist())
            self.touched.update(w.owner[b1[r2 & ~r1]].tolist())

    def go(self, seconds, base=None, arm=None, grip=None):
        """Move base, arm (a list of joint targets followed in order) and gripper together, with smooth timing."""
        m, d, rb = self.w.m, self.w.d, self.w.rb
        b0, g0 = np.array(self.base), self.grip
        b1 = np.array(base if base is not None else self.base)
        b1[2] = b0[2] + (b1[2] - b0[2] + math.pi) % (2 * math.pi) - math.pi
        g1 = self.grip if grip is None else grip
        pts = np.array([self.q] + list(arm if arm is not None else [self.q]))
        n = max(1, int(round(seconds / self.dt)))
        for k in range(n):
            u = (k + 1) / n
            u = u * u * (3 - 2 * u)
            x = u * (len(pts) - 1)
            i = min(int(x), len(pts) - 2)
            qa = pts[i] + (pts[i + 1] - pts[i]) * (x - i)
            for i_a, a in enumerate(rb.base):
                d.ctrl[a] = b0[i_a] + (b1[i_a] - b0[i_a]) * u
            for i_j, j in enumerate(rb.joints):
                d.ctrl[rb.act[j]] = qa[i_j]
            d.ctrl[rb.finger] = g0 + (g1 - g0) * u
            self.mon.before_step()
            mujoco.mj_step(m, d)
            self.mon.tick()
            self.ticks += 1
            if self.ticks % 10 == 0:
                self.seen(b0 + (b1 - b0) * u)
        self.t += n * self.dt
        self.base, self.q, self.grip = tuple(b1), pts[-1], g1

    def drive(self, poses, speed=SPEED):
        """Follow planned base poses: a change of heading is a turn in place, anything else a straight slide; speed in
        m/s (CARRY while the hand holds something)."""
        for x, y, yaw in poses[1:]:
            turn = abs((yaw - self.base[2] + math.pi) % (2 * math.pi) - math.pi)
            self.go(max(0.6, math.dist((x, y), self.base[:2]) / speed, turn / TURN), base=(x, y, yaw))

    def line(self, base, a, b, rot, k=8, seconds=1.5):
        qq, path = self.q, []
        for s in np.linspace(0, 1, k)[1:]:
            qq, _ = self.w.rb.ik(base, np.asarray(a) + s * (np.asarray(b) - np.asarray(a)), rot, qq, iters=300)
            path.append(qq)
        self.go(seconds, arm=path)


def lowest(world, n):
    return float(world.aabb(n)[0][2])


def top_z(w, goal, xy, above, skip=-1):
    """The height where a ray straight down at xy, from height above, first meets something, if that is the goal (skip:
    a body the ray passes through, the object carried); else None."""
    gid = np.zeros(1, np.int32)
    x = mujoco.mj_ray(
        w.m, w.d, np.array([xy[0], xy[1], above], float), np.array([0.0, 0.0, -1.0]), None, 1, int(skip), gid
    )
    return above - x if x >= 0 and w.owner[w.m.geom_bodyid[gid[0]]] == w.objects[goal] else None


def spots(w, goal, rel, half, skip=-1):
    """Where the object may be set down on the goal's top: the named spot (rel from the top's centre) first, then up to
    SPOTS - 1 others spread over the whole top, tried nearest the named spot first. The candidates lie on a SPOT_STEP
    grid; a spot counts when rays at its centre and at the corners of its footprint (half, plus SPOT_EDGE) all meet the
    goal first, at one height: nothing stands there and it is not past the edge. Returns [(x, y, z)]; [] when the top
    has no clear spot."""
    glo, ghi = w.aabb(goal)
    named = (glo[:2] + ghi[:2]) / 2 + np.asarray(rel[:2], float)
    r = half + SPOT_EDGE
    gx, gy = (np.arange(glo[i] + r, ghi[i] - r + 1e-9, SPOT_STEP) for i in (0, 1))

    def clear(p):
        zs = [top_z(w, goal, p + np.array(o), ghi[2] + 0.3, skip) for o in ((0, 0), (r, r), (r, -r), (-r, r), (-r, -r))]
        return np.r_[p, zs[0]] if None not in zs and max(zs) - min(zs) < 0.005 else None

    pool = [c for c in (clear(np.array([x, y])) for x in gx for y in gy) if c is not None]
    first = clear(named)
    picked = [first] if first is not None else []
    while pool and len(picked) < SPOTS:  # farthest from every spot picked so far (the named spot
        ref = [q[:2] for q in picked] or [named]  # counts even when blocked)
        k = max(range(len(pool)), key=lambda i: min(float(np.linalg.norm(pool[i][:2] - q)) for q in ref))
        picked.append(pool.pop(k))
    rest = sorted(picked[1:] if first is not None else picked, key=lambda q: float(np.linalg.norm(q[:2] - named)))
    return ([first] if first is not None else []) + rest


def setup(xml, unit, seed, room_scene, edit, keep, corr=None):
    """The room plus the robot, ready for a task: the floor maps, the start the task names (the free floor nearest it),
    the world jittered, edited for the hint-only check (the first of keep, the task's object) or nudged (edit {"nudge":
    {object: (dx, dy)}}, calibrate._nudged: the objects moved at the file's start) and settled, isolated on keep when
    the case says so. Returns (world, xs, ys, free, start), or the reason the robot has no floor to start on. Frozen
    trials (plan.mode): one that changes nothing in the room's model or the task's objects (the nominal run, or a jitter
    seed) starts from the whole-room settle of its start and seed, kept once and shared by every task of that start
    (World.restore): the same, bit for bit, as before. One that changes something for its task (an edit, or a nudge)
    applies it at the file's start, as before, then settles only its
    zone: the objects within AWAKE_M of keep, params.near and params.needed (corr maps them on a copy), the nudged ones,
    the objects the whole-room settle moves (World.restless) and what touches these (World.freeze); its task then runs
    as before: a nudged one isolated, an edited one not (what the frozen settle left asleep wakes when touched)."""
    p = unit["params"]
    xs, ys, occ = RP.floor_map(room_scene, p["bounds"])
    free = RP.yaw_maps(occ)
    s0 = RP.nearest_free(xs, ys, free, p["start"])
    if s0 is None:
        return "no free floor for the robot within 0.5 m of its start"
    cold, nudge = RP.mode(unit), (edit or {}).get("nudge") or {}
    edit = None if nudge else edit
    start = (s0[0], s0[1], 0.0)
    w = RP.World(xml, start, room_scene.textures, room_scene.phys)
    awake = None
    if cold and not (edit or nudge):
        w.jitter(seed)
        w.restore()  # the kept whole-room settle of this start and seed, bit for bit
    elif cold:  # by name: near and needed on a copy are its own objects
        named = [n if corr is None else corr.get(n) for n in p.get("near", []) + p.get("needed", [])]
        awake = w.restless() | {n for n in named if n in w.objects} | set(nudge)
        w.jitter(seed)
    else:
        w.jitter(seed)
    if edit:
        RP.edit_physics(w, w.objects[keep[0]], edit)
    if nudge:
        w.move(nudge)
    w.settle(
        set(keep)
        if p.get("isolate") and not edit
        else None,  # settle; then isolated if the case says so (an edit never: as before)
        None if awake is None else awake | RP.around(w, keep),
    )
    return w, xs, ys, free, start


def run_task(xml, unit, corr=None, seed=0, room_scene=None, edit=None):
    """Plan and execute one pick-move-place task; returns (details dict, Monitor or None)."""
    p = unit["params"]
    obj, goal = [s if corr is None else corr.get(s) for s in unit["slots"]]
    if obj is None or goal is None:
        return dict(success=False, reason="target or goal object missing"), None
    if RP.missing_near(unit, corr):
        return dict(success=False, reason=RP.missing_near(unit, corr)), None
    r = setup(xml, unit, seed, room_scene, edit, [obj, goal], corr)
    if isinstance(r, str):
        return dict(success=False, reason=r), None
    w, xs, ys, free, start = r
    tb = w.objects[obj]
    others = {n: w.d.xipos[b].copy() for n, b in RP.watched(w, unit, corr, tb).items()}
    mon = Monitor(w, [tb], ignore=w.robot_roots)
    ex = Executor(w, start, mon)
    b0 = RP.boxes(w, unit["slots"])
    det = perform(w, ex, xs, ys, free, dict(object=obj, goal=goal, rel=p["rel"]), others)
    det["started"] = True
    if corr is None:
        det["passed_by"] = RP.passed_by(w, ex, unit["slots"], b0)  # the reference room: what the task needs (calibrate)
    return det, (mon if ex.t else None)  # a plan that fails before anything moves has no rollout


def perform(w, ex, xs, ys, free, st, others):
    """One pick-move-place from where the robot stands, as a step of an activity or the whole task: st names the object,
    the goal and rel (where on the goal's top). Returns the step's details, success included; others: the watched
    objects' positions at the task's start."""
    rr = RP.rby1()
    m, d = w.m, w.d
    obj, goal = st["object"], st["goal"]
    tb = w.objects[obj]
    t0 = ex.t
    dist0 = RP.distances(free, RP.state_of(xs, ys, ex.base))
    pts = np.vstack(
        [Scene.geom_points(w, g) for g in w.geoms(obj) if w.collides(g)]
        or [Scene.geom_points(w, g) for g in w.geoms(obj)]
    )
    lo, hi = pts.min(0), pts.max(0)
    ext = hi - lo
    cb, ang, e = oriented_box(pts)
    c = np.r_[cb[:2], (lo[2] + hi[2]) / 2]
    closing = (math.cos(ang), math.sin(ang)) if e[0] <= e[1] else (-math.sin(ang), math.cos(ang))
    rot = rr.down_grasp(closing)  # across the narrow side
    z_grip = lo[2] + float(np.clip(0.5 * ext[2], 0.015, 0.04))
    pre, grasp, lift = [c[0], c[1], z_grip + 0.12], [c[0], c[1], z_grip], [c[0], c[1], z_grip + 0.15]
    pick = RP.stand(w, xs, ys, free, dist0, c, [pre, grasp, lift], rot)
    if pick is None:
        return dict(success=False, reason="no reachable base pose reaches the object")
    rot = pick[3]
    glo, ghi = w.aabb(goal)
    named = np.array([(glo[0] + ghi[0]) / 2 + st["rel"][0], (glo[1] + ghi[1]) / 2 + st["rel"][1], ghi[2]])
    dist1, place = RP.distances(free, pick[2]), None
    for gc in spots(w, goal, st["rel"], 0.5 * float(max(ext[0], ext[1])), tb) or [
        named
    ]:  # the first spot the robot reaches
        hold0 = gc[2] + (z_grip - lo[2]) + 0.002
        place = RP.stand(
            w,
            xs,
            ys,
            free,
            dist1,
            gc,
            [[gc[0], gc[1], hold0 + 0.10], [gc[0], gc[1], hold0]],
            rot,
            turn_with_base=pick[0][2],
        )
        if place is not None:
            break
    R0 = d.xmat[tb].reshape(3, 3).copy()
    bp = pick[0]
    got = {}
    q_carry, _ = w.rb.ik(bp, RP.carry_point(bp), rot, pick[1][2], iters=800)

    def pick_up():  # drive, grasp and lift; got: grasp (the hand touches it), lift (10 cm up)
        z0 = float(d.xipos[tb][2])
        ex.drive(RP.path(xs, ys, dist0, pick[2]))
        ex.go(2.0, arm=[q_carry])
        ex.line(bp, RP.carry_point(bp), pre, rot, seconds=2.0)
        ex.line(bp, pre, grasp, rot)
        ex.go(1.0, grip=SHUT)
        got["grasp"] = RP.touches(w, tb)
        ex.line(bp, grasp, lift, rot)
        got["lift"] = bool(got["grasp"] and float(d.xipos[tb][2]) - z0 >= 0.10)
        ex.line(bp, lift, RP.carry_point(bp), rot)

    if place is None:  # the goal is out of reach: the pick still earns its stages
        pick_up()
        return dict(success=False, reason="no reachable base pose reaches the goal", **got)
    rot_q = RP.yawed(rot, place[0][2] - pick[0][2])  # the hand keeps its grip angle relative to the body while carrying
    bq = place[0]
    pick_up()
    route = RP.path(xs, ys, dist1, place[2])
    ex.drive(route, CARRY)  # holding the object: slowly
    hold = gc[2] + (d.site_xpos[w.rb.tcp][2] - lowest(w, obj)) + 0.002  # set down 2 mm above the top, not dropped
    above, down = [gc[0], gc[1], hold + 0.10], [gc[0], gc[1], hold]
    ex.line(bq, RP.carry_point(bq), above, rot_q, seconds=2.0)
    ex.line(bq, above, down, rot_q)
    ex.go(0.8, grip=OPEN)
    ex.line(bq, down, above, rot_q, seconds=1.0)
    ex.line(bq, above, RP.carry_point(bq), rot_q, seconds=1.2)
    if len(route) > 1 and math.dist(route[-2][:2], bq[:2]) <= 0.6 and route[-2][2] == bq[2]:
        ex.go(1.5, base=route[-2])  # back out the way it came
    ex.go(1.5, arm=[np.zeros(13)])
    ex.go(2.0)
    p1 = d.xipos[tb]
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, tb, vel, 0)
    tilt = math.degrees(math.acos(np.clip(R0[:, 2] @ d.xmat[tb].reshape(3, 3)[:, 2], -1, 1)))
    touching = any(
        (w.owner[m.geom_bodyid[cn.geom1]] == tb and m.geom_bodyid[cn.geom2] in w.robot)
        or (w.owner[m.geom_bodyid[cn.geom2]] == tb and m.geom_bodyid[cn.geom1] in w.robot)
        for cn in d.contact[: d.ncon]
    )
    moved = {n: round(float(np.linalg.norm(d.xipos[w.objects[n]] - v)), 3) for n, v in others.items()}
    moved = {n: v for n, v in moved.items() if v > 0.02}
    zt = top_z(w, goal, p1[:2], ghi[2] + 0.3, tb)  # resting on the goal's top, anywhere on it: the goal is under its
    det = dict(
        in_goal=bool(zt is not None and abs(lowest(w, obj) - zt) < 0.015),
        tilt_deg=round(tilt, 1),  # centre, its bottom on it
        spot_cm=round(100 * float(np.linalg.norm(gc[:2] - named[:2])), 1),  # from the named spot
        upright=tilt < 15,
        still=bool(np.linalg.norm(vel[3:]) < 0.01),
        released=not touching,
        others_moved=moved,
        seconds=round(ex.t - t0, 1),
        on_time=ex.t - t0 <= DEADLINE,
    )
    det["success"] = bool(
        det["in_goal"] and det["upright"] and det["still"] and det["released"] and not moved and det["on_time"]
    )
    return dict(det, **got)


class NoRollout:
    """Stands in for a Monitor when planning fails before anything moves: nothing to check, the details still
    reported."""

    owners, pen = [], {}

    def __init__(self, details):
        self.details = details

    def ok(self, owners, tol):
        return True

    def ok_rules(self, owners, sink=True):
        return True

    def crossed(self, owners):
        return []

    def report(self, owners):
        return {}


def trial(scene, unit, corr=None, seed=0, edit=None):
    det, mon = run_task(scene.xml, unit, corr, seed, scene, edit)
    if mon is None:
        mon = NoRollout(det)
    mon.details = det
    return [det["success"]], [mon]


def headroom(scene, n):
    lo, hi = scene.aabb(n)
    gid = np.zeros(1, np.int32)
    c = (lo + hi) / 2
    x = mujoco.mj_ray(
        scene.m, scene.d, np.array([c[0], c[1], hi[2] + 0.002]), np.array([0.0, 0.0, 1.0]), None, 1, -1, gid
    )
    return float("inf") if x < 0 else float(x)


def goal_spot(scene, n):
    """A clear 20 cm square on n's top, as an offset from the top's centre, or None."""
    from sbv.functionality.f2_calibrate import on_top

    lo, hi = scene.aabb(n)
    c = (lo + hi) / 2
    for dx, dy in [
        (0, 0),
        (0.1, 0),
        (-0.1, 0),
        (0, 0.1),
        (0, -0.1),
        (0.15, 0.15),
        (-0.15, 0.15),
        (0.15, -0.15),
        (-0.15, -0.15),
    ]:
        if all(
            on_top(scene, n, c[:2] + [dx + sx * 0.1, dy + sy * 0.1], hi[2]) for sx in (-1, 0, 1) for sy in (-1, 0, 1)
        ):
            return [dx, dy]
    return None


def candidates(scene, ctx, start, bounds):
    """Objects a top-down two-finger grasp can take (free, at most 8.5 cm across, under 2 kg, resting on furniture with
    35 cm clear above), each paired with the nearest other top 35 cm to 1.3 m high that has a clear 20 cm spot."""
    scene.reset()
    objs, goals = [], []
    for n in sorted(scene.objects):
        lo, hi = scene.aabb(n)
        ext = hi - lo
        sup = [s for s in ctx["under"].get(n, []) if s != "world"]
        pts = np.vstack([scene.geom_points(g) for g in scene.ogeoms[n] if scene.collides(g)] or [np.zeros((1, 3))])
        e = oriented_box(pts)[2]
        if (
            scene.free_dof(n) is not None
            and min(e[:2]) <= 0.085
            and max(e[:2]) <= 0.3
            and 0.03 <= ext[2] <= 0.35
            and sup
            and scene.m.body_subtreemass[scene.objects[n]] <= 2
            and headroom(scene, n) >= 0.35
        ):
            objs.append((n, sup[0]))
        rel = goal_spot(scene, n) if 0.35 <= hi[2] <= 1.3 and min(ext[:2]) >= 0.25 else None
        if rel:
            goals.append((n, rel))
    xs, ys, occ = RP.floor_map(scene, bounds)
    s0 = RP.nearest_free(xs, ys, RP.yaw_maps(occ), start, within=3.0)
    if s0 is None:
        return []
    start, tasks = [s0[0], s0[1], 0.0], []
    for n, sup in objs:
        opts = sorted(
            (float(np.linalg.norm(scene.center(g)[:2] - scene.center(n)[:2])), g, rel)
            for g, rel in goals
            if g not in (sup, n)
        )
        if opts:
            tasks.append(
                dict(
                    id=f"robot|{n}|{opts[0][1]}",
                    kind="robot",
                    slots=[n, opts[0][1]],
                    params=dict(rel=opts[0][2], start=list(start), bounds=bounds),
                )
            )
    return tasks
