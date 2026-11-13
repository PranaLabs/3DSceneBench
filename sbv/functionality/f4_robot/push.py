"""F4 contact-rich task (doc Appendix C.3, "contact-rich execution"): the RB-Y1 pushes a small object 20 cm across the
top it stands on, to a mark, keeping contact the whole way. The hand leans back (40 degrees, or 30 or 20 where the arm
cannot reach that far) so its palm stays behind the object, its fingertips act as a smooth pusher (friction 0.1), and
they are spread across the object's back, as low as they go: an object pushed between two fingertips centres itself
on them, so the fingertips steer it. The push is closed-loop: every 20 ms the fingertips are re-aimed 1 cm into the
object's back on the straight line to the mark (deeper, 1 mm per tick, while the object lags), at most 8 cm/s, slowing
over the last 4 cm, and the push stops when the object has travelled 20 cm along the line.
What decides the outcome is contact the whole time: an object that is glued, far too heavy, or so grippy that friction
x push height beats its base, does not reach the mark or tips; a tall narrow one tips; one placed near an edge falls.
The mean force the hand applied over the middle half of the push is reported (push_force_N; friction x weight for a
sliding object), so a wrong weight or friction shows even when the task still succeeds. Passes when the object ends
within 3 cm of the mark, upright (under 15 degrees from its start), still, out of contact with the robot, nothing else
moved more than 2 cm, no tunnelling, within the deadline. The mark is placed relative to the object's own start, so the
task tests the push, not exactly where the object was put; the direction is the reference's, in room axes."""

import math
import numpy as np, mujoco
from sbv.functionality.integrity import Monitor
from sbv.functionality.f4_robot import plan as RP
from sbv.functionality.f4_robot.pick_place import Executor, NoRollout, DEADLINE, setup

DIST, TOL, SPEED, TICK, PUSH_MAX, GAP, LEAD, PRESS = 0.20, 0.03, 0.08, 0.02, 15.0, 0.015, 0.03, 0.01
PAD = 0.1  # hand friction for the push: a smooth pusher, so the hand presses only sideways
TILTS = (40, 30, 20)  # hand lean back, degrees, steepest the arm reaches: at 40 the RB-Y1 palm (a 3.6 cm collision
# sphere) stays 1.8 cm behind the fingertips, at 20 it is level with them


def half_along(ext, u):
    return 0.5 * (abs(ext[0] * u[0]) + abs(ext[1] * u[1]))


def leaning(u, angle):
    """Fingers spread across u, pointing down, the palm leaned back (against u) by `angle`."""
    v, up = np.array([-u[1], u[0], 0.0]), np.array([0.0, 0.0, 1.0])
    u3 = np.array([u[0], u[1], 0.0])
    return np.column_stack(
        [v, -math.cos(angle) * u3 - math.sin(angle) * up, math.cos(angle) * up - math.sin(angle) * u3]
    )


def hand_reach(w, rot, u):
    """How far the right fingertips' collision shapes reach ahead along u and below the tool point when the hand has
    orientation rot (read from the model, where the RB-Y1 fingertips collide as small spheres)."""
    m, d = w.m, w.d
    R = d.site_xmat[w.rb.tcp].reshape(3, 3)
    front = below = 0.0
    for g in range(m.ngeom):
        if w.collides(g) and "ee_finger_r" in w.name(m.geom_bodyid[g]):
            off = rot @ (R.T @ (d.geom_xpos[g] - d.site_xpos[w.rb.tcp]))
            rb = m.geom_rbound[g]
            front, below = max(front, off[:2] @ u + rb), max(below, rb - off[2])
    return front, below


def hand_force(w, tb, u):
    """Force the robot applies to the object along u at this instant, summed over their contacts (newtons)."""
    m, d = w.m, w.d
    f, tot = np.zeros(6), 0.0
    con = d.contact[: d.ncon]
    b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
    rob = list(w.robot)
    hit = ((w.owner[b1] == tb) & np.isin(b2, rob)) | ((w.owner[b2] == tb) & np.isin(b1, rob))
    for i in np.flatnonzero(hit):  # the robot-object contacts, in MuJoCo's order: the same sum
        mujoco.mj_contactForce(m, d, int(i), f)
        tot += abs(float((con.frame[i].reshape(3, 3).T @ f[:3])[:2] @ u))
    return tot


def run_push(xml, unit, corr=None, seed=0, room_scene=None, edit=None):
    p = unit["params"]
    obj = unit["slots"][0] if corr is None else corr.get(unit["slots"][0])
    if obj is None:
        return dict(success=False, reason="the object is missing"), None
    if RP.missing_near(unit, corr):
        return dict(success=False, reason=RP.missing_near(unit, corr)), None
    r = setup(xml, unit, seed, room_scene, edit, [obj], corr)
    if isinstance(r, str):
        return dict(success=False, reason=r), None
    w, xs, ys, free, start = r
    tb = w.objects[obj]
    others = {n: w.d.xipos[b].copy() for n, b in RP.watched(w, unit, corr, tb).items()}
    mon = Monitor(w, [tb], ignore=w.robot_roots)
    ex = Executor(w, start, mon)
    b0 = RP.boxes(w, unit["slots"])
    det = perform(w, ex, xs, ys, free, dict(object=obj, dir=p["dir"]), others)
    det["started"] = True
    if corr is None:
        det["passed_by"] = RP.passed_by(w, ex, unit["slots"], b0)
    return det, (mon if ex.t else None)


def perform(w, ex, xs, ys, free, st, others):
    """One push from where the robot stands, as a step of an activity or the whole task: st names the object and dir."""
    m, d = w.m, w.d
    obj = st["object"]
    tb = w.objects[obj]
    t0 = ex.t
    dist0 = RP.distances(free, RP.state_of(xs, ys, ex.base))
    for g in range(m.ngeom):  # higher contact priority: the pad's friction applies whatever the object's
        if w.collides(g) and ("ee_finger" in w.name(m.geom_bodyid[g]) or "EE_BODY" in w.name(m.geom_bodyid[g])):
            m.geom_priority[g], m.geom_friction[g] = 1, [PAD, 0.005, 0.0001]
    lo, hi = w.aabb(obj)
    ext = hi - lo
    u = np.asarray(st["dir"], float)
    u /= np.linalg.norm(u)
    side = np.array([-u[1], u[0]])
    p0 = d.xipos[tb][:2].copy()
    c = (lo[:2] + hi[:2]) / 2
    mark = p0 + DIST * u
    r = half_along(ext, u)
    span = float(np.clip(0.6 * 2 * half_along(ext, side), 0.02, 0.09))  # two contacts either side of the centre
    fork = -(span - 0.006) / 2  # gripper setting: fingertip spacing is 6 mm + 2 x |setting|
    for lean in TILTS:
        rot = leaning(u, math.radians(lean))
        front, below = hand_reach(w, rot, u)
        h = lo[2] + below + 0.008  # as low as the fingertips go
        a0 = np.r_[c - (r + front + GAP) * u, h]
        a1 = a0 + np.r_[DIST * u, 0]
        st = RP.stand(w, xs, ys, free, dist0, np.r_[c, h], [a0 + [0, 0, 0.10], a0, (a0 + a1) / 2, a1], rot)
        if st is not None:
            break
    if st is None:
        return dict(success=False, reason="no reachable base pose reaches the push path")
    base, rot = st[0], st[3]
    R0 = d.xmat[tb].reshape(3, 3).copy()
    ex.go(0.8, grip=fork)
    ex.drive(RP.path(xs, ys, dist0, st[2]))
    q_carry, _ = w.rb.ik(base, RP.carry_point(base), rot, st[1][0], iters=800)
    ex.go(2.0, arm=[q_carry])
    ex.line(base, RP.carry_point(base), a0 + [0, 0, 0.10], rot, seconds=2.0)
    ex.line(base, a0 + [0, 0, 0.10], a0, rot)
    q, t, cmd, forces, press, last = ex.q, 0.0, a0[:2].copy(), [], PRESS, 0.0
    while t < PUSH_MAX:  # closed loop: re-aim behind the object every tick
        done = float((d.xipos[tb][:2] - p0) @ u)
        if done >= DIST - 0.003:
            break  # the object has travelled the full distance along the line
        vmax = min(SPEED, max(0.02, 2.0 * (DIST - done)))  # slow down near the end
        if done - last < 0.25 * vmax * TICK and (
            forces and forces[-1] > 0.05 or np.linalg.norm(c + (done - r - front + press) * u - cmd) < 0.002
        ):
            press = min(
                press + 0.001, r + PRESS
            )  # lagging once touching or arrived: press in (it may sit behind its box)
        aim = c + (done - r - front + press) * u
        last = done  # on the line, just into the object's back
        step = aim - cmd
        if np.linalg.norm(step) > vmax * TICK:
            step *= vmax * TICK / np.linalg.norm(step)
        cmd = cmd + step
        tip = d.site_xpos[w.rb.tcp][:2]
        lead = cmd - tip  # the command runs ahead of the fingers so
        if np.linalg.norm(lead) > LEAD:
            cmd = tip + lead * LEAD / np.linalg.norm(lead)  # the arm pushes, by at most LEAD
        q, _ = w.rb.ik(base, np.r_[cmd, h], rot, q, iters=40)
        ex.go(TICK, arm=[q])
        t += TICK
        forces.append(hand_force(w, tb, u))
    tip = d.site_xpos[w.rb.tcp].copy()
    back = np.r_[tip[:2] - 0.03 * u, h]
    ex.line(base, tip, back, rot, k=3, seconds=0.6)
    ex.line(base, back, back + [0, 0, 0.12], rot, seconds=1.0)
    ex.line(base, back + [0, 0, 0.12], RP.carry_point(base), rot, seconds=1.5)
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
    miss = float(np.linalg.norm(p1[:2] - mark))  # mark: the object's start plus 20 cm along the line
    det = dict(
        miss_cm=round(100 * miss, 1),
        at_mark=miss <= TOL,
        tilt_deg=round(tilt, 1),
        upright=tilt < 15,
        still=bool(np.linalg.norm(vel[3:]) < 0.01),
        released=not touching,
        others_moved=moved,
        lean_deg=lean,
        push_seconds=round(t, 2),
        push_force_N=round(float(np.mean(forces[len(forces) // 4 : 3 * len(forces) // 4 + 1])), 2)
        if forces
        else 0.0,  # mean over the middle half
        seconds=round(ex.t - t0, 1),
        on_time=ex.t - t0 <= DEADLINE,
    )
    det["success"] = bool(
        det["at_mark"] and det["upright"] and det["still"] and det["released"] and not moved and det["on_time"]
    )
    return det


def trial(scene, unit, corr=None, seed=0, edit=None):
    det, mon = run_push(scene.xml, unit, corr, seed, scene, edit)
    if mon is None:
        mon = NoRollout(det)
    mon.details = det
    return [det["success"]], [mon]


def push_direction(scene, ctx, n):
    """Pushes go face-on, along the object's own axes (a corner pushed between two fingertips wedges), where the top has
    a clear straight strip: 20 cm past the object plus 5 cm, room behind it for the fingers, 1 cm beside it. Of the
    clear sides, the one with the most base in front of the balance point wins (a push tips the object when friction x
    push height exceeds that distance), then the one most toward the top's centre. Returns [x, y] or None."""
    from sbv.functionality.f2_calibrate import on_top

    lo, hi = scene.aabb(n)
    ext = hi - lo
    sup = [s for s in ctx["under"].get(n, []) if s != "world"]
    if not sup:
        return None
    top = scene.aabb(sup[0])[1][2]
    c = (lo + hi) / 2
    sc = scene.center(sup[0])[:2]
    best = None
    p = np.vstack([scene.geom_points(g) for g in scene.ogeoms[n] if scene.collides(g)])
    foot = (
        p[p[:, 2] < lo[2] + 0.003][:, :2] - scene.d.subtree_com[scene.objects[n]][:2]
    )  # base outline around the balance point
    R = scene.d.xmat[scene.objects[n]].reshape(3, 3)
    ax = R[:2, 0] if np.linalg.norm(R[:2, 0]) > 0.5 else R[:2, 1]
    ax = ax / np.linalg.norm(ax)
    for k in range(4):
        u = np.array([[0, -1], [1, 0]]) @ u if k else ax
        v = np.array([-u[1], u[0]])
        r, rv = half_along(ext, u), half_along(ext, v)
        pts = [
            c[:2] + a * u + b * v
            for a in np.arange(r + 0.01, r + DIST + 0.05, 0.02)
            for b in (-rv - 0.01, 0.0, rv + 0.01)
        ]
        pts += [c[:2] - (r + 0.03) * u + b * v for b in (-0.02, 0.0, 0.02)]
        if all(on_top(scene, sup[0], q, top) for q in pts):
            score = (round(float((foot @ u).max()) if len(foot) else r, 3), float(u @ (sc - c[:2])))
            if best is None or score > best[0]:
                best = (score, u)
    return None if best is None else best[1].round(4).tolist()


def candidates(scene, ctx, start, bounds):
    """Free objects 3.5 to 25 cm tall and at most 25 cm across, under 2 kg, standing on a top with 35 cm clear above,
    that have a push direction (push_direction)."""
    from sbv.functionality.f4_robot.pick_place import headroom

    scene.reset()
    xs, ys, occ = RP.floor_map(scene, bounds)
    s0 = RP.nearest_free(xs, ys, RP.yaw_maps(occ), start, within=3.0)
    if s0 is None:
        return []
    out = []
    for n in sorted(scene.objects):
        lo, hi = scene.aabb(n)
        ext = hi - lo
        if not (
            scene.free_dof(n) is not None
            and max(ext[:2]) <= 0.25
            and 0.035 <= ext[2] <= 0.25
            and scene.m.body_subtreemass[scene.objects[n]] <= 2
            and headroom(scene, n) >= 0.35
        ):
            continue
        u = push_direction(scene, ctx, n)
        if u:
            out.append(
                dict(
                    id=f"robot_push|{n}",
                    kind="robot_push",
                    slots=[n],
                    params=dict(dir=u, start=[s0[0], s0[1], 0.0], bounds=bounds),
                )
            )
    return out
