"""F4.6 press, as a step of an activity. The part is a slide joint on the object, found as open and close find theirs
(open_close.find_joint: by name, a whole word of it, or the reference room's record). One fingertip presses: the hand is
open (its 3 cm fingertips 10 cm apart), rolled so its fingertips stand one above the other: the upper one at least 3.5
cm higher and back towards the robot, its fingers leaned 30 degrees away from the robot along the part's longer side so
the wrist and torso stay back (the idle left arm folded up, TUCK). The fingertip comes to 6 cm short of the part's top,
touches it, pushes on 1 mm past the travel (deeper, a 3 cm fingertip pushes the neighbouring keys down too) and holds
0.3 s, then lifts back. Consecutive presses that the hand reaches from where the robot stands do not move the base.
Passes only if the part went down at least 80% of its travel, came back within 20% of it once released (a part with no
spring need not come back), no other moving part of the object went down more than half its own travel (no neighbouring
key pressed), the hand is off the object, nothing else moved over 2 cm, within the deadline."""

import math
import numpy as np, mujoco
from sbv.common.scene import Scene
from sbv.functionality.f4_robot import plan as RP, open_close as OC
from sbv.functionality.f4_robot.pick_place import OPEN, DEADLINE
from sbv.functionality.f4_robot.shelve import hand

FRAC, BACK, NEIGHBOUR, PAST, ABOVE, CLEAR, LONG, FRONT = 0.8, 0.2, 0.5, 0.001, 0.06, 0.035, 0.06, 0.03
LEAN, ROLL = (
    math.radians(30),
    math.radians(90),
)  # rolled 90: the fingertips one above the other in the lean   # fingers leaned away from the robot; the open hand rolled onto one fingertip
# The idle left arm, folded up: hanging at rest it reaches 0.70 m and a forward-leaning torso lays it on a keyboard.
# Folded, all of it stays above 1.3 m, within the base's width and 7 cm ahead of its centre (searched over the arm's
# ranges in the robot model).
TUCK = dict(left_arm_0=1.77, left_arm_2=-2.09, left_arm_3=-1.63)


def tuck(w, ex):
    """Fold the left arm up and out of the way; it stays folded for the rest of the run."""
    from sbv.functionality.f4_robot.rby1 import P

    for n, v in TUCK.items():
        w.d.ctrl[w.rb.act[w.m.joint(P + n).id]] = v
    ex.go(1.0)


def top(w, obj, j, u):
    """The middle of the part's top (its face against u), the part's longer side across u and its length along it."""
    m = w.m
    sub = OC._sub(m, m.jnt_bodyid[j])
    gs = [g for g in OC.geoms(w, obj) if m.geom_bodyid[g] in sub and w.collides(g)]
    p = np.vstack([Scene.geom_points(w, g) for g in gs])
    c = (p.min(0) + p.max(0)) / 2
    flat = p - np.outer(p @ u, u)
    flat -= flat.mean(0)
    side = np.linalg.svd(flat)[2][0]
    side = side - u * (side @ u)
    side /= np.linalg.norm(side)
    return c + u * ((p @ u).min() - c @ u), side, float(np.ptp(p @ side))


def tip(w, rot, u):
    """The right fingertip sphere lowest along u when the hand has orientation rot: (its centre from the tool point, its
    radius). The fingertips are 3 cm spheres; open, the two stand 10 cm apart."""
    m, d = w.m, w.d
    R = d.site_xmat[w.rb.tcp].reshape(3, 3)
    tips = [
        (rot @ (R.T @ (d.geom_xpos[g] - d.site_xpos[w.rb.tcp])), float(m.geom_rbound[g]))
        for g in range(m.ngeom)
        if w.collides(g) and "ee_finger_r" in w.name(m.geom_bodyid[g])
    ]  # push.hand_reach
    return max(tips, key=lambda t: t[0] @ u + t[1]), tips


def pose(u, side, out):
    """The hand for one fingertip: fingers pointing along u leaned LEAN away from the robot's side (out), so the wrist
    stays on the robot's side, and the open hand rolled ROLL about them so one fingertip is lowest."""
    away = -out
    f = u + math.tan(LEAN) * away
    f /= np.linalg.norm(f)
    row = np.cross(u, side)
    row -= f * (row @ f)
    row /= np.linalg.norm(row)
    dn = u - f * (u @ f)
    dn /= np.linalg.norm(dn)
    return hand(f, math.cos(ROLL) * row + math.sin(ROLL) * dn)


def robot_side(xs, ys, free, t, side, base):
    """Which way along the part's longer side the robot can stand: the end with free floor 60 cm away, else towards
    it."""
    ok = [
        s
        for s in (side, -side)
        if (lambda p: free[:, int(np.abs(xs - p[0]).argmin()), int(np.abs(ys - p[1]).argmin())].any())(t + 0.6 * s)
    ]
    return ok[0] if len(ok) == 1 else (side if side[:2] @ (np.asarray(base[:2]) - t[:2]) >= 0 else -side)


def perform(w, ex, xs, ys, free, st, others):
    """Press one part from where the robot stands: st has object, part (optional) and ref (the reference room's
    record)."""
    m, d = w.m, w.d
    obj = st["object"]
    t0 = ex.t
    j, why = OC.find_joint(w, obj, None, st.get("ref")) if st.get("ref") else OC.find_joint(w, obj, st.get("part"))
    if j is None:
        return dict(success=False, reason=why)
    if m.jnt_type[j] != OC.SLIDE:
        return dict(success=False, reason=f"{OC.jname(m, j)} does not slide, so it cannot be pressed")
    rec, u = OC.record(w, obj, j)
    qa = m.jnt_qposadr[j]
    travel = rec["travel"]
    q0 = float(d.qpos[qa])
    need = float(
        (st.get("ref") or {}).get("travel", travel)
    )  # how far the reference room's key goes down: pressed is judged against it, not the copy's own travel
    span = lambda k: max(
        abs(r - m.qpos0[m.jnt_qposadr[k]]) for r in m.jnt_range[k]
    )  # each other part's travel from its start
    rest = {k: (m.jnt_qposadr[k], float(d.qpos[m.jnt_qposadr[k]]), float(span(k))) for k in OC.joints(w, obj) if k != j}
    if not any(w.collides(g) for g in OC.geoms(w, obj) if m.geom_bodyid[g] in OC._sub(m, m.jnt_bodyid[j])):
        return dict(success=False, reason=f"{OC.jname(m, j)}: the part has no collision shape to press")
    tuck(w, ex)
    ex.go(0.4, grip=OPEN)
    t, side, length = top(w, obj, j, u)
    side = robot_side(xs, ys, free, t, side, ex.base)
    if length > LONG:
        t = t + side * (length / 2 - FRONT)  # a long key: pressed 3 cm from its front
    rot = pose(u, side, side)
    (off, r), tips = tip(w, rot, u)
    tcp = t - off - r * u  # the lowest fingertip's bottom on the part's top
    if min(off @ u + r - (o @ u + rr) for o, rr in tips if np.linalg.norm(o - off) > 0.03) < CLEAR:
        return dict(success=False, reason="the other fingertip would come within 3.5 cm of the part's top")
    poses = [tcp - ABOVE * u, tcp, tcp + (travel + PAST) * u]
    q, ok = ex.q, True
    for p in poses:
        q, err = w.rb.ik(ex.base, p, rot, q, iters=400)
        ok = ok and err <= 0.01
    base = ex.base
    if not ok:  # drive to where the hand reaches the part, the arm carried
        dist0 = RP.distances(free, RP.state_of(xs, ys, ex.base))
        sp = OC.stand(w, xs, ys, free, dist0, poses, [rot] * 3, key=1)
        if sp is None:
            return dict(success=False, reason="no reachable base pose presses the part")
        here = d.site_xpos[w.rb.tcp].copy()
        ex.line(ex.base, here, RP.carry_point(ex.base), d.site_xmat[w.rb.tcp].reshape(3, 3), seconds=1.5)
        ex.drive(RP.path(xs, ys, dist0, sp[2]))
        base = sp[0]
        q_carry, _ = w.rb.ik(base, RP.carry_point(base), rot, sp[1][0], iters=800)
        ex.go(2.0, arm=[q_carry])
        ex.line(base, RP.carry_point(base), poses[0], rot, seconds=2.0)
    else:
        ex.line(base, d.site_xpos[w.rb.tcp].copy(), poses[0], rot, seconds=1.0)
    ex.line(base, poses[0], poses[1], rot, k=6, seconds=1.0)
    ex.line(base, poses[1], poses[2], rot, k=4, seconds=0.6)
    ex.go(0.3)
    down = abs(float(d.qpos[qa]) - q0)
    pressed = sorted(
        OC.jname(m, k) for k, (a, s, tr) in rest.items() if tr > 0 and abs(float(d.qpos[a]) - s) > NEIGHBOUR * tr
    )
    ex.line(base, poses[2], poses[0], rot, k=4, seconds=0.8)
    ex.go(0.4)
    back = abs(float(d.qpos[qa]) - q0)
    sub = OC._sub(m, w.objects[obj])
    touching = any(
        (m.geom_bodyid[cn.geom1] in sub and m.geom_bodyid[cn.geom2] in w.robot)
        or (m.geom_bodyid[cn.geom2] in sub and m.geom_bodyid[cn.geom1] in w.robot)
        for cn in d.contact[: d.ncon]
    )
    mv = {n: round(float(np.linalg.norm(d.xipos[w.objects[n]] - v)), 3) for n, v in others.items()}
    mv = {n: v for n, v in mv.items() if v > 0.02}
    det = dict(
        part=rec["name"],
        travel=round(travel, 4),
        down=round(down, 4),
        back=round(back, 4),
        pressed=down >= FRAC * need,
        returned=back <= BACK * need or not m.jnt_stiffness[j],
        neighbours_pressed=pressed,
        released=not touching,
        others_moved=mv,
        seconds=round(ex.t - t0, 1),
        on_time=ex.t - t0 <= DEADLINE,
    )
    det["success"] = bool(
        det["pressed"] and det["returned"] and not pressed and det["released"] and not mv and det["on_time"]
    )
    if not det["success"]:
        det["reason"] = (
            f"the {rec['name']} went down {1000 * down:.1f} of {1000 * need:.1f} mm"
            if not det["pressed"]
            else f"pressed {', '.join(pressed)} too"
            if pressed
            else f"the {rec['name']} stayed {1000 * back:.1f} mm down"
            if not det["returned"]
            else RP.why(det)
        )
    return det
