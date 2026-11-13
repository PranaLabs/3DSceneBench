"""F4 shelving (doc Appendix C.3): the RB-Y1 takes a book lying flat on a stack and stands it upright, spine out, in a
bookcase bay. Declared observations, as for pick-and-place: the book's box and pose in the room the robot runs in, and
the floor map. The bay (a box in room coordinates), the way the book's spine faces at the start and the bay's open side
are the task's own (tasks.json, the reference's coordinates), so a copy whose shelf is somewhere else, or whose book
does not fit the bay, fails. The controller pushes the book along its spine side until the spine edge overhangs the
book below; pinches that edge from the spine side, one pad above the book and one below; lifts it, turns it upright and
carries it; slides it into the bay at the shelf's height until the spine is 1.5 cm behind the front edge; lets go and
backs out. The book creeps round a little in the two-finger grip while it is carried (its weight hangs 9 cm from the
pads), so the hand is aimed again from the book's pose at the bookcase and once more just outside the bay. Passes only
if the book ends in the bay (its footprint inside, the spine at most 2 cm proud of the front, its bottom within 1.5 cm
of the shelf), upright (its spine under 10 degrees from vertical), spine out (within 25 degrees of the bay's open
side), still, out of the hand, nothing else moved more than 2 cm, no tunnelling, penetration within the unit's
tolerance (the hand's own contacts with the book are allowed), within the deadline."""

import math
import numpy as np, mujoco
from scipy.spatial.transform import Rotation, Slerp
from sbv.functionality.integrity import Monitor
from sbv.functionality.f4_robot import plan as RP
from sbv.functionality.f4_robot.pick_place import Executor, NoRollout, OPEN, SHUT, CARRY, setup
from sbv.geometry.g2_shape import oriented_box
from sbv.common.scene import Scene

SLIDE, CLEAR, SPINE_IN, DEADLINE, Z = 0.05, 0.012, 0.015, 240.0, np.array([0.0, 0.0, 1.0])
LOOSE = -0.03  # gripper setting that frees a book up to 4 cm thick without swinging the pads wide
PAD_HALF, PAD_BELOW, PAD_BACK = (
    0.016,
    0.017,
    0.019,
)  # closed pads +-1.6 cm wide; 1.7 cm below the tool point; tool point 1.9 cm in from the spine


def hand(point, close):
    """Hand orientation: fingers pointing along `point`, closing along `close`."""
    z = -np.asarray(point, float)
    x = np.asarray(close, float)
    return np.column_stack([x, np.cross(z, x), z])


def box(w, n):
    """Centre, horizontal side directions and extents of a body's collision geometry: (c, [u1, u2], [e1, e2,
    height])."""
    pts = np.vstack([Scene.geom_points(w, g) for g in w.geoms(n) if w.collides(g)])
    cb, a, e = oriented_box(pts)
    u = np.array([[math.cos(a), math.sin(a), 0.0], [-math.sin(a), math.cos(a), 0.0]])
    return np.r_[cb[:2], (pts[:, 2].min() + pts[:, 2].max()) / 2], u, np.asarray(e, float)


def reach(w, xs, ys, free, dist, poses, rot, key=0, tries=150):
    """A base pose from which the hand, starting from its carry pose, follows the poses in turn with orientation rot,
    each within 1 cm: (base, joint angles, state) or None. Tried where pose `key` is 0.25 to 1 m ahead and on the right
    and the fingers point ahead or to the right (the right arm's side), the easiest first (60 cm ahead, 20 cm right,
    fingers 30 degrees right of ahead), then the fewest grid steps away."""
    k, i, j = np.nonzero(free & (dist >= 0))
    yaw = 2 * math.pi * k / RP.NYAW
    t = np.asarray(poses[key], float)
    v = t[:2] - np.c_[xs[i], ys[j]]
    f, lat = v[:, 0] * np.cos(yaw) + v[:, 1] * np.sin(yaw), -v[:, 0] * np.sin(yaw) + v[:, 1] * np.cos(yaw)
    pt = -rot[:2, 2]
    rel = (
        ((math.atan2(pt[1], pt[0]) - yaw + math.pi) % (2 * math.pi) - math.pi)
        if np.linalg.norm(pt) > 0.3
        else np.full(len(k), -0.5)
    )
    ok = np.nonzero((f >= 0.25) & (f <= 1.0) & (lat >= -0.55) & (lat <= 0.2) & (rel >= -1.9) & (rel <= 0.5))[0]
    score = np.abs(f - 0.6) + np.abs(lat + 0.2) + 0.3 * np.abs(rel + 0.5) + 0.002 * dist[k, i, j]
    for n in ok[np.argsort(score[ok], kind="stable")][:tries]:
        base = (float(xs[i[n]]), float(ys[j[n]]), float(yaw[n]))
        if w.rb.ik(base, t, rot, np.zeros(13), iters=150)[1] > 0.03:
            continue  # a quick look first
        q, qs = w.rb.ik(base, np.asarray(RP.carry_point(base)), rot, np.zeros(13), iters=600)[0], []
        for p in poses:
            q, err = w.rb.ik(base, np.asarray(p, float), rot, q, iters=600)
            if err > 0.01:
                break
            qs.append(q)
        else:
            return base, qs, (int(k[n]), int(i[n]), int(j[n]))
    return None


def turn(ex, base, a, b, ra, rb, k=10, seconds=3.0):
    """Move the hand from (a, ra) to (b, rb), turning it on the way."""
    rots = Slerp([0, 1], Rotation.from_matrix([ra, rb]))(np.linspace(0, 1, k + 1)[1:]).as_matrix()
    qq, path = ex.q, []
    for s, r in zip(np.linspace(0, 1, k + 1)[1:], rots):
        qq, _ = ex.w.rb.ik(base, np.asarray(a) + s * (np.asarray(b) - np.asarray(a)), r, qq, iters=300)
        path.append(qq)
    ex.go(seconds, arm=path)


def run_task(xml, unit, corr=None, seed=0, room_scene=None, edit=None):
    p = unit["params"]
    obj, goal = [s if corr is None else corr.get(s) for s in unit["slots"]]
    if obj is None:
        return dict(success=False, reason="the book is missing"), None
    if goal is None:
        return dict(success=False, reason="the bookcase is missing"), None
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
    det = perform(
        w, ex, xs, ys, free, dict(object=obj, goal=goal, bay=p["bay"], spine=p["spine"], front=p["front"]), others
    )
    det["started"] = True
    return det, (mon if ex.t else None)


def perform(w, ex, xs, ys, free, st, others):
    """One shelving from where the robot stands, as a step of an activity or the whole task: st names the book, the
    bookcase, bay, spine and front."""
    m, d = w.m, w.d
    obj = st["object"]
    tb = w.objects[obj]
    t0 = ex.t
    dist0 = RP.distances(free, RP.state_of(xs, ys, ex.base))
    p = st
    if not any(w.collides(g) for g in w.geoms(obj)):
        return dict(
            success=False, reason=f"{obj} has no collision shape, so nothing can hold it"
        )  # a picture-only book, or one the object gate made inert
    c, u, e = box(w, obj)
    s = np.r_[p["spine"], 0.0] / np.linalg.norm(p["spine"])
    f = np.r_[p["front"], 0.0] / np.linalg.norm(p["front"])
    k = int(abs(u[1] @ s) > abs(u[0] @ s))
    sp = u[k] * np.sign(u[k] @ s)
    lng = u[1 - k]  # toward the spine; along the spine
    half, height, thick = e[k] / 2, e[1 - k], e[2]
    if thick > 0.5 * min(e[:2]):
        return dict(success=False, reason="the book does not lie flat")
    R0 = d.xmat[tb].reshape(3, 3).copy()
    B = R0.T @ np.column_stack([sp, lng, Z])
    cb = R0.T @ (c - d.xpos[tb])  # book axes, centre: body frame
    # 1. push the book along its spine side, the closed pads flat against its far edge, 7 mm above its bottom
    rp = hand(-Z, np.cross(sp, Z))
    zp = c[2] - thick / 2 + 0.007 + PAD_BELOW
    fore = np.r_[(c - half * sp)[:2], zp]
    a0, a1 = fore - (PAD_HALF + 0.02) * sp, fore + (SLIDE - PAD_HALF) * sp
    st = reach(w, xs, ys, free, dist0, [a0 + 0.1 * Z, a0, a1, a1 + 0.1 * Z], rp, key=1)
    if st is None:
        return dict(success=False, reason="no reachable base pose reaches the push")
    bp = st[0]
    ex.go(0.8, grip=SHUT)
    ex.drive(RP.path(xs, ys, dist0, st[2]))
    q_carry, _ = w.rb.ik(bp, RP.carry_point(bp), rp, st[1][0], iters=800)
    ex.go(2.0, arm=[q_carry])
    ex.line(bp, RP.carry_point(bp), a0 + 0.1 * Z, rp, seconds=2.0)
    ex.line(bp, a0 + 0.1 * Z, a0, rp)
    ex.line(bp, a0, a1, rp, seconds=SLIDE / 0.03)
    ex.line(bp, a1, a1 + 0.1 * Z, rp)
    # 2. pinch the spine edge, which now overhangs whatever lies under the book
    c, _, _ = box(w, obj)
    bottom = c[2] - thick / 2
    spine = c + half * sp
    under = [
        n
        for n, b in w.objects.items()
        if b != tb and bottom - 0.08 < w.aabb(n)[1][2] < bottom + 0.005 and any(w.collides(g) for g in w.geoms(n))
    ]  # a picture-only thing holds nothing up
    below = [
        q
        for n in under
        for q in np.vstack([Scene.geom_points(w, g) for g in w.geoms(n) if w.collides(g)])
        if abs((q - c) @ lng) < height / 2 and (q - c) @ sp < half + 0.08
    ]
    over = float((spine - c) @ sp - max([(q - c) @ sp for q in below], default=-half))
    if over < 2 * PAD_BACK + CLEAR:
        return dict(success=False, reason=f"the spine overhangs the book below by only {100 * over:.1f} cm")
    dist1, st = RP.distances(free, RP.state_of(xs, ys, ex.base)), None
    for tilt in (0, 20, 40):  # the hand level, or leaning down onto the edge where the arm cannot reach level
        for along in (0.0, 1.0, -1.0):  # the middle of the spine, or 4 cm from either end
            a = math.radians(tilt)
            pt, cl = math.cos(a) * -sp - math.sin(a) * Z, math.sin(a) * -sp + math.cos(a) * Z
            rg = hand(pt, cl)
            g = spine - PAD_BACK * sp + along * (height / 2 - 0.04) * lng
            pre, lift = g + 0.08 * sp, g + 0.05 * Z
            out = lift + 0.12 * sp + 0.08 * Z
            st = reach(w, xs, ys, free, dist1, [pre + 0.15 * Z, pre, g, lift, out], rg, key=2)
            if st:
                break
        if st:
            break
    if st is None:
        return dict(success=False, reason="no reachable base pose reaches the book's spine")
    up = 1.0 if (g - c) @ lng >= 0 else -1.0  # the end of the spine nearer the grip goes up
    bg = st[0]
    ex.line(bp, a1 + 0.1 * Z, RP.carry_point(bp), rp, seconds=1.5)
    ex.drive(RP.path(xs, ys, dist1, st[2]))
    q0, _ = w.rb.ik(bg, RP.carry_point(bg), rg, st[1][0], iters=800)
    ex.go(2.0, arm=[q0], grip=OPEN)
    ex.line(bg, RP.carry_point(bg), pre + 0.15 * Z, rg, seconds=2.0)
    ex.line(bg, pre + 0.15 * Z, pre, rg)
    ex.line(bg, pre, g, rg, seconds=2.0)
    ex.go(1.0, grip=SHUT)
    ex.line(bg, g, lift, rg, seconds=1.5)
    ex.line(bg, lift, out, rg, seconds=1.5)
    # 3. where the hand must be for the book to stand in the bay, spine out, 3 mm above the shelf
    lo, hi = np.asarray(p["bay"][0], float), np.asarray(p["bay"][1], float)
    side = np.array([-f[1], f[0], 0.0])
    corners = np.array([[x, y, lo[2]] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])])
    cc = side * ((lo + hi) / 2 @ side) + f * (max(corners @ f) - SPINE_IN - half) + Z * (lo[2] + 0.003 + height / 2)
    W = np.column_stack([f, up * Z, np.linalg.det(B) * np.cross(f, up * Z)])
    Rs = W @ B.T  # the book as it must stand

    def aim():
        """The hand's orientation and position that stand the book, as it sits in the hand now, in the bay."""
        Rh = d.site_xmat[w.rb.tcp].reshape(3, 3)
        Rht = Rs @ (Rh.T @ d.xmat[tb].reshape(3, 3)).T
        return Rht, cc - Rs @ cb - Rht @ (Rh.T @ (d.xpos[tb] - d.site_xpos[w.rb.tcp]))

    Rht, pht = aim()
    outside = lambda ph: ph + f * (2 * half + 0.05) + 0.004 * Z  # the book just clear of the bay
    dist2 = RP.distances(free, st[2])
    place = reach(w, xs, ys, free, dist2, [outside(pht), pht + 0.004 * Z, pht - 0.0025 * Z], Rht, key=1)
    if place is None:
        return dict(success=False, reason="no reachable base pose reaches the bay")
    bq = place[0]
    rc = RP.yawed(Rht, bg[2] - bq[2])
    turn(ex, bg, out, RP.carry_point(bg), rg, rc)  # the book stands up in the hand, turned as it will travel
    ex.drive(RP.path(xs, ys, dist2, place[2]), CARRY)  # the book in the hand: slowly
    Rh2, pht = aim()
    turn(ex, bq, RP.carry_point(bq), outside(pht), Rht, Rh2, seconds=2.5)  # the book creeps round in the
    Rht, pht = aim()
    ex.line(bq, outside(pht), pht + 0.004 * Z, Rht, seconds=2.0)  # hand on the way: aim again
    ex.line(bq, pht + 0.004 * Z, pht - 0.0025 * Z, Rht, k=3, seconds=0.6)  # down until it stands on the shelf
    ex.go(2.0, grip=LOOSE)
    ex.go(0.5)  # let go slowly, so the squeezed pads do not kick the book over
    ex.line(bq, pht - 0.0025 * Z, outside(pht), Rht, seconds=2.5)
    ex.go(0.5, grip=OPEN)
    ex.line(bq, outside(pht), RP.carry_point(bq), Rht, seconds=1.5)
    ex.go(1.5, arm=[np.zeros(13)])
    ex.go(2.0)
    # 4. the checks
    R1 = d.xmat[tb].reshape(3, 3)
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, tb, vel, 0)
    pts = np.vstack([Scene.geom_points(w, g) for g in w.geoms(obj) if w.collides(g)])
    b0 = pts.min(0)
    tilt = math.degrees(math.acos(np.clip(abs(R1 @ B[:, 1] @ Z), 0, 1)))
    out_deg = math.degrees(math.acos(np.clip(R1 @ B[:, 0] @ f, -1, 1)))
    span = lambda v, q: ((q @ v).min(), (q @ v).max())
    bf, bs, cf, cs = span(f, pts), span(side, pts), span(f, corners), span(side, corners)
    touching = any(
        (w.owner[m.geom_bodyid[cn.geom1]] == tb and m.geom_bodyid[cn.geom2] in w.robot)
        or (w.owner[m.geom_bodyid[cn.geom2]] == tb and m.geom_bodyid[cn.geom1] in w.robot)
        for cn in d.contact[: d.ncon]
    )
    moved = {n: round(float(np.linalg.norm(d.xipos[w.objects[n]] - v)), 3) for n, v in others.items()}
    moved = {n: v for n, v in moved.items() if v > 0.02}
    det = dict(
        in_goal=bool(
            bf[0] >= cf[0] - 0.005
            and bf[1] <= cf[1] + 0.02
            and bs[0] >= cs[0] - 0.005
            and bs[1] <= cs[1] + 0.005
            and abs(b0[2] - lo[2]) < 0.015
        ),
        bottom_cm=round(100 * float(b0[2] - lo[2]), 1),
        tilt_deg=round(tilt, 1),
        upright=tilt < 10,
        spine_deg=round(out_deg, 1),
        spine_out=out_deg < 25,
        still=bool(np.linalg.norm(vel[3:]) < 0.01),
        released=not touching,
        others_moved=moved,
        seconds=round(ex.t - t0, 1),
        on_time=ex.t - t0 <= DEADLINE,
    )
    det["success"] = bool(
        det["in_goal"]
        and det["upright"]
        and det["spine_out"]
        and det["still"]
        and det["released"]
        and not moved
        and det["on_time"]
    )
    return det


def trial(scene, unit, corr=None, seed=0, edit=None):
    det, mon = run_task(scene.xml, unit, corr, seed, scene, edit)
    if mon is None:
        mon = NoRollout(det)
    mon.details = det
    return [det["success"]], [mon]
