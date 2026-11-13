"""F4.4 open and close, as steps of an activity: the RB-Y1 pulls a door or drawer open by its handle, or pushes it shut.
Declared observations, as for the other moves: the part's joint (by name in the copy, else the joint of the same type
whose moving part sits nearest the reference room's), the handle on the moving part (its collision pieces standing at
least 1.2 cm proud of the part's front face, at most 8.5 cm across where the pads close) and the floor map. The hand
comes at the handle from the front, pinches it and follows the part's own motion: a straight pull along a slide, an arc
about a hinge with the hand turning with the door, up to the step's amount (default the travel, at most 25 cm or 70
degrees); then lets go, backs off and parks the arm. Close is the same path back to where the part started. Passes only
if the joint moved at least 80% of the amount (open) or ends within 2 cm or 5 degrees of its start (close), the hand is
off the part, the part is still, nothing else moved over 2 cm, within the deadline. A part with no handle is refused
when the task is named, and fails in a copy that has none: a flush door cannot be pulled by a two-finger hand."""

import math
import numpy as np, mujoco
from scipy.spatial.transform import Rotation
from sbv.functionality.f4_robot import plan as RP
from sbv.functionality.f4_robot.pick_place import OPEN, SHUT, DEADLINE, SPEED
from sbv.functionality.f4_robot.shelve import hand, turn
from sbv.common.scene import Scene

SLIDE, HINGE = int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_HINGE)
MAX, SHUT_TOL = {SLIDE: 0.25, HINGE: math.radians(70)}, {SLIDE: 0.02, HINGE: math.radians(5)}
FRAC, PROUD, ACROSS, CHORDS, Z = 0.8, 0.012, 0.085, 6, np.array([0.0, 0.0, 1.0])
ALONG = 0.5  # |face normal . slide| below this: the part slides along its own face (a sliding door)
BASE = (
    0.52  # the base box's half diagonal (0.69 m by 0.60 m) plus 6 cm: a swinging door or a drawer front must clear it
)


def geoms(x, obj):
    return x.geoms(obj) if hasattr(x, "geoms") else x.ogeoms[obj]


def _sub(m, b):
    """The bodies under b, b included (Scene.subtree, for a World too)."""
    out = {b}
    for k in range(b + 1, m.nbody):
        if m.body_parentid[k] in out:
            out.add(k)
    return out


def joints(x, obj):
    """The hinge and slide joints under an object, in a Scene or a robot World."""
    m = x.m
    b = x.objects[obj]
    return [j for j in range(m.njnt) if x.owner[m.jnt_bodyid[j]] == b and m.jnt_type[j] in (SLIDE, HINGE)]


def jname(m, j):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"


def centre(x, obj):
    lo, hi = x.aabb(obj)
    return (lo + hi) / 2


def part_centre(x, obj, j):
    gs = [g for g in geoms(x, obj) if x.m.geom_bodyid[g] in _sub(x.m, x.m.jnt_bodyid[j])]
    pts = np.vstack([Scene.geom_points(x, g) for g in gs]) if gs else x.d.xipos[x.m.jnt_bodyid[j]][None]
    return (pts.min(0) + pts.max(0)) / 2


def find_joint(x, obj, part=None, ref=None):
    """The part's joint: named by part (the joint's or its body's name, or a whole word of either, such as door), or the
    reference room's record (ref: by name, else the same type with the nearest moving part), or the object's only
    joint."""
    m = x.m
    js = joints(x, obj)
    if not js:
        return None, f"{obj} has no door, drawer or other moving part"
    names = {j: (jname(m, j), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.jnt_bodyid[j]) or "") for j in js}
    if part:
        word = lambda s: part.lower() in (s.lower(),) or part.lower() in s.lower().replace("__", "_").split("_")
        hit = [j for j in js if any(word(s) for s in names[j])]
        if len(hit) == 1:
            return hit[0], None
        return None, (
            f"no part named {part!r} on {obj}; its parts are {', '.join(names[j][0] for j in js)}"
            if not hit
            else f"{part!r} names {len(hit)} parts of {obj}: say which, {', '.join(names[j][0] for j in hit)}"
        )
    if ref:
        same = [j for j in js if names[j][0] == ref["name"]] or [j for j in js if m.jnt_type[j] == ref["type"]]
        if not same:
            return None, f"{obj} has no {'drawer (slide)' if ref['type'] == SLIDE else 'door (hinge)'} part"
        c = centre(x, obj)
        return min(same, key=lambda j: np.linalg.norm(part_centre(x, obj, j) - c - ref["child"])), None
    return (
        (js[0], None)
        if len(js) == 1
        else (None, f'{obj} has {len(js)} moving parts: name one with "part" ({", ".join(names[j][0] for j in js)})')
    )


def record(x, obj, j):
    """The joint in room axes: type, the opening sign and travel from its start, the moving part's centre from the
    object's centre; and the opening direction u at the part."""
    m, d = x.m, x.d
    q0 = float(m.qpos0[m.jnt_qposadr[j]])
    lo, hi = m.jnt_range[j]
    end = hi if abs(hi - q0) >= abs(lo - q0) else lo
    sgn = 1.0 if end >= q0 else -1.0
    pc = part_centre(x, obj, j)
    ax = d.xaxis[j].copy()
    u = ax * sgn if m.jnt_type[j] == SLIDE else sgn * np.cross(ax, pc - d.xanchor[j])
    return dict(
        name=jname(m, j),
        type=int(m.jnt_type[j]),
        sgn=sgn,
        travel=float(abs(end - q0)),
        child=(pc - centre(x, obj)).round(4).tolist(),
    ), (
        u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-9 else np.zeros(3)
    )  # zeros: a hinge through the part's own centre (handle() refuses it)


def handle(x, obj, j, u):
    """The handle on the moving part: the collision pieces standing proud of the part's front face along u, as (centre,
    the direction the pads close, the face's outward normal), or the reason there is none. A door that slides along its
    own face has its handle on whichever side one stands proud. The pads close across the smaller of the handle's extent
    along the face and its height, which must be under 8.5 cm."""
    m = x.m
    sub = _sub(m, m.jnt_bodyid[j])
    pts = {g: Scene.geom_points(x, g) for g in geoms(x, obj) if m.geom_bodyid[g] in sub and x.collides(g)}
    if not pts:
        return None, "the part has no collision shape"
    if not np.any(u):
        return (
            None,
            "the part turns about its own centre, so it has no front to hold",
        )  # record(): a hinge through the part's centre
    side = np.cross(u, Z)
    side /= np.linalg.norm(side)
    face = lambda p: np.ptp(p @ side) * np.ptp(p[:, 2])  # area seen from the front: the panel is a big piece,
    big = [
        g for g in pts if face(pts[g]) >= 0.5 * max(face(p) for p in pts.values())
    ]  # and of those the one farthest forward
    panel = max(big, key=lambda g: (pts[g] @ u).max())
    xy = pts[panel][:, :2] - pts[panel][:, :2].mean(0)
    lng = np.linalg.svd(xy)[2][0]  # (a drawer's back board is as big as its front)
    n = np.array([-lng[1], lng[0], 0.0])
    n *= np.sign(n @ u) or 1.0  # the panel's own normal, facing the way it opens
    for n in [n] if abs(n @ u) >= ALONG else [n, -n]:  # a door sliding along its face: its handle's side
        side = np.cross(n, Z)
        front = float((pts[panel] @ n).max())
        hs = [g for g in pts if g != panel and (pts[g] @ n).min() >= front - 0.003]
        if hs:
            break
    if not hs:
        return None, "no handle or knob stands proud of the part's front face for the hand to pinch"
    p = np.vstack([pts[g] for g in hs])
    lo, hi = p.min(0), p.max(0)
    c = (lo + hi) / 2
    c += n * (
        (front + (p @ n).max()) / 2 - c @ n
    )  # pinch midway between the panel and the handle's face: a knob's neck, a bar's middle
    if (p @ n).max() - front < PROUD:
        return (
            None,
            f"the handle stands only {100 * ((p @ n).max() - front):.1f} cm proud of the part; the fingers need 1.2 cm",
        )
    ext = (np.ptp(p @ side), np.ptp(p[:, 2]))
    if min(ext) > ACROSS:
        return None, f"the handle is {100 * min(ext):.1f} cm across; the hand opens 8.5 cm"
    return (c, side if ext[0] <= ext[1] else Z, n), None


def stand(w, xs, ys, free, dist, poses, rots, key=1, tries=150, avoid=(), clear=None):
    """A base pose from which the hand follows every pose with its own orientation, each within 1 cm (shelve.reach,
    with one orientation per pose), its centre outside every (point, radius) in avoid, where the part will sweep:
    (base, joint angles, state) or None."""
    k, i, j = np.nonzero(free & (dist >= 0))
    yaw = 2 * math.pi * k / RP.NYAW
    t = np.asarray(poses[key], float)
    v = t[:2] - np.c_[xs[i], ys[j]]
    for p, r in avoid:
        keep = np.linalg.norm(np.c_[xs[i], ys[j]] - np.asarray(p[:2]), axis=1) >= r
        k, i, j, yaw, v = k[keep], i[keep], j[keep], yaw[keep], v[keep]
    f, lat = v[:, 0] * np.cos(yaw) + v[:, 1] * np.sin(yaw), -v[:, 0] * np.sin(yaw) + v[:, 1] * np.cos(yaw)
    pt = -rots[key][:2, 2]
    rel = (
        ((math.atan2(pt[1], pt[0]) - yaw + math.pi) % (2 * math.pi) - math.pi)
        if np.linalg.norm(pt) > 0.3
        else np.full(len(k), -0.5)
    )
    ok = np.nonzero((f >= 0.25) & (f <= 1.0) & (lat >= -0.55) & (lat <= 0.2) & (rel >= -1.9) & (rel <= 0.5))[0]
    score = np.abs(f - 0.6) + np.abs(lat + 0.2) + 0.3 * np.abs(rel + 0.5) + 0.002 * dist[k, i, j]
    for n in ok[np.argsort(score[ok], kind="stable")][:tries]:
        base = (float(xs[i[n]]), float(ys[j[n]]), float(yaw[n]))
        if clear and not clear(base):
            continue  # e.g. room to walk a sliding door open
        if w.rb.ik(base, t, rots[key], np.zeros(13), iters=150)[1] > 0.03:
            continue
        q, qs = w.rb.ik(base, np.asarray(RP.carry_point(base)), rots[0], np.zeros(13), iters=600)[0], []
        for p, r in zip(poses, rots):
            q, err = w.rb.ik(base, np.asarray(p, float), r, q, iters=600)
            if err > 0.01:
                break
            qs.append(q)
        else:
            return base, qs, (int(k[n]), int(i[n]), int(j[n]))
    return None


def perform(w, ex, xs, ys, free, st, others):
    """Open or close one part from where the robot stands: st has kind (open or close), object, part (optional), ref
    (the reference room's joint record, from tasks.py) and amount (optional)."""
    m, d = w.m, w.d
    obj = st["object"]
    tb = w.objects[obj]
    t0 = ex.t
    opening = st["kind"] == "open"
    j, why = find_joint(w, obj, None, st.get("ref")) if st.get("ref") else find_joint(w, obj, st.get("part"))
    if j is None:
        return dict(success=False, reason=why)
    rec, u = record(w, obj, j)
    qa = m.jnt_qposadr[j]
    q_start, q_closed = float(d.qpos[qa]), float(m.qpos0[qa])
    hinge = rec["type"] == HINGE
    true_travel = float(
        (st.get("ref") or {}).get("travel", rec["travel"])
    )  # how far the reference room's part opens: the copy's own travel only says how far the hand can take it
    need = min(true_travel, MAX[rec["type"]]) if opening else abs(q_start - q_closed)
    if st.get("amount") and opening:
        need = min(float(st["amount"]), true_travel)
    amount = min(need, rec["travel"]) if opening else need
    rec = dict(rec, need=need)  # the hand's path; "done" is judged against need (outcome)
    if not opening and amount < SHUT_TOL[rec["type"]]:
        return dict(
            success=True,
            moved=0.0,
            amount=0.0,
            done=True,
            note="already closed",
            seconds=0.0,
            on_time=True,
            released=True,
            still=True,
            others_moved={},
        )
    h, why = handle(w, obj, j, u)
    if h is None:
        return dict(success=False, reason=why)
    hc, close, n = h
    ax, anc = d.xaxis[j].copy(), d.xanchor[j].copy()
    sgn = rec["sgn"] * (1 if opening else -1)
    if hinge:
        u = sgn * np.cross(ax, hc - anc)
        u /= np.linalg.norm(u)  # the handle's own direction of travel
    else:
        u = u * (1 if opening else -1)
    if not hinge and abs(n @ u) < ALONG:
        return slide_along(w, ex, xs, ys, free, obj, j, rec, hc, n, u, amount, q_start, q_closed, opening, t0, others)
    into = -u if opening else u
    rot = hand(into, close)  # the fingers point into the furniture, onto the handle
    rots, poses = [rot, rot], [hc - 0.08 * into, hc]  # 8 cm in front of the handle, then the handle
    for k in range(1, CHORDS + 1):
        s = amount * k / CHORDS
        if hinge:
            R = Rotation.from_rotvec(ax * sgn * s).as_matrix()
            poses.append(anc + R @ (hc - anc))
            rots.append(R @ rot)
        else:
            poses.append(hc + u * s)
            rots.append(rot)
    far = max(
        float(np.linalg.norm((q - anc)[:2]))
        for g in geoms(w, obj)
        if m.geom_bodyid[g] in _sub(m, m.jnt_bodyid[j])
        for q in Scene.geom_points(w, g)
    )
    sweep = [(anc, far + BASE)] if hinge else [(poses[-1], BASE)]  # the base box stays out of the part's way
    dist0 = RP.distances(free, RP.state_of(xs, ys, ex.base))
    sp = stand(w, xs, ys, free, dist0, poses, rots, key=1, avoid=sweep)
    if sp is None:
        return dict(success=False, reason="no reachable base pose follows the handle's path")
    base = sp[0]
    ex.drive(RP.path(xs, ys, dist0, sp[2]))
    q_carry, _ = w.rb.ik(base, RP.carry_point(base), rot, sp[1][0], iters=800)
    ex.go(2.0, arm=[q_carry], grip=OPEN)
    ex.line(base, RP.carry_point(base), poses[0], rot, seconds=2.0)
    ex.line(base, poses[0], poses[1], rot, seconds=1.5)
    ex.go(1.0, grip=SHUT)
    for a, b, ra, rb in zip(poses[1:], poses[2:], rots[1:], rots[2:]):
        turn(ex, base, a, b, ra, rb, k=4, seconds=1.2)
    last, rl = poses[-1], rots[-1]
    ex.go(0.8, grip=OPEN)
    out = last + 0.08 * rl[:, 2]  # back off along the fingers, away from the part
    up = np.r_[out[:2], RP.CARRY[2]]  # straight up first, over the open part, then in to the carry pose
    ex.line(base, last, out, rl, seconds=1.0)
    ex.line(base, out, up, rl, seconds=1.5)
    ex.line(base, up, RP.carry_point(base), rl, seconds=1.5)
    ex.go(2.0)
    return outcome(w, ex, obj, j, rec, amount, q_start, q_closed, opening, t0, others)


def outcome(w, ex, obj, j, rec, amount, q_start, q_closed, opening, t0, others):
    """The step's checks once the hand is back: moved far enough (open) or shut, hand off, part still, nothing else
    moved."""
    m, d = w.m, w.d
    qa = m.jnt_qposadr[j]
    q1 = float(d.qpos[qa])
    moved = abs(q1 - q_start)
    shut = abs(q1 - q_closed)
    sub = _sub(m, m.jnt_bodyid[j])
    touching = any(
        (m.geom_bodyid[cn.geom1] in sub and m.geom_bodyid[cn.geom2] in w.robot)
        or (m.geom_bodyid[cn.geom2] in sub and m.geom_bodyid[cn.geom1] in w.robot)
        for cn in d.contact[: d.ncon]
    )
    mv = {n: round(float(np.linalg.norm(d.xipos[w.objects[n]] - v)), 3) for n, v in others.items()}
    mv = {n: v for n, v in mv.items() if v > 0.02}
    amount = rec.get(
        "need", amount
    )  # what the reference room's part does, not what this one can (a copy drawer with 5 mm of travel opened "fully")
    det = dict(
        part=rec["name"],
        moved=round(moved, 4),
        amount=round(amount, 4),
        from_closed=round(shut, 4),
        done=(moved >= FRAC * amount) if opening else (shut <= SHUT_TOL[rec["type"]]),
        released=not touching,
        still=bool(abs(d.qvel[m.jnt_dofadr[j]]) < 0.02),
        others_moved=mv,
        seconds=round(ex.t - t0, 1),
        on_time=ex.t - t0 <= DEADLINE,
    )
    det["success"] = bool(det["done"] and det["released"] and det["still"] and not mv and det["on_time"])
    if not det["done"]:
        det["reason"] = (
            f"the {rec['name']} moved {moved:.3f} of {amount:.3f}"
            if opening
            else f"the {rec['name']} ends {shut:.3f} from closed"
        )
    return det


def path_clear(xs, ys, free, base, u, amount):
    """The base can walk straight along u for amount from base without turning: every 5 cm of the way is free floor."""
    k = int(round(base[2] / (2 * math.pi / RP.NYAW))) % RP.NYAW
    for t in np.arange(0.0, amount + 1e-9, 0.05):
        p = np.asarray(base[:2]) + u[:2] * t
        if not free[k, int(np.abs(xs - p[0]).argmin()), int(np.abs(ys - p[1]).argmin())]:
            return False
    return True


def slide_along(w, ex, xs, ys, free, obj, j, rec, hc, n, u, amount, q_start, q_closed, opening, t0, others):
    """A door that slides along its own face: the fingers point into the face and pinch the bar handle across the slide,
    then the base walks along the wall with the door, the arm held as it pinched, for the whole amount (a doorway the
    robot can drive through needs about a metre, more than an arm reaches from one place); let go, hand back off the
    face and up."""
    rot = hand(-n, u)
    poses = [hc + 0.08 * n, hc]
    dist0 = RP.distances(free, RP.state_of(xs, ys, ex.base))
    sp = stand(w, xs, ys, free, dist0, poses, [rot, rot], key=1, clear=lambda b: path_clear(xs, ys, free, b, u, amount))
    if sp is None:
        return dict(success=False, reason="no reachable base pose pinches the handle with room to walk the door along")
    base = sp[0]
    ex.drive(RP.path(xs, ys, dist0, sp[2]))
    q_carry, _ = w.rb.ik(base, RP.carry_point(base), rot, sp[1][0], iters=800)
    ex.go(2.0, arm=[q_carry], grip=OPEN)
    ex.line(base, RP.carry_point(base), poses[0], rot, seconds=2.0)
    ex.line(base, poses[0], poses[1], rot, seconds=1.5)
    ex.go(1.0, grip=SHUT)
    b = np.array(base, float)
    for k in range(1, CHORDS + 1):  # the base walks the door along, the arm still
        ex.go(max(0.8, amount / CHORDS / (0.5 * SPEED)), base=tuple(b + np.r_[u[:2] * amount * k / CHORDS, 0.0]))
    last = w.d.site_xpos[w.rb.tcp].copy()
    ex.go(0.8, grip=OPEN)
    out = last + 0.08 * n
    up = np.r_[out[:2], RP.CARRY[2]]
    end = ex.base
    ex.line(end, last, out, rot, seconds=1.0)
    ex.line(end, out, up, rot, seconds=1.5)
    ex.line(end, up, RP.carry_point(end), rot, seconds=1.5)
    ex.go(2.0)
    return outcome(w, ex, obj, j, rec, amount, q_start, q_closed, opening, t0, others)
