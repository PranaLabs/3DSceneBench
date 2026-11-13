"""F3 object functionality (doc Appendix C.3): each required object-function instance is tested in the scene with a
standard driver or fixture, no robot policy. A door or drawer is one unit that ANDs its articulation components
(presence, type, connectivity, axis, pivot, limit, opening clearance, rest behavior, actuation response, full required
travel) and the dynamic integrity of its rollouts. Container retention and load bearing are single-outcome units, the
idealized grasp-and-lift a two-force bracket; all are scored like F2 bundles."""

import math
import numpy as np, mujoco
from scipy.optimize import linear_sum_assignment
from sbv.common.scene import Unstable, HINGE, LOAD_H, PAD, sleep_off
from sbv.functionality.integrity import Monitor
from sbv.functionality.f2_physical import pose, turned, tagged

AXIS_TOL, PIVOT_TOL, CONN_TOL, LIMIT_FRAC, CLEAR_PEN = math.radians(10), 0.05, 0.10, 0.9, 0.002
ACT_FRAC, ACT_T, FULL_FRAC, FULL_T, FULL_BUDGET, REST_T = 0.4, 2.0, 0.9, 3.0, 3.0, 2.0
COMPONENTS = [
    "presence",
    "type",
    "connectivity",
    "axis",
    "pivot",
    "limit",
    "clearance",
    "handle",
    "rest",
    "actuation",
    "full_travel",
]
HINT_COMPONENTS = ["rest", "actuation", "full_travel"]  # these depend on joint damping, friction and springs, which
# pictures cannot show; input-only scores the kinematic components


# ---- doors and drawers ---------------------------------------------------------------------------------------------
def jname(m, j):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"


def joint_record(scene, n, j):
    """One joint, in room axes: type, opening sign and travel from the start state, the opening axis, and its pivot and
    moving part relative to the object's bounding-box centre. Room axes on purpose: a cabinet turned 90 degrees in a
    copy opens its drawer in another direction of the room, and that counts as wrong."""
    m, d = scene.m, scene.d
    scene.reset()
    q0 = float(scene.start[m.jnt_qposadr[j]])
    lo, hi = m.jnt_range[j]
    if m.jnt_limited[j]:
        end = hi if abs(hi - q0) >= abs(lo - q0) else lo
        sgn, trav = (1.0 if end > q0 else -1.0), float(abs(end - q0))
    else:
        sgn, trav = 1.0, float("inf")
    olo, ohi = scene.aabb(n)
    c = (olo + ohi) / 2
    sub = scene.subtree(m.jnt_bodyid[j])
    gs = [g for g in scene.ogeoms[n] if m.geom_bodyid[g] in sub]
    clo, chi = scene.aabb(n, gs) if gs else (d.xipos[m.jnt_bodyid[j]],) * 2
    return dict(
        name=jname(m, j),
        type=int(m.jnt_type[j]),
        sgn=sgn,
        travel=trav,
        axis=(d.xaxis[j] * sgn).round(6).tolist(),
        anchor=(d.xanchor[j] - c).round(5).tolist(),
        child=((clo + chi) / 2 - c).round(5).tolist(),
        size=float(np.linalg.norm(chi - clo)),
        half_h=round(float(ohi[2] - olo[2]) / 2, 5),
    )


def from_bottom(rec, half_h):
    """A joint record's moving part and pivot from the object's box bottom centre instead of its box centre: the box's
    half height (for older cases evaluate passes the reference object's)."""
    if half_h is None:
        return rec
    up = lambda v: (np.asarray(v, float) + [0.0, 0.0, half_h]).round(5).tolist()
    return dict(rec, child=up(rec["child"]), anchor=up(rec["anchor"]))


def ref_record(unit):
    """The true joint's record as the copy's is compared: from the bottom centre when the unit knows the true box
    height."""
    ref = unit["params"]["ref"]
    return from_bottom(ref, ref.get("half_h", unit["params"].get("half_h")))


def joint_correspondence(units, scene, corr):
    """(slot, reference joint name) -> generated joint id: within each matched object, the one-to-one pairing of joints
    whose moving parts sit closest, relative to the object centre; a joint of another type (hinge for slide) is matched
    only when nothing better is left."""
    out, by_obj = {}, {}
    for u in units:
        if u["kind"] == "joint":
            by_obj.setdefault(u["slots"][0], []).append(ref_record(u))
    for slot, refs in by_obj.items():
        g = corr.get(slot)
        if g is None or not scene.joints[g]:
            continue
        js = scene.joints[g]
        recs = [joint_record(scene, g, j) for j in js]
        if any(
            r is not ref
            for r, ref in (
                (ref_record(u), u["params"]["ref"]) for u in units if u["kind"] == "joint" and u["slots"][0] == slot
            )
        ):
            recs = [from_bottom(r, r["half_h"]) for r in recs]
        cost = np.array(
            [
                [
                    np.linalg.norm(np.subtract(r["child"], x["child"])) + (100.0 if r["type"] != x["type"] else 0.0)
                    for x in recs
                ]
                for r in refs
            ]
        )
        for i, k in zip(*linear_sum_assignment(cost)):
            out[("joint", slot, refs[i]["name"])] = js[k]
    return out


def drive(scene, j, sgn, F, seconds, qpos, owner):
    """Push the joint toward its open end with generalized force F; returns the travel reached and the Monitor."""
    m, d = scene.m, scene.d
    a, v = m.jnt_qposadr[j], m.jnt_dofadr[j]
    scene.reset(qpos)
    q0 = float(d.qpos[a])
    from sbv.functionality.joint_trace import DriveTrace

    mon = Monitor(scene, [owner])
    trace = DriveTrace(scene, j, sgn, F, seconds)
    try:
        for k in range(int(round(seconds / m.opt.timestep))):
            d.qfrc_applied[v] = sgn * F
            scene.step(m.opt.timestep, mon)
            trace.tick(k)
    finally:
        d.qfrc_applied[v] = 0
    mon.drive_report = trace.finish()
    return float(abs(d.qpos[a] - q0)), mon


def sweep_report(scene, j, sgn, trav, qpos, max_evaluations=None):
    """Whole single-hinge/slide path, with a clearance witness or uncertainty reason."""
    with sleep_off(scene.m):
        return _sweep_report(scene, j, sgn, trav, qpos, max_evaluations)


def _sweep_report(scene, j, sgn, trav, qpos, max_evaluations=None):
    from sbv.common.sweep import CollisionPolicy, certify

    m = scene.m
    if (
        int(m.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
        or not np.isfinite(trav)
        or trav < 0
        or sgn not in (-1, 1)
    ):
        return dict(status="unresolved", reason="unsupported_joint_motion")
    # Never mutate the live simulation while inspecting counterfactual poses.
    d = mujoco.MjData(m)
    base = (scene.start if qpos is None else qpos).copy()
    d.qpos[:] = base
    mujoco.mj_kinematics(m, d)
    policy = CollisionPolicy(m)
    sub = scene.subtree(m.jnt_bodyid[j])
    moving = [g for g in range(m.ngeom) if m.geom_bodyid[g] in sub and policy.active(g)]
    other = [
        g
        for g in range(m.ngeom)
        if m.geom_bodyid[g] not in sub
        and policy.active(g)
        and not scene.name(scene.owner[m.geom_bodyid[g]]).startswith("probe_")
    ]

    def pose(t):
        d.qpos[:] = base
        d.qpos[m.jnt_qposadr[j]] += sgn * t * trav
        mujoco.mj_kinematics(m, d)

    def box(g):  # world AABB of a geom at the current pose
        R = d.geom_xmat[g].reshape(3, 3)
        c = d.geom_xpos[g] + R @ m.geom_aabb[g, :3]
        h = np.abs(R) @ m.geom_aabb[g, 3:]
        return c - h, c + h

    slide = m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE
    pose(1.0)
    end = {a: box(a) for a in moving}
    pose(0.0)
    others = {b: box(b) for b in other if m.geom_type[b] != mujoco.mjtGeom.mjGEOM_PLANE}
    pairs = []
    speeds = []
    for a in moving:
        lo, hi = box(a)
        if slide:  # a translation sweeps exactly the hull of its start and end boxes
            speed = trav
            lo, hi = np.minimum(lo, end[a][0]), np.maximum(hi, end[a][1])
        else:  # an arc stays within its travel bound of the start box
            offset = d.geom_xpos[a] - d.xanchor[j]
            radius = np.linalg.norm(np.cross(offset, d.xaxis[j])) + m.geom_rbound[a]
            speed = trav * radius
            lo, hi = lo - speed, hi + speed
        for b in other:
            if not policy.allows(a, b):
                continue
            if b in others and ((others[b][0] > hi + CLEAR_PEN).any() or (others[b][1] < lo - CLEAR_PEN).any()):
                continue
            pairs.append((a, b))
            speeds.append(speed)
    # Every pair needs its two endpoints plus some refinement; a fixed total would leave big rooms unexamined.
    budget = max(20000, 50 * len(pairs)) if max_evaluations is None else max_evaluations
    report = certify(m, d, pairs, speeds, pose, CLEAR_PEN, budget)
    report.update(tolerance=CLEAR_PEN, pair_count=len(pairs), required_travel=trav)
    if "geoms" in report:
        report["names"] = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in report["geoms"]]
        report["joint_position"] = float(base[m.jnt_qposadr[j]] + sgn * report["fraction"] * trav)
    return report


def rest_stays(scene, j, sgn, half, qpos, owner):
    """Set half open and let go: does it stay (within 5 degrees or 1 cm) for 2 s? Returns (stays, Monitor)."""
    m, d = scene.m, scene.d
    scene.reset(qpos)
    a = m.jnt_qposadr[j]
    target = d.qpos[a] + sgn * half
    d.qpos[a] = target
    mujoco.mj_forward(m, d)
    mon = Monitor(scene, [owner])
    scene.step(REST_T, mon)
    return bool(abs(d.qpos[a] - target) <= (math.radians(5) if m.jnt_type[j] == HINGE else 0.01)), mon


def joint_trial(scene, unit, corr=None, qpos=None):
    """Raw observations for one door or drawer: the static components, then the rollouts. Returns (obs, monitors)."""
    p = unit["params"]
    ref = ref_record(unit)
    slot = unit["slots"][0]
    obs = dict.fromkeys(COMPONENTS, False)
    obs["rest"] = None
    g = slot if corr is None else corr.get(slot)
    if corr is None:
        j = next((k for k in scene.joints.get(g, []) if jname(scene.m, k) == ref["name"]), None)  # unnamed joints too
    else:
        j = corr.get(("joint", slot, ref["name"]))
    if g is None or j is None:
        return obs, []
    rec = joint_record(scene, g, j)
    owner = scene.objects[g]
    c = scene.center(g)
    ax = np.array(rec["axis"])
    if ref is not p["ref"]:
        rec = from_bottom(rec, rec["half_h"])  # both from their boxes' bottom centres
    conn = float(
        np.linalg.norm(np.subtract(rec["child"], ref["child"]))
    )  # margins: reported with the components (summary.txt)
    obs["margins"] = dict(
        connectivity_m=round(conn, 4),
        connectivity_bar_m=round(max(CONN_TOL, 0.25 * ref["size"]), 4),
        pivot_bar_m=PIVOT_TOL,
    )
    obs.update(
        presence=True,
        type=rec["type"] == ref["type"],
        connectivity=conn <= obs["margins"]["connectivity_bar_m"],
        axis=bool(math.acos(np.clip(ax @ np.array(ref["axis"]), -1, 1)) <= AXIS_TOL),
        limit=rec["travel"] >= LIMIT_FRAC * ref["travel"],
    )
    if ref["type"] == HINGE:
        w = c + np.array(ref["anchor"]) - (c + np.array(rec["anchor"]))
        e = float(np.linalg.norm(np.cross(w, ax)))
        obs["pivot"] = e <= PIVOT_TOL
        obs["margins"]["pivot_m"] = round(e, 4)
    else:
        obs["pivot"] = True
    obs["handle"] = not ref.get("handle") or has_handle(scene, g, j)  # a true part with a handle needs one
    trav = min(ref["travel"], rec["travel"])
    mons = []
    try:
        obs["clearance_report"] = sweep_report(scene, j, rec["sgn"], trav, qpos)
        obs["clearance"] = obs["clearance_report"]["status"] == "clear"
        obs["rest"], mr = rest_stays(scene, j, rec["sgn"], 0.5 * trav, qpos, owner)
        travel, acts = tagged(
            [drive(scene, j, rec["sgn"], f * p["F"], ACT_T, qpos, owner) for b in p["bars"] for f in (1 / b, b)], 2
        )
        obs["actuation"] = [
            [t >= ACT_FRAC * ref["travel"] for t in travel[i : i + 2]] for i in range(0, len(travel), 2)
        ]  # per bar
        t, mo = drive(scene, j, rec["sgn"], FULL_BUDGET * p["F_full"], FULL_T, qpos, owner)
        obs["full_travel"] = t >= FULL_FRAC * ref["travel"]
        mons = [mr] + acts + [mo]
        obs["drive_reports"] = [m_.drive_report for m_ in acts] + [mo.drive_report]
        obs["diagnosis"] = (
            "geometric_obstruction"
            if obs["clearance_report"]["status"] == "collision"
            else "clearance_unresolved"
            if not obs["clearance"]
            else "actuation_shortfall"
            if not obs["full_travel"] or not obs["actuation"][0][1]
            else "actuation_bracket_mismatch"
            if obs["actuation"][0][0]
            else "required_travel_reached"
        )
        for m_ in mons:
            m_.component = (
                "rest" if m_ is mr else "full_travel" if m_ is mo else "actuation"
            )  # judged with its component
    except Unstable:
        obs.update(rest=None, actuation=False, full_travel=False, diagnosis="unstable_rollout")
    return obs, mons


def has_handle(scene, n, j):
    """Whether the moving part of joint j has a handle the robot hand can pinch (f4_robot/open_close.handle: a piece
    standing at least 1.2 cm proud of the part's front, at most 8.5 cm across). The robot's open and close moves need
    one."""
    from sbv.functionality.f4_robot import open_close as OC

    try:
        return OC.handle(scene, n, j, OC.record(scene, n, j)[1])[0] is not None
    except Exception:
        return False  # a part too odd to read has no usable handle


def joint_components(unit, obs, bar=0):
    """The eleven components (COMPONENTS) at one pass bar (0 = main); only actuation depends on the bar."""
    if obs.get("diagnosis") == "unstable_rollout":
        return dict.fromkeys(
            COMPONENTS, False
        )  # the part blew up when driven: nothing about it passes, as for every other unit kind
    comp = {k: bool(obs[k]) for k in COMPONENTS if k not in ("rest", "actuation")}
    comp["rest"] = obs["rest"] is not None and obs["rest"] == unit["expect"]["rest"]
    comp["actuation"] = bool(obs["actuation"]) and obs["actuation"][bar] == [False, True]
    return comp


# ---- containers, grasp, load --------------------------------------------------------------------------------------
def solid_hit(scene, p, u, skip):
    """Distance to, and owner of, the first colliding geom along a ray; visual-only geoms are seen through."""
    m, d = scene.m, scene.d
    gid = np.zeros(1, np.int32)
    travelled = 0.0
    for _ in range(8):
        x = mujoco.mj_ray(m, d, p, u, None, 1, skip, gid)
        if x < 0:
            return None, None
        if scene.collides(int(gid[0])):
            return travelled + x, int(scene.owner[m.geom_bodyid[gid[0]]])
        p = p + u * (x + 1e-4)
        travelled += x + 1e-4
    return None, None


def walled_in(scene, n, ball):
    """The ball sits below the top of object n, and a ray from it along each of 8 horizontal directions meets n."""
    p = scene.d.xipos[ball].copy()
    lo, hi = scene.aabb(n)
    reach = float(np.linalg.norm(hi[:2] - lo[:2]))
    if not lo[2] - 0.005 < p[2] < hi[2] - 0.005:
        return False
    for t in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        x, o = solid_hit(scene, p, np.array([np.cos(t), np.sin(t), 0.0]), ball)
        if x is None or x > reach or o != scene.objects[n]:
            return False
    return True


def container_once(scene, n, qpos):
    """A 2 cm ball released 5 cm above the opening must end below the rim, walled in on every side by the object, and
    stay there. (A chair seat holds the ball below the top of the backrest, but nothing walls it in.)"""
    scene.reset(qpos)
    lo, hi = scene.aabb(n)
    c = (lo + hi) / 2
    scene.place("ball", [c[0], c[1], hi[2] + 0.05])
    ball = scene.probe["ball"]
    mon = Monitor(scene, [ball])
    d = scene.d
    inside = lambda: walled_in(scene, n, ball)
    scene.step(2.0, mon)
    first = inside()
    scene.step(0.5, mon)
    return bool(first and inside() and np.linalg.norm(d.cvel[ball][3:]) < 0.05), mon


LIFT, LIFT_T, HOLD_T = 0.10, 1.0, 1.0  # carry: lift 10 cm in 1 s, then hold 1 s


def carry(scene, n, bodies, mon):
    """Lift container n straight up by LIFT, keeping its pose otherwise, then hold it there. The container is moved by
    hand (pose and velocity set every step), so only its contents respond. Returns, per body, whether it came along:
    it ends at least 80% of the lift higher than it started."""
    m, d = scene.m, scene.d
    a, v = scene.free_dof(n)
    p0, q0 = d.qpos[a : a + 3].copy(), d.qpos[a + 3 : a + 7].copy()
    z0 = [d.xipos[b][2] for b in bodies]
    dt = m.opt.timestep
    steps = int(round(LIFT_T / dt))
    for k in range(steps + int(round(HOLD_T / dt))):
        up = k < steps
        d.qpos[a : a + 3] = p0 + [0, 0, LIFT * min(k, steps) / steps]
        d.qpos[a + 3 : a + 7] = q0
        d.qvel[v : v + 6] = 0
        d.qvel[v + 2] = LIFT / LIFT_T if up else 0.0
        scene.step(dt, mon)
    return [bool(d.xipos[b][2] - z >= 0.8 * LIFT) for b, z in zip(bodies, z0)]


def holds_once(scene, n, items, qpos):
    """F3.5: every item the reference room declares the container holds comes along when it is lifted 10 cm. A pencil
    half out of its cup comes along; one leaning on the cup from outside, or lying beside it, stays behind."""
    scene.reset(qpos)
    bodies = [scene.objects[i] for i in items]
    mon = Monitor(scene, bodies)
    return carry(scene, n, bodies, mon), mon


# 3.6 turn over: what a container or a small holder holds must fall out when it is turned upside down, as balls fall out
# of a bag turned over. F3.2 and F3.5 reward holding only; a copy whose contents are wedged into the walls or fused shut
# passed both.
TURN_LIFT, TURN_LIFT_T, TURN_T, TURN_HOLD_T = (
    0.30,
    1.0,
    1.0,
    1.5,
)  # lift 30 cm in 1 s, turn 180 degrees in 1 s, hold 1.5 s
TURN_AXIS = np.array([1.0, 0.0, 0.0])  # a horizontal axis through the centre of mass
HAND = 1e4  # the turned object's mass factor during the turn: a hand does not yield (a 7 g coaster under a 211 g
# cup otherwise gives way inside every step, and even a welded cup sagged 12 cm)
LET_GO_M = 0.10  # fallen out: moved this far in the holder's frame and outside the holder's box


def _hand_constants(scene):
    """Recompute the mass-derived constants (invweight0: contact and weld softness) and keep the state."""
    m, d = scene.m, scene.d
    q, v = d.qpos.copy(), d.qvel.copy()
    mujoco.mj_setConst(m, d)
    d.qpos[:], d.qvel[:] = q, v
    mujoco.mj_forward(m, d)


def turn_over(scene, n, mon):
    """Lift object n by TURN_LIFT, turn it 180 degrees about TURN_AXIS through its centre of mass, hold it there. It is
    moved by hand (pose and velocity set every step, as carry() does) and made HAND times heavier meanwhile."""
    m, d = scene.m, scene.d
    a, v = scene.free_dof(n)
    b = scene.objects[n]
    links = scene.links[n]
    saved = m.body_mass[links].copy(), m.body_inertia[links].copy()
    m.body_mass[links] *= HAND
    m.body_inertia[links] *= HAND
    _hand_constants(scene)
    try:
        p0, q0, c0 = d.qpos[a : a + 3].copy(), d.qpos[a + 3 : a + 7].copy(), d.xipos[b].copy()
        dt = m.opt.timestep
        n1, n2, n3 = (int(round(t / dt)) for t in (TURN_LIFT_T, TURN_T, TURN_HOLD_T))
        qr, q, R, Rb = (np.zeros(k) for k in (4, 4, 9, 9))
        for k in range(n1 + n2 + n3):
            if k < n1:
                lift, th, vz, w = TURN_LIFT * (k + 1) / n1, 0.0, TURN_LIFT / TURN_LIFT_T, 0.0
            elif k < n1 + n2:
                lift, th, vz, w = TURN_LIFT, math.pi * (k - n1 + 1) / n2, 0.0, math.pi / TURN_T
            else:
                lift, th, vz, w = TURN_LIFT, math.pi, 0.0, 0.0
            mujoco.mju_axisAngle2Quat(qr, TURN_AXIS, th)
            mujoco.mju_quat2Mat(R, qr)
            r = R.reshape(3, 3) @ (p0 - c0)
            mujoco.mju_mulQuat(q, qr, q0)
            mujoco.mju_quat2Mat(Rb, q)
            omega = TURN_AXIS * w
            d.qpos[a : a + 3] = c0 + [0, 0, lift] + r
            d.qpos[a + 3 : a + 7] = q
            d.qvel[v : v + 3] = np.array([0, 0, vz]) + np.cross(omega, r)
            d.qvel[v + 3 : v + 6] = Rb.reshape(3, 3).T @ omega
            scene.step(dt, mon)
    finally:
        m.body_mass[links], m.body_inertia[links] = saved
        _hand_constants(scene)


def _in_frame(scene, n, body):
    d, b = scene.d, scene.objects[n]
    return d.xmat[b].reshape(3, 3).T @ (d.xipos[body] - d.xpos[b])


def let_go(scene, n, bodies, mon):
    """Turn n over; per body, whether it fell out: it moved LET_GO_M in n's frame and its centre left n's box."""
    lo, hi = scene.aabb(n)
    name = {b: k for k, b in scene.objects.items()}

    def over(x):  # the item's box and n's overlap in plan at the start (1 cm of slack): it is on n or in it
        if x not in name:
            return True  # the probe ball: put in through the opening
        a, b = scene.aabb(name[x])
        return bool(np.all(a[:2] <= hi[:2] + 0.01) and np.all(b[:2] >= lo[:2] - 0.01))

    start = [over(x) for x in bodies]
    r0 = [_in_frame(scene, n, x) for x in bodies]
    turn_over(scene, n, mon)
    lo, hi = scene.aabb(n)
    return [
        bool(
            on
            and np.linalg.norm(_in_frame(scene, n, x) - r) > LET_GO_M
            and (np.any(scene.d.xipos[x] < lo) or np.any(scene.d.xipos[x] > hi))
        )
        for x, r, on in zip(bodies, r0, start)
    ]  # an item left beside n never "fell out" of it


def turnover_once(scene, n, items, opening, key, qpos):
    """F3.6: turn n upside down; each of items (on it or in it at the start) and, with an opening, the 2 cm ball put in
    through it (as F3.2 does) must fall out, as they do in the reference room. A ball that does not get in is a failed
    run."""
    scene.reset(qpos)
    bodies = [scene.objects[i] for i in items]
    if opening is None:
        mon = Monitor(scene, bodies)  # the turned object is moved by hand, as in carry()
    else:
        aim = _aim_ball(scene, n, opening, key)
        bodies = bodies + [scene.probe["ball"]]
        mon = Monitor(scene, bodies)
        if aim is None:
            return NO_SURFACE, mon
        if not _ball_past(scene, aim, mon):
            return "the ball does not get in", mon
    return let_go(scene, n, bodies, mon), mon


def opening_once(scene, n, opening, key, qpos):
    """F3.2 at an opening the reference room declares (functions.json): the 2 cm ball starts just outside it and moves
    in along its direction at 0.5 m/s, settles for 1.5 s, and must then be all the way in (its centre at least its
    radius past the opening, so a ball resting on a lid or a solid top is not in) and held: carried along when a movable
    container is lifted 10 cm, or walled in by a fixed one. The opening was frozen in room axes beside a sample of the
    true container's collision surface (key); the copy's container is lined up with that sample first, because a copy
    picks its own body origin and axes."""
    scene.reset(qpos)
    ball = scene.probe["ball"]
    aim = _aim_ball(scene, n, opening, key)
    mon = Monitor(scene, [ball])
    if aim is None:
        mon.details = dict(outcome=NO_SURFACE)
        return False, mon
    if not _ball_past(scene, aim, mon):
        return False, mon
    if scene.free_dof(n) is None:
        return bool(walled_in(scene, n, ball) and np.linalg.norm(scene.d.cvel[ball][3:]) < 0.05), mon
    return carry(scene, n, [ball], mon)[0], mon


NO_SURFACE = "the container has no collision surface"


def _aim_ball(scene, n, opening, key):
    """Line the container up with the true one's collision sample (key) and start the 2 cm ball just outside the
    opening, moving in along its direction at 0.5 m/s. Returns (opening centre, direction, fit error), or None when the
    copy's container has no collision surface to line up (nothing can hold the ball: the run fails, the grade goes
    on)."""
    from sbv.common.align import collision_points, upright_fit
    from sbv.common.scene import BALL_R

    pts = collision_points(scene, n, len(key))
    if not len(pts):
        return None
    R, t, err = upright_fit(np.asarray(key, float), pts)
    pos, u = R @ opening["pos"] + t, R @ opening["dir"]
    u = u / np.linalg.norm(u)
    ball = scene.probe["ball"]
    scene.place("ball", pos - u * (BALL_R + 0.005))
    a = scene.m.jnt_dofadr[scene.m.body_jntadr[ball]]
    scene.d.qvel[a : a + 3] = 0.5 * u
    return pos, u, err


def _ball_past(scene, aim, mon):
    """Let the aimed ball settle 1.5 s; True if its centre ends at least its radius past the opening (a ball resting on
    a lid or a solid top is not in)."""
    from sbv.common.scene import BALL_R

    pos, u, err = aim
    mon.details = dict(opening_world=pos.round(4).tolist(), fit_rms_m=round(err, 4))
    scene.step(1.5, mon)
    past = float((scene.d.xipos[scene.probe["ball"]] - pos) @ u)
    mon.details["past_opening_m"] = round(past, 4)
    return past >= BALL_R


def lift_once(scene, n, F, qpos, check_clear=False):
    """Two fingers close with force F across the object's narrow side at its centre-of-mass height, then lift 20 cm in
    1 s; True if the object rises 15 cm. Returns None when the open fingers would cut into another object."""
    m, d = scene.m, scene.d
    scene.reset(qpos)
    b = scene.objects[n]
    lo, hi = scene.aabb(n)
    ext = hi - lo
    yaw = 0.0 if ext[0] <= ext[1] else math.pi / 2
    w = min(ext[0], ext[1])
    c = (lo + hi) / 2
    z = float(np.clip(d.xipos[b][2], lo[2] + PAD[2] + 0.005, hi[2]))
    quat = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    wr = scene.probe["wrist"]
    a, v = m.jnt_qposadr[m.body_jntadr[wr]], m.jnt_dofadr[m.body_jntadr[wr]]
    d.mocap_pos[scene.mocap] = [c[0], c[1], z]
    d.mocap_quat[scene.mocap] = quat
    d.qpos[a : a + 3] = [c[0], c[1], z]
    d.qpos[a + 3 : a + 7] = quat
    d.qvel[v : v + 6] = 0
    ja = {s: m.jnt_qposadr[scene.slide[s]] for s in "lr"}
    dof = {s: m.jnt_dofadr[scene.slide[s]] for s in "lr"}
    gap = w / 2 + PAD[0] + 0.002
    d.qpos[ja["l"]] = -gap
    d.qpos[ja["r"]] = gap
    mujoco.mj_forward(m, d)
    if check_clear:
        fingers = set(scene.finger.values())
        for cn in d.contact[: d.ncon]:
            b1, b2 = m.geom_bodyid[cn.geom1], m.geom_bodyid[cn.geom2]
            if (b1 in fingers or b2 in fingers) and scene.owner[b1] != b and scene.owner[b2] != b and -cn.dist > 0.003:
                return None, None
    z0 = d.xipos[b][2]
    mon = Monitor(scene, [b])
    dt = m.opt.timestep

    def hold():
        d.qfrc_applied[dof["l"]] = F
        d.qfrc_applied[dof["r"]] = -F

    for _ in range(int(round(0.3 / dt))):
        hold()
        scene.step(dt, mon)
    for k in range(int(round(1.0 / dt))):
        hold()
        d.mocap_pos[scene.mocap][2] = z + 0.2 * (k + 1) * dt
        scene.step(dt, mon)
    for _ in range(int(round(0.3 / dt))):
        hold()
        scene.step(dt, mon)
    return bool(d.xipos[b][2] - z0 >= 0.15), mon


def load_once(scene, n, rel, qpos):
    """A 5 kg, 10 cm box set on the object's top must stay there for 3 s, and the object must not move or tilt."""
    scene.reset(qpos)
    lo, hi = scene.aabb(n)
    c = (lo + hi) / 2
    pos = np.array([c[0] + rel[0], c[1] + rel[1], hi[2] + LOAD_H + 0.005])
    p0, R0 = pose(scene, n)
    scene.place("load", pos)
    L = scene.probe["load"]
    mon = Monitor(scene, [scene.objects[n], L])
    scene.step(3.0, mon)
    pl = scene.d.xipos[L]
    p1, R1 = pose(scene, n)
    return bool(
        pos[2] - pl[2] < 0.03
        and np.linalg.norm(pl[:2] - pos[:2]) < 0.05
        and np.linalg.norm(p1 - p0) < 0.02
        and turned(R0, R1) < math.radians(5)
    ), mon


def trial(scene, unit, corr=None, qpos=None):
    """Container, grasp and load units: (outcome list, monitors), outcome None when the object is missing."""
    if unit["kind"] == "joint":
        return joint_trial(scene, unit, corr, qpos)
    n = unit["slots"][0] if corr is None else corr.get(unit["slots"][0])
    p = unit["params"]
    if n is None:
        return None, []
    try:
        if unit["kind"] == "holds":
            items = [i if corr is None else corr.get(i) for i in unit["slots"][1:]]
            if None in items:
                return None, []
            if scene.free_dof(n) is None:
                return "the container has no free joint, so it cannot be lifted", []  # a copy's fault: it fails
            o, mo = holds_once(scene, n, items, qpos)
            return o, [mo]
        if unit["kind"] == "container":
            o, mo = container_once(scene, n, qpos)
            return [o], [mo]
        if unit["kind"] == "opening":
            o, mo = opening_once(scene, n, p["opening"], p["key"], qpos)
            return [o], [mo]
        if unit["kind"] == "turnover":
            items = [i if corr is None else corr.get(i) for i in unit["slots"][1:]]
            if None in items:
                return None, []
            if scene.free_dof(n) is None:
                return "the holder has no free joint, so it cannot be turned over", []  # a copy's fault: it fails
            o, mo = turnover_once(scene, n, items, p.get("opening"), p.get("key"), qpos)
            return o, [mo]
        if unit["kind"] == "load":
            o, mo = load_once(scene, n, p["rel"], qpos)
            return [o], [mo]
        if unit["kind"] == "grasp":
            o, mons = tagged(
                [lift_once(scene, n, f * p["F"], qpos) for b in p["bars"] for f in (1 / b, b)], 2
            )  # squeeze: weight / friction
            return o, [mo for mo in mons if mo]
    except Unstable:
        return "unstable", []
    raise ValueError(f"unknown F3 unit kind {unit['kind']}")
