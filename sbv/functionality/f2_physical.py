"""F2 physical validity (doc Appendix C.3): an initialization bundle per required object, and reference-calibrated
perturbation bundles (push, tilt, slide to rest, support removal, edge drop, push chain).

A bundle is a dict: id, kind, slots (the reference objects it names), params (frozen in the reference room), expect
(the outcome list the reference gives), tol (dynamic penetration tolerance) and hint_only. trial() runs one bundle in
any scene through the slot correspondence. A bundle passes only if its outcome equals expect and every rollout keeps
its dynamic integrity (no barrier crossing; penetration within tol).
"""

import math
import numpy as np, mujoco
from scipy.spatial import ConvexHull, QhullError
from sbv.common.scene import Unstable, MUG_H, HINGE
from sbv.functionality.integrity import Monitor, HOLD_S
from sbv.common.gravity_cache import cached

G = 9.81
BARS = (
    2.0,
    1.25,
)  # pass bars on a hidden amount (weight x friction, friction), main first: the copy's amount is within
# k times the true one either way. Factor 2 is the convention for amounts guessed from pictures
# (docs/reference/VERIFIERS.md, "Pass bars"); factor 1.25, the old bar, is reported beside it
MOVE, TURN = 0.05, math.radians(15)  # a push or chain breaks an object: it slides 5 cm or turns 15 degrees
STILL_MOVE, STILL_TURN, STILL_DRIFT, STILL_SPIN, JOINT_LIN, JOINT_ANG = (
    0.01,
    0.1,
    0.001,
    math.radians(1),
    0.01,
    math.radians(5),
)
PUSH_T, AFTER_T, TILT_T, TILT_MOVE, SLIDE_T, SLIDE_D, REST_V, DROP_T, CHAIN_PUSH, CHAIN_T = (
    0.1,
    2.5,
    1.5,
    0.02,
    2.0,
    0.10,
    0.05,
    2.0,
    0.2,
    3.0,
)
PEN_TOL = 0.005  # the smallest penetration tolerance a unit gets (reference rooms rest at ~0.2 mm)
RAMP_T, RAMP_MOVE, RAMP_TURN, RAMP_SPAN = (
    2.0,
    0.02,
    math.radians(5),
    2.5,
)  # slow push: the force grows
# from 0 over 2 s; giving way is 2 cm or 5 degrees; a copy is pushed up to
# 2.5 times the true breaking force, past the widest bar
AWAY_M, AWAY_TURN, FLAG_SHARE = (
    0.05,
    math.radians(30),
    0.10,
)  # not at rest: an object that ends 5 cm or 30
# degrees from where it was put is tested no further; a copy with more than
# 10% of its paired objects not still is flagged (reported, not zeroed)


def part(x, b, nb):
    """Bar b's share of an outcome or expect list that holds nb bars' outcomes one after another."""
    if not isinstance(x, list) or nb == 1:
        return x
    k = len(x) // nb
    return x[b * k : (b + 1) * k]


def tagged(runs, per):
    """(outcome, Monitor) pairs, per of them for each bar in turn: the outcomes, and the monitors tagged with their
    bar."""
    for i, (_, mo) in enumerate(runs):
        if mo is not None:
            mo.bar = i // per
    return [x for x, _ in runs], [mo for _, mo in runs]


def turned(R0, R1):
    return float(np.arccos(np.clip((np.trace(R0.T @ R1) - 1) / 2, -1, 1)))


def pose(scene, n):
    """The object's centre of mass (all its links) and its root's orientation: independent of where the root frame
    sits."""
    b = scene.objects[n]
    return scene.d.subtree_com[b].copy(), scene.d.xmat[b].reshape(3, 3).copy()


def broke(p0, R0, p1, R1):
    return bool(np.linalg.norm(p1[:2] - p0[:2]) > MOVE or turned(R0, R1) > TURN)


# ---- initialization bundle ---------------------------------------------------------------------------------------
def static_checks(scene, n):
    """Mass, inertia, centre of mass and contact parameters of one object, in its current configuration."""
    m, d = scene.m, scene.d
    root = scene.objects[n]
    links = scene.links[n]
    movers = [m.jnt_bodyid[j] for j in scene.joints[n]] + ([root] if scene.free_dof(n) else [])
    mass = all(np.isfinite(m.body_subtreemass[b]) and m.body_subtreemass[b] > 0 for b in movers)
    inertia = True
    for b in links:
        if m.body_mass[b] > 0:
            I = np.sort(m.body_inertia[b])
            inertia &= bool(np.isfinite(I).all() and I[0] > 0 and I[0] + I[1] >= I[2] * (1 - 1e-6))
    com = True
    if movers:
        pts = np.vstack(
            [scene.geom_points(g) for g in scene.ogeoms[n]] or [np.zeros((0, 3))]
        )  # all geometry, visual included
        try:
            eq = ConvexHull(pts).equations
            com = bool((eq[:, :3] @ d.subtree_com[root] + eq[:, 3]).max() <= 0.005)
        except (QhullError, ValueError):
            com = len(pts) > 0
    gs = [g for g in scene.ogeoms[n] if scene.collides(g)]
    contact = all(
        np.isfinite(m.geom_friction[g]).all()
        and (m.geom_friction[g] >= 0).all()
        and np.isfinite(m.geom_solref[g]).all()
        and np.isfinite(m.geom_solimp[g]).all()
        for g in gs
    )
    return dict(mass=bool(mass), inertia=bool(inertia), com=com, contact=bool(contact))


def init_record(scene, qpos=None, keep=None):
    """One gravity rollout of the whole scene from its submitted initial state (settle_s, then a 1 s late window); per
    object: static checks, root stillness (it settles, then neither drifts nor spins in the late window), joint
    stillness and integrity. keep (a dict) receives the state at settle_s as "q1": from the submitted state that is
    Scene.settle()'s start, bit for bit (the same reset and steps; the Monitor only reads), so the caller need not
    settle. SBV_GRAVITY_CACHE: kept for the same model and start (sbv/common/gravity_cache.py); a cached record leaves
    keep empty."""
    q = scene.q0 if qpos is None else qpos
    scene.reset(q)  # the key sees the model as the rollout runs it (probes parked)
    return cached(scene, "init-v2", lambda: _init_record(scene, q, keep), q)  # v2: moved_m and turned_deg added


def _init_record(scene, qpos, keep=None):
    m, d = scene.m, scene.d
    scene.reset(qpos)
    names = list(scene.objects)
    static = {n: static_checks(scene, n) for n in names}
    p0 = {n: pose(scene, n) for n in names}
    q0 = d.qpos.copy()
    mon = Monitor(scene, [scene.objects[n] for n in names])
    out = {}
    try:
        scene.step(scene.phys["settle_s"], mon)
        p1 = {n: pose(scene, n) for n in names}
        q1 = d.qpos.copy()
        if keep is not None:
            keep["q1"] = q1
        scene.step(1.0, mon)
        p2 = {n: pose(scene, n) for n in names}
        unstable = False
    except Unstable:
        unstable = True
    for n in names:
        rec = dict(role=scene.role[n], unstable=unstable, **static[n])
        if not unstable:
            rec["still"] = bool(
                np.linalg.norm(p1[n][0] - p0[n][0]) <= STILL_MOVE
                and turned(p0[n][1], p1[n][1]) <= STILL_TURN
                and np.linalg.norm(p2[n][0] - p1[n][0]) <= STILL_DRIFT
                and turned(p1[n][1], p2[n][1]) <= STILL_SPIN
            )
            rec["joints_still"] = all(
                abs(q1[m.jnt_qposadr[j]] - q0[m.jnt_qposadr[j]]) <= (JOINT_ANG if m.jnt_type[j] == HINGE else JOINT_LIN)
                for j in scene.joints[n]
            )
            rec["moved_m"] = round(float(np.linalg.norm(p2[n][0] - p0[n][0])), 4)
            rec["turned_deg"] = round(math.degrees(turned(p0[n][1], p2[n][1])), 2)
        rec["integrity"] = mon.report([scene.objects[n]])
        rec["mass_kg"] = float(m.body_subtreemass[scene.objects[n]])
        out[n] = rec
    return out


def role_ok(true_role, copy_role):
    """The copy's role (free, fixed, with or without joints) must be the true one, except that a piece the reference
    room fixes may stand free: pictures cannot show whether a cabinet is bolted to the wall, and a free 77 kg cabinet
    that stays put serves a robot the same; staying put is the "still" check. A movable reference object welded in place
    still fails (a robot could not move it), and the joints must match."""
    if not copy_role:
        return False
    tb, _, tj = true_role.partition("+")
    cb, _, cj = copy_role.partition("+")
    return tj == cj and (tb == cb or (tb == "fixed" and cb == "free"))


def away(rec):
    """The object ended far from where it was put in the gravity rollout: AWAY_M or AWAY_TURN (whole 6 s)."""
    return bool(
        rec
        and not rec.get("unstable")
        and (rec.get("moved_m", 0.0) > AWAY_M or math.radians(rec.get("turned_deg", 0.0)) > AWAY_TURN)
    )


def init_passes(unit, rec):
    e = unit["expect"]
    if rec is None or rec["unstable"]:
        return False
    ok = (
        role_ok(e["role"], rec["role"])
        and rec["mass"]
        and rec["inertia"]
        and rec["com"]
        and rec["contact"]
        and rec["still"]
        and not rec["integrity"]["crossed"]
        and not rec["integrity"].get("sweep_unresolved", False)
        and (
            rec["integrity"].get("sunk_s", 0.0) < HOLD_S
            if unit.get("rules")
            else max(rec["integrity"]["max_penetration"], rec["integrity"].get("penetration_upper", 0.0)) <= unit["tol"]
        )
    )
    if e.get("joints_still"):
        ok = ok and rec["joints_still"]
    if e.get("mass_range"):
        ok = ok and e["mass_range"][0] <= rec["mass_kg"] <= e["mass_range"][1]
    return bool(ok)


# ---- perturbation trials (each returns an outcome and the rollout's Monitor) --------------------------------------
def push_once(scene, n, F, u, qpos, seconds=PUSH_T, after=AFTER_T, watch=()):
    """Push through the object's centre of mass, three quarters of the way up its height; the force acts on the root
    body, so the torque is taken about the root's own centre."""
    scene.reset(qpos)
    m, d = scene.m, scene.d
    b = scene.objects[n]
    lo, hi = scene.aabb(n)
    p0, R0 = pose(scene, n)
    w0 = {c: pose(scene, c) for c in watch}
    f = F * np.array([u[0], u[1], 0.0])
    tau = np.cross(np.array([p0[0], p0[1], lo[2] + 0.75 * (hi[2] - lo[2])]) - d.xipos[b], f)
    mon = Monitor(scene, [b] + [scene.objects[c] for c in watch])
    for _ in range(int(round(seconds / m.opt.timestep))):
        d.xfrc_applied[b, :3] = f
        d.xfrc_applied[b, 3:] = tau
        scene.step(m.opt.timestep, mon)
    d.xfrc_applied[b] = 0
    scene.step(after, mon)
    if watch:
        return [broke(*w0[c], *pose(scene, c)) for c in watch], mon
    return broke(p0, R0, *pose(scene, n)), mon


def ramp_once(scene, n, Fmax, u, qpos):
    """Slow push (it replaces the 0.1 s blow, whose outcome followed the mass: twice the mass needed twice the force,
    half the friction 0.64 of it). A horizontal force through the centre of mass, three quarters up the object's height
    and toward u, grows from 0 to Fmax over RAMP_T. Returns the force at which the object gives way (slides RAMP_MOVE or
    turns RAMP_TURN), or None if it holds at Fmax, and the rollout's Monitor. A free object that slides gives way near
    weight x friction; a tall one that tips, at the force whose moment its weight cannot hold."""
    scene.reset(qpos)
    m, d = scene.m, scene.d
    b = scene.objects[n]
    lo, hi = scene.aabb(n)
    p0, R0 = pose(scene, n)
    at = np.array([p0[0], p0[1], lo[2] + 0.75 * (hi[2] - lo[2])])
    e = np.array([u[0], u[1], 0.0])
    mon = Monitor(scene, [b])
    steps = int(round(RAMP_T / m.opt.timestep))
    try:
        for k in range(1, steps + 1):
            f = Fmax * k / steps * e
            d.xfrc_applied[b, :3] = f
            d.xfrc_applied[b, 3:] = np.cross(at - d.xipos[b], f)
            scene.step(m.opt.timestep, mon)
            p1, R1 = pose(scene, n)
            if np.linalg.norm(p1[:2] - p0[:2]) > RAMP_MOVE or turned(R0, R1) > RAMP_TURN:
                return Fmax * k / steps, mon
        return None, mon
    finally:
        d.xfrc_applied[b] = 0


def ramp_outcome(Fc, F, bars):
    """The push outcome list from a copy's breaking force Fc against the true F: per bar k, (gives way at F/k, gives way
    at kF); the truth's is (False, True) at every bar."""
    return [bool(Fc is not None and Fc <= f) for k in bars for f in (F / k, F * k)]


def tilt_once(scene, n, sup, deg, u, qpos):
    """Tilt gravity by deg toward u for 1.5 s with every other free object held in place (the supporter does not slide
    away, and nothing else runs into the object): does the object slip or turn on its support? sup is kept for the
    bundle's slot list; holding everything else makes room axes the supporter's axes."""
    m, d = scene.m, scene.d
    g0 = m.opt.gravity.copy()
    th = math.radians(deg)
    gm = float(np.linalg.norm(g0))
    try:
        scene.reset(qpos)
        p0, R0 = pose(scene, n)
        held = [
            (a, v, d.qpos[a : a + 7].copy())
            for a, v in (scene.free_dof(k) for k in scene.objects if k != n and scene.free_dof(k) is not None)
        ]
        m.opt.gravity[:] = [gm * math.sin(th) * u[0], gm * math.sin(th) * u[1], -gm * math.cos(th)]
        mon = Monitor(
            scene,
            [scene.objects[n]],
            pinned=[scene.objects[k] for k in scene.objects if k != n and scene.free_dof(k) is not None],
        )
        for _ in range(int(round(TILT_T / m.opt.timestep))):
            scene.step(m.opt.timestep, mon)
            for a, v, q in held:
                d.qpos[a : a + 7] = q
                d.qvel[v : v + 6] = 0
        p1, R1 = pose(scene, n)
        return bool(np.linalg.norm(p1[:2] - p0[:2]) > TILT_MOVE or turned(R0, R1) > TURN), mon
    finally:
        m.opt.gravity[:] = g0


def slide_once(scene, n, sup, v, u, qpos):
    scene.reset(qpos)
    m, d = scene.m, scene.d
    f = scene.free_dof(n)
    mon = Monitor(scene, [scene.objects[n]])
    if f is None:
        return (False, True), mon  # a welded object never slides
    s = scene.objects.get(sup) if sup else None
    p0 = pose(scene, n)[0]
    s0 = d.xipos[s].copy() if s is not None else np.zeros(3)
    d.qvel[f[1] : f[1] + 2] = v * np.asarray(u)
    mujoco.mj_forward(m, d)
    scene.step(SLIDE_T, mon)
    p1 = pose(scene, n)[0]
    s1 = d.xipos[s].copy() if s is not None else np.zeros(3)
    return (
        bool(np.linalg.norm((p1 - s1)[:2] - (p0 - s0)[:2]) > SLIDE_D),
        bool(np.linalg.norm(d.qvel[f[1] : f[1] + 3]) < REST_V),
    ), mon


def remove_once(scene, sup, deps, qpos):
    m = scene.m
    gs = scene.ogeoms[sup]
    keep = [(m.geom_contype[g], m.geom_conaffinity[g]) for g in gs]
    try:
        for g in gs:
            m.geom_contype[g] = 0
            m.geom_conaffinity[g] = 0
        scene.reset(qpos)
        z0 = [scene.d.xipos[scene.objects[w]][2] for w in deps]
        mon = Monitor(scene, [scene.objects[w] for w in deps])
        scene.step(DROP_T, mon)
        return [bool(z - scene.d.xipos[scene.objects[w]][2] > MOVE) for z, w in zip(z0, deps)], mon
    finally:
        for g, (ct, ca) in zip(gs, keep):
            m.geom_contype[g] = ct
            m.geom_conaffinity[g] = ca


def surface_z(scene, n, xy, above):
    """Height of n's own surface straight below xy (a ray down from `above`), or None if n has nothing there."""
    m, d = scene.m, scene.d
    gid = np.zeros(1, np.int32)
    p = np.array([xy[0], xy[1], above])
    for _ in range(8):
        x = mujoco.mj_ray(m, d, p, np.array([0.0, 0.0, -1.0]), None, 1, -1, gid)
        if x < 0:
            return None
        g = int(gid[0])
        if scene.collides(g):
            return float(p[2] - x) if scene.owner[m.geom_bodyid[g]] == scene.objects[n] else None
        p = p - np.array([0.0, 0.0, x + 1e-4])
    return None


def edge_spots(scene, n, rel):
    """The two mug spots of an edge unit in this scene, 3 cm inside and outside this object's own top edge (before, the
    reference room's spots were kept relative to the box centre, so the unit failed any copy whose top was more than 3
    cm larger or smaller: a size check, and size is geometry's). The reference room's two spots give the direction and
    the sideways offset; the edge is found as calibration finds it (f2_calibrate.cal_edge): walking along that direction
    from the box centre, shifted sideways, on this object's top. None when it has no top to walk on."""
    from sbv.functionality.f2_calibrate import top_hit  # (f2_calibrate imports this module)

    a, b = np.asarray(rel[0], float), np.asarray(rel[1], float)
    nrm = (b - a) / np.linalg.norm(b - a)
    side = np.array([-nrm[1], nrm[0]])
    lo, hi = scene.aabb(n)
    c = (lo + hi)[:2] / 2
    top = float(hi[2])
    lateral = float(a @ side)
    for shift in (lateral, lateral - 0.1, lateral + 0.1, lateral - 0.2, lateral + 0.2):
        start = c + shift * side
        if top_hit(scene, n, start, top) != "top":
            continue
        for k in range(1, 400):
            h = top_hit(scene, n, start + 0.01 * k * nrm, top)
            if h != "top":
                break
        if h == "off":
            edge = start + 0.01 * (k - 1) * nrm
            return [(edge - 0.03 * nrm).tolist(), (edge + 0.03 * nrm).tolist()]
    return None


def edge_once(scene, n, xy, qpos):
    """A mug released 2 cm above the object's surface at the spot xy (room coordinates; its box top where the object has
    no surface there, past the edge) falls more than 10 cm, or stays. The surface is found by a ray, so a table with a
    raised rim is judged at its real top."""
    scene.reset(qpos)
    lo, hi = scene.aabb(n)
    top = surface_z(scene, n, xy, hi[2] + 0.5)
    z = (hi[2] if top is None else top) + MUG_H + 0.02
    scene.place("mug", [xy[0], xy[1], z])
    mon = Monitor(scene, [scene.probe["mug"]])
    scene.step(DROP_T, mon)
    return bool(z - scene.d.xipos[scene.probe["mug"]][2] > 0.1), mon


def trial(scene, unit, corr=None, qpos=None):
    """Run one perturbation bundle. Returns (outcome list, monitors); outcome is None when a named object is missing
    and "unstable" when the engine fails during the intervention."""
    k, p = unit["kind"], unit["params"]
    nm = (lambda s: s) if corr is None else corr.get
    names = [nm(s) for s in unit["slots"]]
    if any(x is None for x in names):
        return None, []
    try:
        if k == "push" and p.get("ramp"):  # slow push: one ramp to past the widest bar gives the breaking force
            Fc, mo = ramp_once(scene, names[0], RAMP_SPAN * p["F"], p["dir"], qpos)
            return ramp_outcome(Fc, p["F"], p["bars"]), [mo]
        if k == "push":
            return tagged(
                [push_once(scene, names[0], f * p["F"], p["dir"], qpos) for b in p["bars"] for f in (1 / b, b)], 2
            )
        if k == "tilt":  # tan(slip angle) is the friction: angles frozen at calibration
            sup = names[1] if len(names) > 1 else None
            return tagged(
                [tilt_once(scene, names[0], sup, a, p["dir"], qpos) for lo_hi in p["angles"] for a in lo_hi], 2
            )
        if k == "slide":  # a flick slides v^2 / (2 mu g): friction within k is speed within sqrt(k)
            sup = names[1] if len(names) > 1 else None
            o, mons = tagged(
                [
                    slide_once(scene, names[0], sup, f * p["v"], p["dir"], qpos)
                    for b in p["bars"]
                    for f in (b**-0.5, b**0.5)
                ],
                2,
            )
            return [x for i in range(0, len(o), 2) for x in (o[i][0], o[i + 1][0], o[i][1], o[i + 1][1])], mons
        if k == "remove":  # removal and its control, one contrast unit (doc C.3)
            o, mo = remove_once(scene, names[0], names[1:], qpos)
            if "control" not in p:
                return o, [mo]
            c = nm(p["control"])
            if c is None:
                return None, []
            o2, mo2 = remove_once(scene, c, names[1:], qpos)
            return o + o2, [mo, mo2]
        if k == "edge":  # the spots at this object's own edge; no top to walk on: the mug falls
            scene.reset(qpos)
            spots = edge_spots(scene, names[0], p["rel"])
            if spots is None:
                return [True, True], []
            r = [edge_once(scene, names[0], xy, qpos) for xy in spots]
            return [x for x, _ in r], [mo for _, mo in r]
        if k == "chain":
            o, mo = push_once(scene, names[0], p["F"], p["dir"], qpos, CHAIN_PUSH, CHAIN_T, watch=names[1:])
            return o, [mo]
    except Unstable:
        return "unstable", []
    raise ValueError(f"unknown F2 bundle kind {k}")
