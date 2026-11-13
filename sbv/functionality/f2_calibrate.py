"""Calibrate F2 bundles on the reference room only (doc C.3, "Reference-calibrated brackets"): find each transition,
check that the outcome switches once over a grid of strengths around it (push, tilt, slide), freeze the two stimuli
and the reference outcomes. Generated scenes are never recalibrated."""

import math, zlib
import numpy as np, mujoco
from sbv.functionality import f2_physical as P
from sbv.common.scene import Unstable, MUG_R

GRID = (
    0.5,
    0.65,
    0.8,
    1.25,
    1.6,
    2.0,
)  # strengths around a transition, as factors on the hidden amount: both bars' ends
TILT_MAX = 60.0  # degrees: the true slip or tip angle is searched up to here; it was 35, tan 35 = 0.70,
# so an object with friction over 0.7 on its support, rubber feet or a fabric seat, got no tilt unit)


def support_pairs(scene):
    """(supporter, supported) from upward contact normals or measured frictional load.
    Rims/shoulders need not touch either object's globally lowest point.
    """
    from sbv.geometry.support import contact_edges

    scene.reset()
    m = scene.m
    names = {b: n for n, b in scene.objects.items()}
    pairs = set()
    for s, w in contact_edges(scene):
        lo, hi = int(scene.owner[m.geom_bodyid[s]]), int(scene.owner[m.geom_bodyid[w]])
        arch = lo not in names and not scene.name(lo).startswith("probe_") and m.body_weldid[m.geom_bodyid[s]] == 0
        if lo != hi and hi in names and (arch or lo in names):
            pairs.add(("world" if arch else names[lo], names[hi]))
    return sorted(pairs)


def context(scene, center):
    pairs = support_pairs(scene)
    under = {}
    for s, w in pairs:
        under.setdefault(w, []).append(s)
    return dict(center=np.asarray(center[:2], float), pairs=pairs, under=under)


def supporter(ctx, n):
    s = [x for x in ctx["under"].get(n, []) if x != "world"]
    return s[0] if s else None


def toward_centre(scene, n, ctx):
    v = ctx["center"] - scene.center(n)[:2]
    L = float(np.linalg.norm(v))
    return (v / L).tolist() if L > 0.1 else [1.0, 0.0]


def bisect(test, lo, hi, steps, log=False):
    """Smallest strength in [lo, hi] at which test() is True, or None if test(hi) is False or test(lo) is True."""
    if not test(hi) or test(lo):
        return None
    for _ in range(steps):
        mid = (lo * hi) ** 0.5 if log else 0.5 * (lo + hi)
        if test(mid):
            hi = mid
        else:
            lo = mid
    return hi


def cal_push(scene, n, ctx):
    """The slow push (P.ramp_once): the true breaking force, pushed toward the room's centre up to three times the
    weight (friction 3, or a tip). An object that holds at that force gets no push unit."""
    if scene.free_dof(n) is None:
        return None
    u = toward_centre(scene, n, ctx)
    F, _ = P.ramp_once(scene, n, 3 * scene.m.body_subtreemass[scene.objects[n]] * P.G + 30, u, None)
    if F is None:
        return None
    for _ in range(4):  # again at the rate a copy is pushed (up to RAMP_SPAN F in RAMP_T),
        F1, _ = P.ramp_once(
            scene, n, P.RAMP_SPAN * F, u, None
        )  # until the force it gives way at stops changing: the force
        if F1 is None:
            return None  # has risen a while by the time it moved 2 cm, so the rate matters
        done, F = abs(F1 - F) <= 0.02 * F, F1
        if done:
            break
    return dict(
        id=f"push|{n}", kind="push", slots=[n], params=dict(F=round(F, 4), dir=u, bars=list(P.BARS), ramp=True)
    ), [False, True] * len(P.BARS)


def cal_tilt(scene, n, ctx):
    if scene.free_dof(n) is None:
        return None
    s, u = supporter(ctx, n), toward_centre(scene, n, ctx)
    once = lambda x: P.tilt_once(scene, n, s, x, u, None)[0]
    a = bisect(once, 2.0, TILT_MAX, 8)
    at = lambda k: round(
        max(math.degrees(math.atan(k * math.tan(math.radians(a)))), 0.5), 3
    )  # friction k times the true one
    if a is None or [once(at(x)) for x in GRID] != [False] * 3 + [True] * 3:
        return None
    return dict(
        id=f"tilt|{n}",
        kind="tilt",
        slots=[n] + ([s] if s else []),
        params=dict(angle=round(a, 3), bars=list(P.BARS), angles=[[at(1 / b), at(b)] for b in P.BARS], dir=u),
    ), [False, True] * len(P.BARS)


def cal_slide(scene, n, ctx):
    if scene.free_dof(n) is None:
        return None
    s, u = supporter(ctx, n), toward_centre(scene, n, ctx)
    once = lambda x: P.slide_once(scene, n, s, x, u, None)[0][0]
    v = bisect(once, 0.05, 4.0, 8)
    if v is None or [once(v * x**0.5) for x in GRID] != [False] * 3 + [True] * 3:
        return None
    return dict(
        id=f"slide|{n}",
        kind="slide",
        slots=[n] + ([s] if s else []),
        params=dict(v=round(v, 4), dir=u, bars=list(P.BARS)),
    ), [False, True, True, True] * len(P.BARS)


def cal_remove(scene, ctx):
    """Each object that carries others: remove it and watch everything resting on it, directly or not, together with a
    control in the same unit: remove the nearest other object the same dependents do not rest on, and nothing may fall.
    (A control on its own would pass in any copy where nothing moves.)"""
    above = {}
    for s, w in ctx["pairs"]:
        above.setdefault(s, set()).add(w)

    def connected(start, graph):
        seen, todo = {start}, [start]
        while todo:
            for node in graph.get(todo.pop(), ()):
                if node != "world" and node not in seen:
                    seen.add(node)
                    todo.append(node)
        return seen - {start}

    deps = lambda s: connected(s, above)
    below = lambda w: connected(w, ctx["under"])
    events = []
    for s in sorted(k for k in above if k != "world"):
        ds = sorted(w for w in deps(s) if scene.free_dof(w) is not None)
        if not ds:
            continue
        held = set(ds) | {s} | set().union(*[below(w) for w in ds])
        others = [o for o in scene.objects if o not in held]
        params = (
            dict(control=min(others, key=lambda o: float(np.linalg.norm(scene.center(o) - scene.center(s)))))
            if others
            else {}
        )
        events.append((dict(id=f"remove|{s}", kind="remove", slots=[s] + ds, params=params), None))
    return events


def top_hit(scene, n, xy, top):
    """What a ray straight down at xy meets first: "top" (this object's top surface), "blocked" (something standing on
    it) or "off" (nothing at the top's height: past the edge)."""
    gid = np.zeros(1, np.int32)
    x = mujoco.mj_ray(
        scene.m, scene.d, np.array([xy[0], xy[1], top + 0.3]), np.array([0.0, 0.0, -1.0]), None, 1, -1, gid
    )
    if x < 0 or 0.3 - x < -0.01:
        return "off"
    if scene.owner[scene.m.geom_bodyid[gid[0]]] == scene.objects[n] and abs(0.3 - x) < 0.01:
        return "top"
    return "blocked"


def on_top(scene, n, xy, top):
    return top_hit(scene, n, xy, top) == "top"


def cal_edge(scene, n, ctx):
    """A mug released 3 cm inside the top's edge facing the room centre (stays) and 3 cm past it (falls)."""
    scene.reset()
    lo, hi = scene.aabb(n)
    c = (lo + hi) / 2
    ext = hi - lo
    top = float(hi[2])
    if top < 0.2 or min(ext[:2]) < 0.2:
        return None
    u = ctx["center"] - c[:2]
    nrm = max((np.array(v, float) for v in ((1, 0), (-1, 0), (0, 1), (0, -1))), key=lambda v: float(v @ u))
    side = np.array([-nrm[1], nrm[0]])
    ring = [(MUG_R + 0.005) * np.array([np.cos(t), np.sin(t)]) for t in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    for shift in (0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3):
        start, edge = c[:2] + shift * side, None
        if not on_top(scene, n, start, top):
            continue
        for k in range(1, 400):
            h = top_hit(scene, n, start + 0.01 * k * nrm, top)
            if h != "top":
                edge = start + 0.01 * (k - 1) * nrm if h == "off" else None
                break
        if (
            edge is not None
            and k >= 5
            and all(on_top(scene, n, edge - 0.03 * nrm + r, top) for r in ring if r @ nrm <= 0.02)
        ):
            break
    else:
        return None
    rel = [(edge - 0.03 * nrm - c[:2]).round(4).tolist(), (edge + 0.03 * nrm - c[:2]).round(4).tolist()]
    return dict(id=f"edge|{n}", kind="edge", slots=[n], params=dict(rel=rel)), [False, True]


def cal_chain(scene, ctx, pushes):
    """A floor object pushed toward furniture within 40 cm with twice its breaking force for twice as long (four times
    the impulse), and every item on that furniture watched; kept only if something on it moves in the reference."""
    F = {u["slots"][0]: u["params"]["F"] for u in pushes}
    on = {}
    for s, w in ctx["pairs"]:
        if s != "world":
            on.setdefault(s, []).append(w)
    out = []
    for a in sorted(F):
        if ctx["under"].get(a) != ["world"] or scene.m.body_subtreemass[scene.objects[a]] > 25:
            continue
        la, ha = scene.aabb(a)
        for b, items in sorted(on.items()):
            lb, hb = scene.aabb(b)
            gap = float(np.linalg.norm(np.maximum(0, np.maximum(lb[:2] - ha[:2], la[:2] - hb[:2]))))
            items = sorted(c for c in items if c != a and scene.free_dof(c) is not None)
            if b == a or not items or gap > 0.4:
                continue
            u = scene.center(b)[:2] - scene.center(a)[:2]
            u = (u / np.linalg.norm(u)).round(4).tolist()
            out.append(
                (
                    dict(
                        id=f"chain|{a}|{b}", kind="chain", slots=[a] + items, params=dict(F=round(2 * F[a], 4), dir=u)
                    ),
                    None,
                )
            )
    return out


def finalize(scene, unit, want, run=P.trial):
    """Freeze the reference outcome as expect and set the penetration tolerance; returns a reason if rejected."""
    try:
        out, mons = run(scene, unit)
    except Unstable:
        return "reference run is numerically unstable"
    if out is None or out == "unstable":
        return "reference run failed"
    if want is not None and out != want:
        return f"reference outcome {out}, the test needs {want}"
    if unit["kind"] == "chain" and not any(out):
        return "nothing on the furniture moves in the reference"
    if unit["kind"] == "remove" and not any(out[: len(unit["slots"]) - 1]):
        return "nothing falls when the supporter is removed in the reference"
    if unit["kind"] == "remove" and "control" in unit["params"] and any(out[len(unit["slots"]) - 1 :]):
        return "something falls in the control removal in the reference"
    if any(mo.report(mo.owners)["crossed"] for mo in mons):
        return "reference rollout crosses a barrier"
    if any(mo.report(mo.owners).get("sweep_unresolved") for mo in mons):
        return "reference motion check is unresolved"
    pen = max([max(mo.pen.values() or [0.0]) for mo in mons] or [0.0])
    unit["expect"] = out
    unit["tol"] = round(max(P.PEN_TOL, 1.5 * pen + 0.002), 4)
    return None


def perturb(scene, rng):
    """One plausible copy of what pictures cannot show: every movable object's mass scaled by 0.7 to 1.3, and every
    object's friction shifted by up to 0.15, fixed objects and the architecture included (MuJoCo takes the larger of
    two touching frictions, so shifting one side only would often change nothing). Returns what to restore."""
    m = scene.m
    saved = (m.body_mass.copy(), m.body_inertia.copy(), m.geom_friction.copy())
    for n, _, movable in scene.draw_order:  # the whole room's list, so a small model draws the same numbers
        k = rng.uniform(0.7, 1.3) if movable else None
        df = rng.uniform(-0.15, 0.15)
        if n not in scene.links:
            continue
        if k is not None:
            m.body_mass[scene.links[n]] *= k
            m.body_inertia[scene.links[n]] *= k
        for g in scene.ogeoms[n]:
            m.geom_friction[g, 0] = max(0.05, m.geom_friction[g, 0] + df)
    df = rng.uniform(-0.15, 0.15)
    for g in range(m.ngeom):
        if (
            scene.owner[m.geom_bodyid[g]] == 0
            or scene.name(scene.owner[m.geom_bodyid[g]]) not in scene.objects
            and not scene.name(scene.owner[m.geom_bodyid[g]]).startswith("probe_")
        ):
            m.geom_friction[g, 0] = max(0.05, m.geom_friction[g, 0] + df)
    mujoco.mj_setConst(m, scene.d)
    return saved


def decidable(scene, unit, draws, need, run=P.trial):
    """Per pass bar, main first (one entry for a unit without bars): True if the reference outcome at that bar holds in
    at least `need` of `draws` plausible weight and friction variations."""
    m = scene.m
    rng = np.random.default_rng(zlib.crc32(unit["id"].encode()))
    nb = len(unit["params"].get("bars", ())) or 1
    hold = [0] * nb
    impossible = False
    for i in range(draws):
        saved = perturb(scene, rng)
        try:
            # Once no bar can reach `need`, skip the middle rollouts: the same draws are made (same RNG) and the last
            # rollout still runs, so scene.d ends where it always did for the calibrators after this one.
            if not impossible or i == draws - 1:
                out = run(scene, unit)[0]
                for b in range(nb):
                    hold[b] += P.part(out, b, nb) == P.part(unit["expect"], b, nb)
        except Unstable:
            pass
        finally:
            m.body_mass[:], m.body_inertia[:], m.geom_friction[:] = saved
            mujoco.mj_setConst(m, scene.d)
        impossible = all(h + (draws - i - 1) < need for h in hold)
    return [h >= need for h in hold]
