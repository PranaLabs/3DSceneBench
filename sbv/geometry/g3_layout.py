"""Geometry axis, G3 (doc Geometry and Appendix B): relation satisfaction, placement validity (support,
non-penetration, bounds) and navigation feasibility for the robot base. Static checks on the submitted arrangement."""

import numpy as np, mujoco
from scipy import ndimage

CELL, LOW, HIGH, DOWN = 0.05, 0.05, 1.6, np.array([0.0, 0.0, -1.0])  # obstacles stand 5 cm to 1.6 m high


def world_geoms(scene):
    """Architecture: collidable geoms on the world body and on top-level architectural bodies (not objects, not
    probes)."""
    out = []
    for g in range(scene.m.ngeom):
        o = int(scene.owner[scene.m.geom_bodyid[g]])
        name = scene.name(o)
        if scene.collides(g) and (o == 0 or (name not in scene.objects and not name.startswith("probe_"))):
            out.append(g)
    return out


def boxes(scene, gs):
    return {g: (lambda p: (p.min(0), p.max(0)))(scene.geom_points(g)) for g in gs}


def near(scene, ga, gb, margin):
    """The geoms of gb whose world boxes come within margin of the box around ga (planes always do). The plain form of
    GeometrySnapshot.near(): tests/test_geometry_performance.py checks the snapshot against it."""
    if not ga:
        return []
    pa = np.vstack([scene.geom_points(g) for g in ga])
    lo, hi = pa.min(0) - margin, pa.max(0) + margin
    return [
        g
        for g, (l, h) in boxes(scene, gb).items()
        if scene.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE or ((l <= hi).all() and (h >= lo).all())
    ]


class GeometrySnapshot:
    """Broad phase for one unchanged pose. Never retain across reset/step/model edits.

    Keep the original geom_points bounds, candidate order and plane treatment;
    only the repeated vertex transforms and Python box comparisons disappear.
    Exact distances still use pair_gap, including fixed-fixed pairs.
    """

    def __init__(self, scene, geoms):
        self.bounds = boxes(scene, dict.fromkeys(geoms))
        self.lo = np.zeros((scene.m.ngeom, 3))
        self.hi = self.lo.copy()
        for g, (lo, hi) in self.bounds.items():
            self.lo[g], self.hi[g] = lo, hi
        self.planes = scene.m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE

    def near(self, ga, gb, margin):
        if not ga or not gb:
            return []
        ga, gb = np.asarray(ga, int), np.asarray(list(dict.fromkeys(gb)), int)
        lo, hi = self.lo[ga].min(0) - margin, self.hi[ga].max(0) + margin
        hit = self.planes[gb] | ((self.lo[gb] <= hi).all(1) & (self.hi[gb] >= lo).all(1))
        return gb[hit].tolist()


STEP, AXES = 0.001, np.vstack([np.eye(3), -np.eye(3)])


def pair_gap(scene, a, b, distmax):
    """Signed distance between geoms a and b (negative = penetration), capped at distmax. MuJoCo's depth is unreliable
    when two shapes just touch: a lamp base standing flush on a desk can read 76 mm deep (the long way out, down through
    the desk), and which flush pairs do so changes with the platform and with nanometre moves. A signed distance changes
    no faster than the shape moves, so if a moved 1 mm along an axis is v from b, it is at least v - 1 mm from b now. A
    penetration counts only as deep as the shallowest of MuJoCo's reading and those six bounds."""
    m, d = scene.m, scene.d
    v = mujoco.mj_geomDistance(m, d, a, b, distmax, None)
    if v >= 0:
        return v
    x = d.geom_xpos[a].copy()
    try:
        for u in AXES * STEP:
            d.geom_xpos[a] = x + u
            v = max(v, mujoco.mj_geomDistance(m, d, a, b, distmax, None) - STEP)
    finally:
        d.geom_xpos[a] = x
    return v


def gap(scene, ga, gb, distmax=0.1, snapshot=None):
    """Smallest signed distance between two geom sets (negative = penetration), capped at distmax."""
    best = distmax
    snapshot = snapshot if snapshot is not None else GeometrySnapshot(scene, list(ga) + list(gb))
    for a in ga:
        for b in snapshot.near([a], gb, distmax):
            best = min(best, pair_gap(scene, a, b, distmax))
    return best


def supported_by(scene, a, gb, tol):
    """Local upward collision contact, including rim/shoulder support above the object's lowest point."""
    from sbv.geometry.support import SupportIndex

    return SupportIndex(scene, tol).on(a, gb)


def relation(rel, scene, corr, tol, support=None):
    a = corr.get(rel["a"])
    b = corr.get(rel["b"]) if rel.get("b") not in (None, "world") else rel.get("b")
    if a is None or ("b" in rel and b is None):
        return False
    gb = world_geoms(scene) if b == "world" else ([g for g in scene.ogeoms[b] if scene.collides(g)] if b else [])
    t = rel["type"]
    if t == "on":
        return support.on(a, gb) if support is not None else supported_by(scene, a, gb, tol)
    if t == "distance":  # {"min": m, "max": m}: gap between the two objects' geometry (b may be "world")
        g = gap(scene, [g for g in scene.ogeoms[a] if scene.collides(g)], gb, distmax=rel["max"] + 0.01)
        return rel["min"] <= g <= rel["max"]
    if b == "world" or b is None:
        return False  # inside, relative and facing need an object b
    ca = scene.center(a)
    if t == "inside":
        lo, hi = scene.aabb(b)
        return bool((lo - tol <= ca).all() and (ca <= hi + tol).all())
    cb = scene.center(b)
    if t == "relative":  # {"axis": 0|1|2, "sign": +1|-1, "margin": m}: a beyond b along an axis
        return bool((ca - cb)[rel["axis"]] * rel["sign"] > rel.get("margin", 0.0))
    if t == "facing":  # front = the root body's +x axis (submission convention)
        f = scene.d.xmat[scene.objects[a]].reshape(3, 3)[:2, 0]
        v = (cb - ca)[:2]
        return bool(
            np.degrees(np.arccos(np.clip(f @ v / max(np.linalg.norm(f) * np.linalg.norm(v), 1e-9), -1, 1)))
            <= rel.get("tol_deg", 30)
        )
    raise ValueError(f"unknown relation type {t}")


def relations(rels, scene, corr, tol, pen=None):
    """pen: the case's penetration_tol; an "on" relation then follows placement's rule for support through a
    penetration."""
    from sbv.geometry.support import SupportIndex

    support = SupportIndex(scene, tol, pen) if any(r["type"] == "on" for r in rels) else None
    res = [dict(r, passed=relation(r, scene, corr, tol, support)) for r in rels]
    return dict(score=sum(r["passed"] for r in res) / len(res) if res else None, relations=res)


def touching(rel, scene, corr, settled=True):
    """Each "on" relation of relations()'s report also needs the item to rest on its support by contact once the copy
    has settled (scene at its settled start); a hovering or fallen item fails it. Updates rel in place: each "on" row
    gets touching and passed, and the score. settled False (the copy's settle diverged): nothing touches."""
    from sbv.geometry.support import contact_edges

    scene.reset()
    own = {int(b): n for n, b in scene.objects.items()}
    m = scene.m
    pairs = (
        {
            (own.get(int(scene.owner[m.geom_bodyid[s]])), own.get(int(scene.owner[m.geom_bodyid[w]])))
            for s, w in contact_edges(scene)
        }
        if settled
        else set()
    )
    for r in rel["relations"]:
        if r["type"] != "on":
            continue
        a, b = corr.get(r["a"]), corr.get(r["b"])
        r["touching"] = bool(a and b and (b, a) in pairs)
        r["passed"] = bool(r["passed"] and r["touching"])
    rel["score"] = sum(r["passed"] for r in rel["relations"]) / len(rel["relations"]) if rel["relations"] else None


ROOT_SPAN, CHAIN_NEAR, CHAIN_SPAN = 0.02, 0.02, 0.025  # placement: support gap slopes (m), see placement()


def chain_roots(parents, n):
    """The bottom of n's support chain: following supporters down through objects with local support, the objects with
    none (n itself if it has none; a loop resting on nothing has none)."""
    seen, todo, out = set(), [n], []
    while todo:
        x = todo.pop()
        if x in seen or x == "world":
            continue
        seen.add(x)
        if parents[x]:
            todo.extend(parents[x])
        else:
            out.append(x)
    return sorted(out)


def placement(spec, slots, scene, corr):
    """Every generated object, plus every missing required slot (which fails): a contact path to architecture (or a
    frozen, touching mount), no penetration deeper than pen_tol, and its box inside the room bounds. Penetration is
    measured as signed distance between geoms, not from contacts: MuJoCo makes no contacts between two fixed bodies, so
    a cabinet sunk into a wall would otherwise pass. A row with all three scores 1. A row with no path but no
    penetration and in bounds scores by its chain root's gap g to support (SupportIndex.gaps, read up to 10 cm): the
    root itself 1 if g <= support_tol, else 1 - (g - support_tol) / ROOT_SPAN; each object resting on that chain 1 if g
    <= CHAIN_NEAR, else 1 - (g - CHAIN_NEAR) / CHAIN_SPAN; never below 0. A floating base costs only itself, and by a
    slope, not a 1 cm cliff."""
    tol, pen, btol = spec["support_tol"], spec["penetration_tol"], spec["bounds_tol"]
    lo_b, hi_b = map(np.asarray, spec["bounds"])
    from sbv.geometry.support import SupportIndex

    arch = world_geoms(scene)
    scene.reset(scene.q0)
    index = SupportIndex(scene, tol, pen)
    parents, paths = index.graph(spec.get("mounts", []), corr)
    gaps = index.gaps(parents, paths)
    snapshot = GeometrySnapshot(scene, arch + [g for gs in scene.ogeoms.values() for g in gs if scene.collides(g)])
    exempt = set(spec.get("exempt", []))
    slot_of = {v: k for k, v in corr.items() if isinstance(k, str)}
    rows = []
    for n in sorted(scene.objects):
        if slot_of.get(n) in exempt:
            continue  # the reference itself fails here: not a usable criterion
        own = [g for g in scene.ogeoms[n] if scene.collides(g)]
        others = arch + [g for k in scene.objects if k != n for g in scene.ogeoms[k] if scene.collides(g)]
        sup = n in paths
        lo, hi = scene.aabb(n)
        rows.append(
            dict(
                object=n,
                support=bool(sup),
                support_local=bool(parents[n]),
                support_path=paths.get(n, []),
                support_reason="grounded_contact_path" if sup else "no_path_to_architecture_or_authorized_mount",
                non_penetration=bool(gap(scene, own, others, 0.01, snapshot=snapshot) >= -pen),
                in_bounds=bool((lo >= lo_b - btol).all() and (hi <= hi_b + btol).all()),
            )
        )
    for o in slots:
        if o["slot"] not in corr and o["slot"] not in exempt:
            rows.append(dict(object=o["slot"], missing=True, support=False, non_penetration=False, in_bounds=False))
    for r in rows:
        r["passed"] = r["support"] and r["non_penetration"] and r["in_bounds"]
        r["score"] = float(r["passed"])
        if r["support"] or r.get("missing"):
            continue
        roots = chain_roots(parents, r["object"])
        g, root = min(
            ((gaps[x], x) for x in roots if gaps.get(x) is not None), default=(None, roots[0] if roots else None)
        )
        r.update(support_gap=g, support_root=root)
        if g is not None and r["non_penetration"] and r["in_bounds"]:
            near, span = (CHAIN_NEAR, CHAIN_SPAN) if parents[r["object"]] else (tol, ROOT_SPAN)
            r["score"] = min(1.0, max(0.0, 1 - (g - near) / span))
    rate = lambda k: sum(r[k] for r in rows) / len(rows) if rows else None
    return dict(
        score=rate("score"),
        support=rate("support"),
        non_penetration=rate("non_penetration"),
        in_bounds=rate("in_bounds"),
        objects=rows,
    )


def mounts(spec, scene):
    """The reference room's mounts, for calibration: spec's own, plus one for each fixed object with no support path
    that sinks into nothing deeper than penetration_tol and touches the building, or a fixed object with a path, within
    support_tol (SupportIndex.mounted): a picture, clock, shelf or sink on a wall. The building is tried first, then the
    fixed objects in name order; repeated while new mounts give new paths (a clock on a mounted shelf). Without a mount
    such an object is exempt from placement, and so is whatever copy object pairs to it."""
    from sbv.geometry.support import SupportIndex

    arch = world_geoms(scene)
    scene.reset(scene.q0)
    index = SupportIndex(scene, spec["support_tol"], spec["penetration_tol"])
    m = scene.m
    out = list(spec.get("mounts", []))
    paths = index.graph(out)[1]
    own = {n: [g for g in scene.ogeoms[n] if scene.collides(g)] for n in scene.objects}
    snapshot = GeometrySnapshot(scene, arch + [g for gs in own.values() for g in gs])
    fixed = [n for n in sorted(scene.objects) if m.body_weldid[scene.objects[n]] == 0]
    left = [
        n
        for n in fixed
        if n not in paths
        and gap(scene, own[n], arch + [g for k in scene.objects if k != n for g in own[k]], 0.01, snapshot=snapshot)
        >= -spec["penetration_tol"]
    ]
    while left:
        new = [
            dict(a=n, b=b)
            for n in left
            for b in [
                next((b for b in ["world"] + [k for k in fixed if k != n and k in paths] if index.mounted(n, b)), None)
            ]
            if b
        ]
        if not new:
            break
        out += new
        paths = index.graph(out)[1]
        left = [n for n in left if n not in paths]
    return out


def occupancy(scene, bounds):
    """Conservative collision-only obstacle cells from 5 cm to 1.6 m.
    AABBs cover finite collision geoms, including tall cabinets. Infinite planes
    block their forbidden half-space. Visual-only geoms never obstruct a cell.
    """
    scene.reset(scene.q0)
    m, d = scene.m, scene.d
    (x0, y0, _), (x1, y1, _) = bounds
    xs, ys = np.arange(x0 + CELL / 2, x1, CELL), np.arange(y0 + CELL / 2, y1, CELL)
    occ = np.zeros((len(xs), len(ys)), bool)
    for g in range(m.ngeom):
        if not scene.collides(g) or scene.name(scene.owner[m.geom_bodyid[g]]).startswith("probe_"):
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            normal = d.geom_xmat[g].reshape(3, 3)[:, 2]
            xx, yy = np.meshgrid(xs, ys, indexing="ij")
            signed = (
                normal[0] * (xx - d.geom_xpos[g, 0])
                + normal[1] * (yy - d.geom_xpos[g, 1])
                + min(normal[2] * (LOW - d.geom_xpos[g, 2]), normal[2] * (HIGH - d.geom_xpos[g, 2]))
            )
            occ |= signed < CELL * np.linalg.norm(normal[:2]) / 2
            continue
        pts = scene.geom_points(g)
        lo, hi = pts.min(0), pts.max(0)
        if hi[2] < LOW or lo[2] > HIGH:
            continue
        ii, jj = (
            np.where((xs >= lo[0] - CELL / 2) & (xs <= hi[0] + CELL / 2))[0],
            np.where((ys >= lo[1] - CELL / 2) & (ys <= hi[1] + CELL / 2))[0],
        )
        if len(ii) and len(jj):
            occ[ii[0] : ii[-1] + 1, jj[0] : jj[-1] + 1] = True
    return xs, ys, occ


def ground_mask(scene, xs, ys, floor_z=0.0, height_tol=0.02):
    """Fixed, collidable, level surface near the planar robot's z=0 floor, whatever its body is called.
    Five samples per 5 cm cell; the grid cannot certify sub-cell holes/stairs.
    Names never establish support; table tops are excluded by height, movable bodies by their joints.
    """
    from sbv.common.rays import first_hit

    m = scene.m  # a free or articulated platform is not ground, even if it is named floor
    gs = [
        g
        for g in range(m.ngeom)
        if scene.collides(g)
        and m.body_weldid[m.geom_bodyid[g]] == 0
        and not scene.name(scene.owner[m.geom_bodyid[g]]).startswith("probe_")
    ]
    bounds = boxes(scene, gs)
    mask = np.zeros((len(xs), len(ys)), bool)
    down = np.array([0.0, 0.0, -1.0])
    offsets = [(0, 0)] + [(a * CELL / 2, b * CELL / 2) for a in (-1, 1) for b in (-1, 1)]
    planes = {g for g in gs if scene.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE}
    candidates = [
        g
        for g in gs
        if g in planes or bounds[g][0][2] <= floor_z + height_tol and bounds[g][1][2] >= floor_z - height_tol
    ]
    # Each candidate is listed in the cells its box reaches (planes in all), in candidate order: the same per-cell lists
    # as testing every candidate's box against every cell, without 170 box tests per cell on a 77,000-cell room.
    nearby = [[] for _ in range(len(xs) * len(ys))]
    for g in candidates:
        if g in planes:
            cells = range(len(nearby))
        else:
            lo, hi = bounds[g]
            ii = np.flatnonzero((lo[0] <= xs + CELL / 2) & (hi[0] >= xs - CELL / 2))
            jj = np.flatnonzero((lo[1] <= ys + CELL / 2) & (hi[1] >= ys - CELL / 2))
            cells = (ii[:, None] * len(ys) + jj).ravel()
        for c in cells:
            nearby[c].append(g)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            near_g = nearby[i * len(ys) + j]
            if not near_g:
                continue
            ok = True
            for dx, dy in offsets:
                hit = first_hit(scene, near_g, np.array([x + dx, y + dy, floor_z + height_tol + 0.001]), down)
                if hit is None or abs(height_tol + 0.001 - hit[0]) > height_tol or hit[2][2] < 0.98:
                    ok = False
                    break
            mask[i, j] = ok
    return mask


def floor_plan(scene, bounds, radius):
    """Free space needs both floor support and obstacle clearance for the base."""
    xs, ys, occ = occupancy(scene, bounds)
    ground = ground_mask(scene, xs, ys)
    margin = radius + CELL * np.sqrt(2) / 2
    k = max(1, int(np.ceil(margin / CELL)))
    yy, xx = np.mgrid[-k : k + 1, -k : k + 1]
    blocked = ndimage.binary_dilation(occ | ~ground, structure=(xx**2 + yy**2) * CELL**2 <= margin**2, border_value=1)
    return xs, ys, ~blocked


def navigation_start(scene, bounds, radius, preferred):
    """Choose a supported reference start; fail calibration clearly if none exists."""
    xs, ys, free = floor_plan(scene, bounds, radius)
    cells = np.argwhere(free)
    if not len(cells):
        raise ValueError(
            "no supported navigation start: provide fixed architectural collision ground at z=0 "
            "inside the room bounds, with clearance for the robot footprint"
        )
    pts = np.c_[xs[cells[:, 0]], ys[cells[:, 1]]]
    return pts[np.linalg.norm(pts - preferred, axis=1).argmin()].round(3).tolist()


def navigation(spec, scene, corr):
    """Each route passes if the free floor connected to the start comes within `approach` of the goal object's
    footprint."""
    if not spec.get("routes"):
        return dict(score=None, routes=[])
    xs, ys, free = floor_plan(scene, spec["bounds"], spec["base_radius"])
    cells = np.argwhere(free)
    res = []
    pts = np.c_[xs[cells[:, 0]], ys[cells[:, 1]]] if len(cells) else np.zeros((0, 2))
    dist = np.linalg.norm(pts - spec["start"], axis=1) if len(pts) else np.zeros(0)
    if len(pts) and dist.min() <= 0.5:
        lab = ndimage.label(free)[0]
        s = cells[dist.argmin()]
        comp = lab == lab[s[0], s[1]]
        reach = np.c_[xs[np.argwhere(comp)[:, 0]], ys[np.argwhere(comp)[:, 1]]]
    else:
        reach = np.zeros((0, 2))
    for r in spec["routes"]:
        g = corr.get(r["goal"])
        ok = False
        if g is not None and len(reach):
            lo, hi = scene.aabb(g)
            ok = bool(
                np.linalg.norm(np.maximum(0, np.maximum(lo[:2] - reach, reach - hi[:2])), axis=1).min() <= r["approach"]
            )
        res.append(dict(r, passed=ok))
    return dict(score=sum(r["passed"] for r in res) / len(res), routes=res, start_connected=bool(len(reach)))
