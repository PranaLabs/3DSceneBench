"""Geometry axis, G2 object geometry (doc Geometry and Appendix A): surface F-score and Chamfer distance on matched
objects; pose-and-size success. Measured on the submitted arrangement, before any settling."""

import math
import numpy as np, mujoco, trimesh
from scipy.spatial import cKDTree, ConvexHull

T = mujoco.mjtGeom
SIZE_BARS = (1.25, 2.0)  # G2.3 pass bars on each side's length, main first: factor 1.25 is the convention for geometry
# measured from pictures (depth delta1); factor 2 is reported beside it (VERIFIERS.md, "Pass bars")


# ---- G2 shape --------------------------------------------------------------------------------------------------------
def geom_mesh(scene, g):
    m, d = scene.m, scene.d
    t, s = m.geom_type[g], m.geom_size[g]
    if t == T.mjGEOM_MESH:
        i = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[i] : m.mesh_vertadr[i] + m.mesh_vertnum[i]]
        f = m.mesh_face[m.mesh_faceadr[i] : m.mesh_faceadr[i] + m.mesh_facenum[i]]
    else:
        if t == T.mjGEOM_BOX:
            tm = trimesh.creation.box(extents=2 * s[:3])
        elif t == T.mjGEOM_SPHERE:
            tm = trimesh.creation.icosphere(subdivisions=2, radius=s[0])
        elif t == T.mjGEOM_ELLIPSOID:
            tm = trimesh.creation.icosphere(subdivisions=2)
            tm.vertices *= s[:3]
        elif t == T.mjGEOM_CYLINDER:
            tm = trimesh.creation.cylinder(radius=s[0], height=2 * s[1], sections=24)
        elif t == T.mjGEOM_CAPSULE:
            tm = trimesh.creation.capsule(height=2 * s[1], radius=s[0], count=[24, 12])
        else:
            return None
        v, f = np.asarray(tm.vertices), np.asarray(tm.faces)
    return v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g], f


def surface_samples(scene, n, count=3000, seed=0):
    """Area-weighted surface samples of what a camera sees of an object (Scene.visible). Planes, heightfields and SDF
    geoms have no closed surface and are left out."""
    tris = [
        r[0][r[1]] for r in (geom_mesh(scene, g) for g in scene.visible(scene.ogeoms[n])) if r is not None and len(r[1])
    ]
    if not tris:
        return np.zeros((0, 3))
    tri = np.concatenate(tris)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    k = rng.choice(len(tri), count, p=area / area.sum())
    r1, r2 = rng.random((2, count))
    s = np.sqrt(r1)[:, None]
    t = tri[k]
    return (1 - s) * t[:, 0] + s * (1 - r2[:, None]) * t[:, 1] + s * r2[:, None] * t[:, 2]


def oriented_box(pts):
    """The smallest-area rectangle around the footprint (tried along each convex-hull edge) and the height: (centre,
    angle of the rectangle's first side, [length along it, width across it, height]). It does not change when the object
    is turned about z, unlike principal axes, which wander with sampling noise on near-square objects."""
    xy = pts[:, :2]
    best = None
    try:
        hull = xy[ConvexHull(xy).vertices]
    except Exception:
        hull = xy  # a line or a point: no 2D hull
    for i in range(len(hull)):
        e = hull[(i + 1) % len(hull)] - hull[i]
        if np.linalg.norm(e) < 1e-9:
            continue
        a = math.atan2(e[1], e[0])
        r = hull @ np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        lo, hi = r.min(0), r.max(0)
        if best is None or np.prod(hi - lo) < best[0] - 1e-12:
            best = (float(np.prod(hi - lo)), a, lo, hi)
    a, lo, hi = (0.0, xy.min(0), xy.max(0)) if best is None else best[1:]
    m = (lo + hi) / 2
    c, s = math.cos(a), math.sin(a)
    z0, z1 = float(pts[:, 2].min()), float(pts[:, 2].max())
    return (
        np.array([c * m[0] - s * m[1], s * m[0] + c * m[1], (z0 + z1) / 2]),
        a,
        [float(hi[0] - lo[0]), float(hi[1] - lo[1]), z1 - z0],
    )


def normalized(pts):
    """Centred on its footprint rectangle, turned so the rectangle lies along x and y, scaled to unit box diagonal."""
    c, a, ext = oriented_box(pts)
    R = np.array([[math.cos(-a), -math.sin(-a), 0], [math.sin(-a), math.cos(-a), 0], [0, 0, 1]])
    return (pts - c) @ R.T / max(float(np.linalg.norm(ext)), 1e-9)


def shape_scores(ref, gen, tau):
    """Surface precision, recall, F-score at tau and unsquared Chamfer distance, after removing placement, turn about z
    and one uniform scale (normalized); the best of the four quarter turns that remain, since a rectangle has four."""
    a, b0 = normalized(ref), normalized(gen)
    ta = cKDTree(a)
    best = None
    for k in range(4):
        c, s = math.cos(k * math.pi / 2), math.sin(k * math.pi / 2)
        b = b0 @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T
        dg, dr = ta.query(b)[0], cKDTree(b).query(a)[0]
        P, R = float((dg < tau).mean()), float((dr < tau).mean())
        out = dict(
            fscore=0.0 if P + R == 0 else 2 * P * R / (P + R),
            precision=P,
            recall=R,
            chamfer=float((dg.mean() + dr.mean()) / 2),
        )
        if best is None or out["chamfer"] < best["chamfer"]:
            best = out
    return best


def oriented_dims(pts):
    """Length and width of the smallest footprint rectangle (longer first), then height: independent of how the object
    is turned and of its body-frame convention."""
    e = oriented_box(pts)[2]
    return [max(e[0], e[1]), min(e[0], e[1]), e[2]]


# --- G2 pose and size ------------------------------------------------------------------------------------------------
# G2.3 credit: full within a check's bar, then a half cosine down to 0 at RAMP times the bar (size: at
RAMP = 2.0
# the reference bar, factor 2), so a side 1.3 times the true length earns most of the credit instead of none
LONG_ASPECT, LONG_DIAG, LONG_TOL, LONG_COPY = (
    1.5,
    0.5,
    30.0,
    1.1,
)  # a true footprint 1.5 times as long as wide on a box diagonal
# of 0.5 m or more: the copy's footprint long axis within 30 degrees of the true one, unless the copy's own footprint is
# within 10% of square (its long axis is then sampling noise, and its sides already answer to the size bar)


def yaw(scene, n):
    R = scene.d.xmat[scene.objects[n]].reshape(3, 3)
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


def long_axis(pts):
    """Direction of the smallest footprint rectangle's longer side (oriented_box), in degrees modulo 180."""
    _, a, e = oriented_box(pts)
    return (math.degrees(a) + (0.0 if e[0] >= e[1] else 90.0)) % 180


def ramp(x, bar, end):
    """Credit for an error x against a bar: 1 up to bar, 0 from end on, a half cosine between."""
    if x <= bar:
        return 1.0
    if x >= end:
        return 0.0
    return 0.5 * (1 + math.cos(math.pi * (x - bar) / (end - bar)))


def pose_size(reqs, slots, scene, corr, samples, ref_samples):
    """Per required object: position within tol of the reference centre, every side within the size bar (a ratio: factor
    k passes 1/k to k times the true length), the declared orientation constraint if any, and on a long object (true
    footprint LONG_ASPECT times as long as wide or more, box diagonal LONG_DIAG or more, copy footprint not within 10%
    of square) the copy's footprint long axis within LONG_TOL degrees of the true one (ref_samples, the reference room's
    samples by slot), modulo 180: sizes are sorted
    and shape tries four quarter turns, so neither sees a bookcase turned 90 degrees. An object passes only if present
    and every applicable check passes. size_bars holds the main bar, then the reference bar reported beside it. The
    score is smooth: each check earns ramp() credit, full within its bar and 0 at RAMP times it (size on the log of the
    ratio, so 0 at the reference bar), and an object earns the product of its checks' credits, 0 if missing. passed, the
    hard verdict at the main bar, and its share score_passed stay in the report."""
    ref = {o["slot"]: o for o in slots}
    rows = []
    for r in reqs:
        g, o = corr.get(r["slot"]), ref[r["slot"]]
        row = dict(slot=r["slot"], present=g is not None)
        if g is not None:
            row["position_error"] = float(np.linalg.norm(scene.center(g) - np.asarray(o["center"])))
            dims = oriented_dims(samples[g]) if len(samples[g]) else [0.0, 0.0, 0.0]
            row["dimension_errors"] = [
                abs(x / y - 1) if y >= 0.005 else None for x, y in zip(dims, o["dims"])
            ]  # a flat poster has no depth
            row["size_ratio"] = max(
                [max(x / y, y / x) if x > 0 else math.inf for x, y in zip(dims, o["dims"]) if y >= 0.005] or [1.0]
            )
            checks = [row["position_error"] <= r["position_tol"]]
            credit = ramp(row["position_error"], r["position_tol"], RAMP * r["position_tol"])
            credit *= ramp(
                math.log(row["size_ratio"]), math.log(r["size_bars"][0]), math.log(r["size_bars"][-1])
            )  # 1.25 to 2: 0.5 at 1.58
            if r.get("yaw_tol") is not None:
                sym = 360.0 / r.get("symmetry", 1)
                e = abs((yaw(scene, g) - o["yaw"] + sym / 2) % sym - sym / 2)
                row["orientation_error"] = e
                checks.append(e <= r["yaw_tol"])
                credit *= ramp(e, r["yaw_tol"], RAMP * r["yaw_tol"])
            t, c = (
                o["dims"],
                ref_samples.get(r["slot"], ()),
            )  # small items left out: measurements.json says "longer first", not which way
            if (
                t[0] + 1e-6 >= LONG_ASPECT * t[1]
                and math.hypot(*t) >= LONG_DIAG
                and len(c)
                and len(samples[g])
                and dims[0] >= LONG_COPY * dims[1]
            ):  # 1.2 by 0.8 is 1.5 times
                e = abs((long_axis(samples[g]) - long_axis(c) + 90) % 180 - 90)
                row["long_axis_error"] = float(e)
                checks.append(e <= LONG_TOL)
                credit *= ramp(e, LONG_TOL, RAMP * LONG_TOL)
            row["passed"], row["passed_reference_bar"] = (
                all(checks) and row["size_ratio"] <= k for k in r["size_bars"]
            )
            row["score"] = round(credit, 6)
        else:
            row["passed"] = row["passed_reference_bar"] = False
            row["score"] = 0.0
        rows.append(row)
    share = lambda k: sum(float(x[k]) for x in rows) / len(rows) if rows else None
    return dict(
        score=share("score"),
        score_passed=share("passed"),
        score_reference_bar=share("passed_reference_bar"),
        objects=rows,
    )
