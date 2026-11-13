"""Line a copy's object up with the reference object by shape, so a frame frozen in the reference room lands on the
copy. Copies choose their own body origin and axes; only the collision surface is comparable. Upright objects only:
rotation about z plus a translation, from 12 starting headings, refined by nearest-point fitting."""

import numpy as np
from scipy.spatial import cKDTree


def collision_points(scene, n, count=600, seed=0):
    """Area-weighted samples of an object's collision surface, in world coordinates."""
    from sbv.geometry.g2_shape import geom_mesh

    tris = [
        r[0][r[1]]
        for r in (geom_mesh(scene, g) for g in scene.ogeoms[n] if scene.collides(g))
        if r is not None and len(r[1])
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


def upright_fit(src, dst, headings=12, iters=25, tie=0.002):
    """R (about z), t and the symmetric RMS gap with dst ~ src @ R.T + t. A shape that looks the same turned (a square
    box) fits several turns about equally well; of the fits within `tie` metres of the best, the least turned one wins,
    so the copy is read in the reference room's orientation unless its shape says otherwise."""
    ts, td = cKDTree(src), cKDTree(dst)
    fits = []
    for a in np.arange(headings) * 2 * np.pi / headings:
        c, s = np.cos(a), np.sin(a)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        t = dst.mean(0) - src.mean(0) @ R.T
        for _ in range(iters):
            q = dst[td.query(src @ R.T + t)[1]]
            p0, q0 = src.mean(0), q.mean(0)
            U, _, Vt = np.linalg.svd((src - p0)[:, :2].T @ (q - q0)[:, :2])
            r = Vt.T @ U.T
            if np.linalg.det(r) < 0:
                Vt[1] *= -1
                r = Vt.T @ U.T
            R = np.eye(3)
            R[:2, :2] = r
            t = q0 - p0 @ R.T
        moved = src @ R.T + t
        err = np.sqrt((np.mean(td.query(moved)[0] ** 2) + np.mean(cKDTree(moved).query(dst)[0] ** 2)) / 2)
        fits.append((R, t, float(err)))
    best = min(f[2] for f in fits)
    return min((f for f in fits if f[2] <= best + tie), key=lambda f: abs(np.arctan2(f[0][1, 0], f[0][0, 0])))
