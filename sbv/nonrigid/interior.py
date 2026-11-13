"""A fluid holder's real inside, as 1.5 mm cubes: the free space reached from the fluid's base without leaving the
holder, up to its rim. Spilled fluid is fluid outside it. The outline (convex hull) used before counted fluid that had
poured out under a holder's neck or beside it as still held.
"""

import numpy as np
import mujoco
from scipy import ndimage
from scipy.spatial import ConvexHull

from .room import _model

CUBE = 0.0015


def _piece(m, d, g):
    """One collision piece: its world box, and a test of how deep points (in its own frame) are inside it."""
    R, p, t, sz, G = (
        d.geom_xmat[g].reshape(3, 3),
        d.geom_xpos[g],
        mujoco.mjtGeom(m.geom_type[g]),
        m.geom_size[g],
        mujoco.mjtGeom,
    )
    if t == G.mjGEOM_MESH:
        v = m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]] :][: m.mesh_vertnum[m.geom_dataid[g]]]
        w, e = v @ R.T + p, ConvexHull(v).equations
        return w.min(0), w.max(0), R, p, lambda q: -np.max(q @ e[:, :3].T + e[:, 3], 1)
    depth = {
        G.mjGEOM_BOX: lambda q: np.min(sz - np.abs(q), 1),
        G.mjGEOM_SPHERE: lambda q: sz[0] - np.linalg.norm(q, axis=1),
        G.mjGEOM_CAPSULE: lambda q: sz[0]
        - np.hypot(np.hypot(q[:, 0], q[:, 1]), np.maximum(np.abs(q[:, 2]) - sz[1], 0)),
        G.mjGEOM_CYLINDER: lambda q: np.minimum(sz[0] - np.hypot(q[:, 0], q[:, 1]), sz[1] - np.abs(q[:, 2])),
        G.mjGEOM_ELLIPSOID: lambda q: (1 - np.sqrt(((q / sz) ** 2).sum(1))) * sz.min(),
    }.get(t, lambda q: np.full(len(q), -1.0))
    c = (
        np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * m.geom_aabb[g][3:]
        + m.geom_aabb[g][:3]
    )
    w = c @ R.T + p
    return w.min(0), w.max(0), R, p, depth


def inside(scene_xml, body, base, slack=2):
    """(lo, cubes) of the holder's inside, grown by `slack` cubes; `base` is a point just above its floor."""
    m, d = _model(scene_xml, {body})
    pieces = [
        _piece(m, d, g)
        for g in range(m.ngeom)
        if (m.geom_contype[g] or m.geom_conaffinity[g]) and m.body_rootid[m.geom_bodyid[g]]
    ]  # the holder's own pieces: copies keep walls and floor in worldbody
    lo = np.min([x[0] for x in pieces], 0) - 3 * CUBE
    solid = np.zeros(np.ceil((np.max([x[1] for x in pieces], 0) + 3 * CUBE - lo) / CUBE).astype(int) + 1, bool)
    for a, b, R, p, depth in pieces:  # each piece fills only the block of cubes it can reach
        i0, i1 = np.floor((a - lo) / CUBE).astype(int), np.ceil((b - lo) / CUBE).astype(int) + 1
        Q = np.stack(np.meshgrid(*[lo[k] + CUBE * np.arange(i0[k], i1[k]) for k in range(3)], indexing="ij"), -1)
        solid[i0[0] : i1[0], i0[1] : i1[1], i0[2] : i1[2]] |= (depth((Q.reshape(-1, 3) - p) @ R) >= -1e-4).reshape(
            Q.shape[:3]
        )
    bottom, top = np.nonzero(solid.any((0, 1)))[0][[0, -1]]
    seed = tuple(np.clip(np.round((np.asarray(base) - lo) / CUBE).astype(int), 0, np.array(solid.shape) - 1))
    if solid[seed]:  # the base is in a solid piece (the holder's floor, a stem standing in it): the nearest free cube
        w, (x, y, z) = 27, seed  # at or above it, within 4 cm (never below, which could be under the holder)
        win = solid[max(x - w, 0) : x + w + 1, max(y - w, 0) : y + w + 1, z : z + w + 1]
        if (~win).any():
            j = ndimage.distance_transform_edt(win, return_indices=True)[1][:, x - max(x - w, 0), y - max(y - w, 0), 0]
            seed = (max(x - w, 0) + j[0], max(y - w, 0) + j[1], z + j[2])

    def region(level):  # free space from the seed below a level, and whether it reaches the box's sides (gets out)
        free = ~solid
        free[:, :, level:] = False
        free[:, :, :bottom] = False
        lab, _ = ndimage.label(free)
        if not lab[seed]:  # still in a solid piece: no inside (never the solid itself)
            return np.zeros_like(free), True
        reg = lab == lab[seed]
        return reg, bool(lab[seed]) and bool(reg[0].any() or reg[-1].any() or reg[:, 0].any() or reg[:, -1].any())

    a, b = seed[2] + 1, top + 1  # the highest level that still closes: the rim
    while b - a > 1:
        a, b = (a, (a + b) // 2) if region((a + b) // 2)[1] else ((a + b) // 2, b)
    reg = region(a)[0]
    return lo, ndimage.binary_dilation(reg, iterations=slack) if slack else reg  # iterations=0 would grow forever


def held_test(lo, cubes):
    """points (in the holder's starting frame) -> True where inside."""

    def held(p):
        i = np.round((np.asarray(p) - lo) / CUBE).astype(int)
        ok = np.all((i >= 0) & (i < np.array(cubes.shape)), 1)
        out = np.zeros(len(i), bool)
        out[ok] = cubes[tuple(i[ok].T)]
        return out

    return held
