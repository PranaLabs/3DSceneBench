"""Conservative interval clearance for prescribed motion, with explicit uncertainty."""

import numpy as np
import mujoco
from sbv.common.collision import distance

SUPPORTED = {
    int(x)
    for x in (
        mujoco.mjtGeom.mjGEOM_PLANE,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        mujoco.mjtGeom.mjGEOM_BOX,
        mujoco.mjtGeom.mjGEOM_MESH,
    )
}


class CollisionPolicy:
    """Respect explicit pairs/excludes and masks, but do not blanket-ignore parents."""

    def __init__(self, m):
        self.m = m
        self.pairs = {tuple(sorted((int(a), int(b)))) for a, b in zip(m.pair_geom1, m.pair_geom2)}
        self.excludes = set(int(x) for x in m.exclude_signature)
        self.paired_geoms = {g for pair in self.pairs for g in pair}

    def active(self, g):
        return g in self.paired_geoms or bool(self.m.geom_contype[g] or self.m.geom_conaffinity[g])

    def allows(self, a, b):
        m = self.m
        if tuple(sorted((a, b))) in self.pairs:
            return True
        x, y = sorted((int(m.geom_bodyid[a]), int(m.geom_bodyid[b])))
        if x == y or (x << 16) + y in self.excludes:
            return False
        return bool((m.geom_contype[a] & m.geom_conaffinity[b]) or (m.geom_contype[b] & m.geom_conaffinity[a]))


def certify(m, d, pairs, speeds, pose, tol, max_evaluations=20000, resolution=1e-4):
    """Certify no pair penetrates >tol on fraction [0,1].

    pose(t) updates *scratch* d; speeds[k] bounds the sum of pair point speeds
    per unit fraction. A midpoint distance minus half-interval travel bounds
    every unsampled point. Unsupported/budget-limited intervals never pass.
    The guarantee is relative to the distance oracle and prescribed path.
    A sampled depth fails only above tol plus rounding (1 nm); between samples
    the bound is resolved to `resolution` above tol. Without that margin a
    part resting exactly at tol along a stretch of its path (a drawer front
    flush 2 mm into its carcass sides) can never be certified.
    """
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    speeds = np.asarray(speeds, float)
    result = dict(status="clear", evaluations=0, worst_penetration=0.0)
    if not len(pairs):
        return result
    if not np.isfinite(speeds).all() or (speeds < 0).any():
        return dict(result, status="unresolved", reason="invalid_motion_bound")
    if any(int(m.geom_type[g]) not in SUPPORTED for g in pairs.ravel()):
        return dict(result, status="unresolved", reason="unsupported_collision_geometry")

    def measure(t, ids):
        if result["evaluations"] + len(ids) > max_evaluations:
            return None
        pose(t)
        values = np.array([distance(m, d, *pairs[k], max(1.0, speeds[k] + tol)) for k in ids])
        result["evaluations"] += len(ids)
        if not np.isfinite(values).all():
            return None
        k = int(np.argmin(values))
        result["worst_penetration"] = max(result["worst_penetration"], float(-values[k]))
        if values[k] < -tol - 1e-9:
            result.update(
                status="collision", fraction=float(t), geoms=pairs[ids[k]].tolist(), penetration=float(-values[k])
            )
        return values

    ids = np.arange(len(pairs))
    for t in (0.0, 1.0):
        v = measure(t, ids)
        if v is None:
            return dict(result, status="unresolved", reason="evaluation_budget")
        if result["status"] == "collision":
            return result
    todo = [(0.0, 1.0, ids)]
    while todo:
        lo, hi, ids = todo.pop()
        mid = (lo + hi) / 2
        v = measure(mid, ids)
        if v is None:
            return dict(result, status="unresolved", reason="evaluation_budget", interval=[lo, hi])
        if result["status"] == "collision":
            return result
        pending = ids[v + tol + resolution < speeds[ids] * (hi - lo) / 2]
        if len(pending):
            if hi - lo < 1e-9:
                return dict(result, status="unresolved", reason="resolution_limit", interval=[lo, hi])
            todo.extend([(mid, hi, pending), (lo, mid, pending)])
    return result
