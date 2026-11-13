"""Bound maximum penetration over an interpolated rigid-geometry path."""

import mujoco
import numpy as np
from sbv.common.collision import distance
from sbv.common.sweep import SUPPORTED


def depth_bounds(m, d, pairs, speeds, pose, accuracy=0.0005, budget=20000):
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    speeds = np.asarray(speeds, float)
    n = len(pairs)
    observed = np.zeros(n)
    upper = np.zeros(n)
    evaluations = 0
    witness = {}
    endpoints = []
    unresolved = set(
        k
        for k, (a, b) in enumerate(pairs)
        if int(m.geom_type[a]) not in SUPPORTED or int(m.geom_type[b]) not in SUPPORTED
    )
    all_ids = np.array([k for k in range(n) if k not in unresolved], int)
    ids = all_ids

    def measure(t, ids):
        nonlocal evaluations
        if evaluations + len(ids) > budget:
            unresolved.update(ids.tolist())
            return None
        pose(t)
        v = np.array([distance(m, d, *pairs[k], max(1.0, speeds[k] + accuracy)) for k in ids])
        evaluations += len(ids)
        if not np.isfinite(v).all():
            unresolved.update(ids.tolist())
            return None
        for k, x in zip(ids, v):
            if -x > observed[k]:
                observed[k] = -x
                witness[k] = float(t)
        return v

    if len(ids):
        for t in (0.0, 1.0):
            values = measure(t, ids)
            endpoints.append(None if values is None else dict(zip(ids, values)))
    todo = [(0.0, 1.0, ids)] if len(ids) else []
    while todo:
        lo, hi, ids = todo.pop()
        v = measure((lo + hi) / 2, ids)
        if v is None:
            continue
        bound = np.maximum(0.0, -v + speeds[ids] * (hi - lo) / 2)
        done = bound <= observed[ids] + accuracy
        upper[ids[done]] = np.maximum(upper[ids[done]], bound[done])
        pending = ids[~done]
        if len(pending):
            if hi - lo < 1e-9:
                unresolved.update(pending.tolist())
            else:
                todo.extend([((lo + hi) / 2, hi, pending), (lo, (lo + hi) / 2, pending)])
    upper = np.maximum(upper, observed)
    crossed = [
        int(k)
        for k in all_ids
        if len(endpoints) == 2
        and all(e is not None and e[k] >= -1e-6 for e in endpoints)
        and observed[k] > accuracy
        and (observed[k] >= half_thin(m, pairs[k]) or other_side(m, d, pairs[k], pose))
    ]
    return dict(
        observed=observed,
        upper=upper,
        unresolved=sorted(unresolved),
        evaluations=evaluations,
        witness=witness,
        crossed=crossed,
    )


def half_thin(m, pair):
    """Half the thinnest extent of the thinner geom: sunk that deep on the way, a geom went through it (a full turn
    in one step ends where it began, on its own side)."""
    return float(min(m.geom_aabb[int(g)][3:].min() for g in pair))


def other_side(m, d, pair, pose, touch=1e-5):
    """Clear of b at both ends of the step yet overlapping it on the way: a crossing only if a's centre ends on the
    other side of b's surface (the plane at b's nearest point, from the end where they are apart). A box resting on
    another that dips 0.5 mm into it mid-step and lifts off again stays on its side, as does any contact touching at
    both ends."""
    a, b = (int(x) for x in pair)
    ends = []
    for t in (0.0, 1.0):
        pose(t)
        ft = np.zeros(6)
        gap = mujoco.mj_geomDistance(m, d, a, b, 1.0, ft)
        ends.append((gap, ft[:3].copy(), ft[3:].copy(), d.geom_xpos[a].copy()))
    (g0, a0, b0, c0), (g1, a1, b1, c1) = ends
    if g0 > touch:
        return float((a0 - b0) @ (c1 - b0)) < 0  # the plane from where the step starts, a's centre where it ends
    if g1 > touch:
        return float((a1 - b1) @ (c0 - b1)) < 0
    return False
