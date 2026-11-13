"""Geometric distance helpers, independent of whether MuJoCo generates contacts."""

import numpy as np
import mujoco


def distance(m, d, a, b, cap=1.0):
    """Signed separation lower bound; exact separating-axis overlap for box pairs.

    Native GJK can return zero for deeply nested, concentric boxes. SAT avoids
    that degeneracy. For separated boxes its positive result is a lower bound
    on Euclidean distance, which is conservative for clearance certification.
    Other pairs use MuJoCo's distance oracle and its supported geom types.
    """
    if m.geom_type[a] == m.geom_type[b] == mujoco.mjtGeom.mjGEOM_BOX:
        A, B = d.geom_xmat[a].reshape(3, 3), d.geom_xmat[b].reshape(3, 3)
        axes = np.vstack((A.T, B.T, np.cross(A.T[:, None, :], B.T[None, :, :]).reshape(-1, 3)))
        norm = np.linalg.norm(axes, axis=1)
        axes = axes[norm > 1e-10] / norm[norm > 1e-10, None]
        h = np.abs(axes @ A) @ m.geom_size[a] + np.abs(axes @ B) @ m.geom_size[b]
        return float(min(cap, np.max(np.abs(axes @ (d.geom_xpos[b] - d.geom_xpos[a])) - h)))
    return float(mujoco.mj_geomDistance(m, d, a, b, cap, None))
