"""Ray queries over an explicit geom set, without visual-geometry interference."""

import numpy as np
import mujoco


def first_hit(scene, geoms, p, u):
    m, d = scene.m, scene.d
    best = None
    for g in geoms:
        normal = np.zeros(3)
        t = int(m.geom_type[g])
        if t == int(mujoco.mjtGeom.mjGEOM_MESH):
            x = mujoco.mj_rayMesh(m, d, g, p, u, normal)
        elif t == int(mujoco.mjtGeom.mjGEOM_HFIELD):
            x = mujoco.mj_rayHfield(m, d, g, p, u, normal)
        elif t == int(mujoco.mjtGeom.mjGEOM_SDF):
            continue  # unsupported, never a fabricated hit
        else:
            x = mujoco.mju_rayGeom(d.geom_xpos[g], d.geom_xmat[g], m.geom_size[g], p, u, t, normal)
        if x >= 0 and (best is None or x < best[0]):
            best = (float(x), int(g), normal)
    return best
