"""Helpers for a room's package task code (schema.py has the contract)."""

import numpy as np

SIGMA = 0.001  # the same 1 mm the built-in tests use (sbv/common/scene.py jittered)


def jitter(model, data, seed, sigma=SIGMA):
    """Move every free root by up to sigma in x and y, from the run's seed; seed 0 leaves the scene as it is. Returns
    whether anything moved: a run reports it as jitter_applied."""
    if not seed:
        return False
    import mujoco

    rng, moved = np.random.default_rng(seed), False
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        a = model.jnt_qposadr[j]
        data.qpos[a : a + 2] += rng.uniform(-sigma, sigma, 2)
        moved = True
    mujoco.mj_forward(model, data)
    return moved


def body(model, name):
    """The body id of an object the task acts on (ctx["objects"][true name]); a clear error when the scene lacks it."""
    import mujoco

    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if b < 0:
        raise KeyError(f"no body named {name!r} in this scene")
    return b
