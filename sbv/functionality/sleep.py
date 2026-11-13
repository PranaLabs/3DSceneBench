"""Isolation for doors and drawers, with MuJoCo's sleeping islands.

At the first step of each rollout in an isolated block, every contact island holding none of the test's objects, no
probe and nothing restless (moving by itself in the init records) is put to sleep: velocity zeroed and first step
undone, so each position change still matches its velocity (trajectory.py). It wakes when an awake tree touches it.
Nothing leaves the model, so the clearance sweep and rays still see every collider. What shares a contact island with
the test's objects at the settled start stays awake (rest_stays sets a door half open; a cup on it must fall, not hang
asleep). A rollout with other gravity, masses, frictions or collision filters runs awake: the records hold for the
room's own. Sleeping freezes motion below TOL: a body asleep through a 3 s rollout moves under 0.3 mm, the records'
still is 1 mm.
"""

import contextlib
import mujoco
import numpy as np

KINDS = {"joint"}  # doors and drawers; the other kinds already run locally (units.LOCAL_KINDS)
TOL = 1e-4  # m/s; MuJoCo's 1e-3 froze a slow creep into a wrong pass
_SLEEP, _NEVER = int(mujoco.mjtEnableBit.mjENBL_SLEEP), int(mujoco.mjtSleepPolicy.mjSLEEP_NEVER)
PARAMS = ("body_mass", "geom_friction", "geom_contype", "geom_conaffinity")
AWAKE = False  # awake(): the whole room awake, isolation and the robot's sleep stand aside


@contextlib.contextmanager
def awake():
    """Every rollout inside runs with the whole room awake, which is what isolation stands in for. For a copy MuJoCo
    cannot put to sleep: 3.14 stops with FatalError on a tendon equality ("mj_wakeEquality: tendon equality does not yet
    support sleeping") and on some islands ("mj_sleep: found sleeping tree"); evaluate runs such a unit again inside
    this."""
    global AWAKE
    was, AWAKE = AWAKE, True
    try:
        yield
    finally:
        AWAKE = was


def restless(recs):
    """Objects some init record saw moving by themselves (unstable, not still, or a joint drifting): never asleep."""
    return sorted(
        {
            n
            for r in recs
            for n, x in r.items()
            if x["unstable"] or not x.get("still", True) or not x.get("joints_still", True)
        }
    )


def islands(scene):
    """The objects sharing each contact island at the settled start, as sets of names (any link counts); cached per
    start."""
    c = getattr(scene, "_islands", None)
    if c is None or not np.array_equal(c[0], scene.start):
        m = scene.m
        scene.reset()
        isl = scene.d.tree_island[: m.ntree].copy()
        by = {}
        for n, ls in scene.links.items():
            for t in {int(m.body_treeid[k]) for k in ls} - {-1}:
                if isl[t] >= 0:
                    by.setdefault(int(isl[t]), set()).add(n)
        c = scene._islands = (scene.start.copy(), list(by.values()))
    return c[1]


class _Isolation:
    """Scene.on_step: at a rollout's first step, with the room's own parameters, every island without a kept tree
    sleeps."""

    def __init__(self, scene, keep):
        m = scene.m
        self.s = scene
        self.keep = np.zeros(m.ntree, bool)
        self.keep[sorted(keep)] = True
        self.base = [m.opt.gravity.copy()] + [getattr(m, k).copy() for k in PARAMS]

    def __call__(self):
        m, d = self.s.m, self.s.d
        if not 0 < d.time <= 1.5 * m.opt.timestep:
            return
        if not all(np.array_equal(a, b) for a, b in zip(self.base, [m.opt.gravity] + [getattr(m, k) for k in PARAMS])):
            m.opt.enableflags &= ~_SLEEP
            return  # changed parameters: this rollout runs as the whole room
        m.opt.enableflags |= _SLEEP
        isl = d.tree_island[: m.ntree]
        busy = np.unique(isl[self.keep & (isl >= 0)])
        ready = ~self.keep & ((isl < 0) | ~np.isin(isl, busy))
        d.tree_asleep[: m.ntree][ready] = -1
        idx = ready[m.dof_treeid]
        if idx.any():  # back at rest at the start pose, as the records saw them
            mujoco.mj_integratePos(m, d.qpos, np.where(idx, d.qvel, 0.0), -m.opt.timestep)
            d.qvel[idx] = 0.0
            mujoco.mj_forward(m, d)


@contextlib.contextmanager
def isolated(scene, names, restless):
    """The block's rollouts keep to what the named objects reach (module docstring); restless None: the whole room
    awake."""
    if restless is None or AWAKE:
        yield
        return
    m = scene.m
    names = set(names) | set(restless)
    for members in islands(scene):
        if members & names:
            names |= members
    probes = set(scene.probe.values())
    keep = {int(m.body_treeid[k]) for k in range(1, m.nbody) if scene.owner[k] in probes} | {
        int(m.body_treeid[k]) for n in names for k in scene.links.get(n, [])
    }
    keep.discard(-1)
    saved = (m.tree_sleep_policy.copy(), int(m.opt.enableflags), float(m.opt.sleep_tolerance))
    try:
        m.tree_sleep_policy[sorted(keep)] = _NEVER
        m.opt.enableflags |= _SLEEP
        m.opt.sleep_tolerance = TOL
        scene.on_step = _Isolation(scene, keep)
        yield
    finally:
        m.tree_sleep_policy[:] = saved[0]
        m.opt.enableflags, m.opt.sleep_tolerance = saved[1:]
        scene.on_step = None
