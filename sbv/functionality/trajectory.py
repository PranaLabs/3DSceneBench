"""Between-step checks against fixed collision geometry on an FK motion path.

Native contacts handle ordinary steps. Larger surface movements are checked
adaptively in scratch data. Solver timestep, forces and live state are unchanged.
"""

import mujoco
import numpy as np
from sbv.common.motion import MotionBounds, pose_path
from sbv.common.scene import sleep_off
from sbv.common.sweep import CollisionPolicy
from sbv.common.swept_depth import depth_bounds

ACCURACY = 0.0005


class Trajectory:
    def __init__(self, scene, owners, ignore, budget=20000):
        self.s, self.owners, self.ignore, self.budget = scene, owners, ignore, budget
        m = scene.m
        self.gowners = scene.owner[m.geom_bodyid]
        self.policy = CollisionPolicy(m)
        self.watched = np.array(
            [
                g
                for g in range(m.ngeom)
                if self.policy.active(g) and self.gowners[g] in owners and m.body_weldid[m.geom_bodyid[g]] != 0
            ],
            int,
        )
        self.static = np.array(
            [
                g
                for g in range(m.ngeom)
                if self.policy.active(g)
                and self.gowners[g] not in ignore
                and m.body_weldid[m.geom_bodyid[g]] == 0
                and not scene.name(self.gowners[g]).startswith("probe_")
            ],
            int,
        )
        self.bounds = MotionBounds(m, scene.d.qpos, self.watched)
        with sleep_off(m):  # scratch data never steps (sleep_off)
            self.scratch = mujoco.MjData(m)
            self.scratch.qpos[:] = scene.d.qpos
            mujoco.mj_kinematics(m, self.scratch)
        self.static_pos = self.scratch.geom_xpos[self.static].copy()
        self.static_rot = self.scratch.geom_xmat[self.static].copy()
        R = self.static_rot.reshape(-1, 3, 3)
        self.centres = self.static_pos + np.einsum("nij,nj->ni", R, m.geom_aabb[self.static, :3])
        self.halves = np.einsum("nij,nj->ni", np.abs(R), m.geom_aabb[self.static, 3:])
        self.planes = m.geom_type[self.static] == mujoco.mjtGeom.mjGEOM_PLANE
        self.q0 = scene.d.qpos.copy()
        self.t0 = float(scene.d.time)
        self.prepared = False
        self.pen = {o: 0.0 for o in owners}
        self.upper = dict(self.pen)
        self.events = []
        self.crossings = {}
        self.unresolved = set()
        self.reasons = {}
        self.evaluations = 0
        self.segments = 0
        self.fast_steps = 0
        self.mark_unresolved(
            [int(self.gowners[g]) for g in self.watched if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE],
            "moving_plane",
        )
        if m.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4:
            self.mark_unresolved(owners, "unsupported_integrator_RK4")

    def mark_unresolved(self, owners, reason):
        for o in owners:
            self.unresolved.add(o)
            self.reasons.setdefault(o, set()).add(reason)

    def before_step(self):
        self.q0[:] = self.s.d.qpos
        self.t0 = float(self.s.d.time)
        self.prepared = True

    def update(self, native):
        with sleep_off(self.s.m):
            self._update(native)

    def _update(self, native):
        m, d = self.s.m, self.s.d
        dt = float(d.time) - self.t0
        if not self.prepared or dt <= 0:
            self.mark_unresolved(self.owners, "missing_step_start")
            return
        self.prepared = False
        # Euler/implicit integrators integrate the just-computed velocity. Using
        # that displacement preserves offset hinge arcs and >180-degree turns.
        delta = d.qvel * dt
        check = self.q0.copy()
        mujoco.mj_integratePos(m, check, delta, 1.0)
        error = np.zeros(m.nv)
        mujoco.mj_differentiatePos(m, error, 1.0, check, d.qpos)
        error_norm = float(np.linalg.norm(error))
        if not np.isfinite(error_norm) or error_norm > 1e-7 or not self.bounds.valid(self.q0, d.qpos):
            self.mark_unresolved(self.owners, "integration_path_not_bounded")
            return
        travel = self.bounds.travel(delta)
        self.charge(native, travel)
        if not len(travel) or travel.max() <= ACCURACY:
            self.fast_steps += 1
            return
        self.segments += 1
        pose = pose_path(m, self.scratch, self.q0, delta)
        pose(0.0)
        pairs, speeds, charged = [], [], []
        for a, bound in zip(self.watched, travel):
            if bound <= ACCURACY:
                continue
            R = self.scratch.geom_xmat[a].reshape(3, 3)
            centre = self.scratch.geom_xpos[a] + R @ m.geom_aabb[a, :3]
            half = np.abs(R) @ m.geom_aabb[a, 3:]
            near = self.planes | (np.abs(self.centres - centre) <= self.halves + half + bound + ACCURACY).all(axis=1)
            for b in self.static[near]:
                if self.gowners[a] != self.gowners[b] and self.policy.allows(int(a), int(b)):
                    pairs.append((int(a), int(b)))
                    speeds.append(bound)
                    charged.append(int(self.gowners[a]))
        if not pairs:
            return
        r = depth_bounds(m, self.scratch, pairs, speeds, pose, accuracy=ACCURACY, budget=self.budget)
        self.evaluations += r["evaluations"]
        for k, o in enumerate(charged):
            self.pen[o] = max(self.pen[o], float(r["observed"][k]))
            self.upper[o] = max(self.upper[o], float(r["upper"][k]))
            if k in r["unresolved"]:
                self.mark_unresolved([o], "distance_query_unresolved_or_budget_exhausted")
            if k in r["crossed"] or r["observed"][k] > 0.002:
                a, b = pairs[k]
                crossed = k in r["crossed"]
                if crossed:
                    self.crossings.setdefault(o, f"geoms {[a, b]} cross between {[self.t0, float(d.time)]}")
                existing = next((e for e in self.events if e["geoms"] == [a, b] and e["crossed"] == crossed), None)
                if existing is not None and existing["penetration"] >= r["observed"][k]:
                    continue
                event = dict(
                    owner=o,
                    geoms=[a, b],
                    names=[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in (a, b)],
                    interval=[self.t0, float(d.time)],
                    fraction=r["witness"].get(k),
                    penetration=float(r["observed"][k]),
                    crossed=crossed,
                    qpos_start=self.q0.tolist(),
                    qpos_end=d.qpos.tolist(),
                )
                if existing is not None:
                    self.events[self.events.index(existing)] = event
                elif len(self.events) < 24:
                    self.events.append(event)

    def charge(self, native, travel):  # each owner's bound: its overlap plus its fastest geom's travel, all at once
        top = np.zeros(self.s.m.nbody)
        np.maximum.at(top, self.gowners[self.watched], travel)
        for o in self.owners:
            self.upper[o] = max(self.upper[o], native.get(o, 0.0) + min(float(top[o]), ACCURACY))

    def crossed(self, owners):
        return [event for o, event in self.crossings.items() if o in owners]
