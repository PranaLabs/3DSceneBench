"""Joint-space motion bounds and FK paths, independent of the live simulator."""

import mujoco
import numpy as np
from scipy.sparse import csr_matrix


class MotionBounds:
    """Surface travel upper bounds for a constant generalized displacement.

    Each rotation uses a conservative chain-length radius, including all child
    joint offsets and slide excursions. This covers offset hinges and compound
    joints; interpolating geom centres instead would replace hinge arcs by chords.
    """

    def __init__(self, m, q0, geoms):
        self.m, self.geoms = m, np.asarray(geoms, int)
        self.slide_limits = {}
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
                self.slide_limits[j] = (
                    max(abs(q0[m.jnt_qposadr[j]]), np.max(np.abs(m.jnt_range[j])) if m.jnt_limited[j] else 0) + 1.0
                )
        lr, lc, lv, ar, ac, av = [], [], [], [], [], []
        for i, g in enumerate(self.geoms):
            radius = float(np.linalg.norm(m.geom_pos[g]) + m.geom_rbound[g])
            b = int(m.geom_bodyid[g])
            while b:
                js = range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b])
                extra = sum(2 * np.linalg.norm(m.jnt_pos[j]) + self.slide_limits.get(j, 0) for j in js)
                for j in js:
                    typ = int(m.jnt_type[j])
                    if typ in (0, 2):
                        lr.append(i)
                        lc.append(j)
                        lv.append(1.0)
                    if typ in (0, 1, 3):
                        ar.append(i)
                        ac.append(j)
                        av.append(radius + extra)
                radius += extra + np.linalg.norm(m.body_pos[b])
                b = int(m.body_parentid[b])
        shape = (len(self.geoms), m.njnt)
        self.linear = csr_matrix((lv, (lr, lc)), shape=shape)
        self.angular = csr_matrix((av, (ar, ac)), shape=shape)
        active = np.unique(lc + ac).astype(int)
        self.free = active[m.jnt_type[active] == 0]
        self.ball = active[m.jnt_type[active] == 1]
        self.slide = active[m.jnt_type[active] == 2]
        self.hinge = active[m.jnt_type[active] == 3]
        self.slide_limits = {j: v for j, v in self.slide_limits.items() if j in active}

    def travel(self, delta):
        m = self.m
        lin = np.zeros(m.njnt)
        ang = np.zeros(m.njnt)
        a = m.jnt_dofadr[self.free, None]
        lin[self.free] = np.linalg.norm(delta[a + np.arange(3)], axis=1)
        ang[self.free] = np.linalg.norm(delta[a + np.arange(3, 6)], axis=1)
        ang[self.ball] = np.linalg.norm(delta[m.jnt_dofadr[self.ball, None] + np.arange(3)], axis=1)
        lin[self.slide] = np.abs(delta[m.jnt_dofadr[self.slide]])
        ang[self.hinge] = np.abs(delta[m.jnt_dofadr[self.hinge]])
        return np.asarray(self.linear @ lin + self.angular @ ang)

    def valid(self, q0, q1):
        return all(
            max(abs(q0[self.m.jnt_qposadr[j]]), abs(q1[self.m.jnt_qposadr[j]])) <= limit
            for j, limit in self.slide_limits.items()
        )


def pose_path(m, scratch, q0, delta):
    def pose(t):
        scratch.qpos[:] = q0
        mujoco.mj_integratePos(m, scratch.qpos, delta, t)
        mujoco.mj_kinematics(m, scratch)

    return pose
