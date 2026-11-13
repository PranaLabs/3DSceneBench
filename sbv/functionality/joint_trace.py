"""Bounded diagnostic trace of a force-driven joint; no modification to physics."""

import math
import mujoco
import numpy as np


class DriveTrace:
    def __init__(self, scene, joint, sign, force, seconds):
        self.s, self.j, self.sign = scene, joint, sign
        m, d = scene.m, scene.d
        self.a, self.v = m.jnt_qposadr[joint], m.jnt_dofadr[joint]
        self.q0 = float(d.qpos[self.a])
        self.sub = scene.subtree(m.jnt_bodyid[joint])
        self.ingeom = np.isin(m.geom_bodyid, list(self.sub))  # which geoms belong to the moving part
        self.stride = max(1, math.ceil(seconds / m.opt.timestep / 150))
        self.samples = []
        self.peak_constraint = 0.0
        self.record = dict(
            joint=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint),
            command_force=sign * force,
            force_unit="N m" if m.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE else "N",
            duration=seconds,
            timestep=float(m.opt.timestep),
            q_start=self.q0,
            damping=float(m.dof_damping[self.v]),
            frictionloss=float(m.dof_frictionloss[self.v]),
            stiffness=float(m.jnt_stiffness[joint]),
        )

    def tick(self, k, final=False):
        m, d = self.s.m, self.s.d
        self.peak_constraint = max(self.peak_constraint, abs(float(d.qfrc_constraint[self.v])))
        if k % self.stride and not final:
            return
        contacts = []
        con = d.contact[: d.ncon]
        g = self.ingeom
        for i in np.flatnonzero(
            g[con.geom1] != g[con.geom2]
        ):  # contacts across the moving part's boundary, in MuJoCo's order
            i = int(i)
            c = con[i]
            force = np.zeros(6)
            if c.efc_address >= 0:
                mujoco.mj_contactForce(m, d, i, force)
            contacts.append(
                dict(geoms=[int(c.geom1), int(c.geom2)], depth=max(0.0, -float(c.dist)), normal_force=float(force[0]))
            )
        contacts = sorted(contacts, key=lambda c: abs(c["normal_force"]), reverse=True)[:4]
        self.samples.append(
            dict(
                time=float(d.time),
                q=float(d.qpos[self.a]),
                velocity=float(d.qvel[self.v]),
                force_time=float(d.time - m.opt.timestep),
                constraint_force=float(d.qfrc_constraint[self.v]),
                passive_force=float(d.qfrc_passive[self.v]),
                contacts=contacts,
            )
        )

    def finish(self):
        if not self.samples or self.samples[-1]["time"] != float(self.s.d.time):
            self.tick(1, final=True)
        q = float(self.s.d.qpos[self.a])
        return dict(
            self.record,
            q_end=q,
            opening_travel=self.sign * (q - self.q0),
            peak_constraint_force=self.peak_constraint,
            samples=self.samples,
        )
