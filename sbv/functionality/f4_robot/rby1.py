"""Put the Rainbow Robotics RB-Y1 (MuJoCo Menagerie, Apache-2.0) into one of our MuJoCo rooms, and steer its arm.

The wheels are replaced by a base that slides and turns on the floor (first version); torso and arms keep the model's
own joints and position motors, with gravity compensation, and the arm is steered by inverse kinematics solved here
(damped least squares on the 6 torso + 7 arm joints, fingertips pointing straight down). Physics options follow the
robot model's own scene (implicit integrator, elliptic friction cones, impratio 100). Robot files:
data/menagerie/rainbow_robotics_rby1 at the repository root (scripts/get_robot.sh; not in git), or SBV_ROBOT_DIR.
"""

import os
import numpy as np, mujoco

from sbv import ROOT

ROBOT_DIR = os.environ.get("SBV_ROBOT_DIR") or os.path.join(ROOT, "data", "menagerie", "rainbow_robotics_rby1")
P = "rb_"  # prefix for everything that belongs to the robot
ARM = "right"  # the arm that does the work
DT = 0.002
ITERATIONS = 100  # Newton iteration cap, MuJoCo's default. The solver stops at its tolerance first
# wherever it converges, so this changes nothing there (the existing rooms' robot trials are the same bit for bit).


def build(room_xml, start, goal=None, textures=True):
    """Room plus robot in one model. start = (x, y, yaw) of the robot base on the floor; goal = (center, half size)
    of a see-through green box drawn where the object should end up (no collisions). textures=False: the room's
    pictures become 1-pixel textures (sbv.common.scene.flat_textures)."""
    room = mujoco.MjSpec.from_file(room_xml)
    if not textures:
        from sbv.common.scene import flat_textures

        flat_textures(room)
    room.meshdir = os.path.join(os.path.dirname(os.path.abspath(room_xml)), room.meshdir)
    rob = mujoco.MjSpec.from_file(os.path.join(ROBOT_DIR, "rby1a_1.2.xml"))
    rob.meshdir = os.path.join(ROBOT_DIR, rob.meshdir)
    base = rob.body("base")
    for j in list(base.joints):  # the free joint goes; the base slides and turns instead
        rob.delete(j)
    for name, kind, axis in [
        ("base_x", mujoco.mjtJoint.mjJNT_SLIDE, [1, 0, 0]),
        ("base_y", mujoco.mjtJoint.mjJNT_SLIDE, [0, 1, 0]),
        ("base_yaw", mujoco.mjtJoint.mjJNT_HINGE, [0, 0, 1]),
    ]:
        base.add_joint(name=name, type=kind, axis=axis, damping=0, armature=0, range=[-100, 100])
    for a in list(rob.actuators):  # wheel motors go with the wheels' job
        if "wheel" in a.name:
            rob.delete(a)
    for b in rob.bodies:  # arms and torso float against gravity, as a real controller would hold them
        b.gravcomp = 1
    for name, gain, damp, lim in [("base_x", 3e4, 6e3, 100), ("base_y", 3e4, 6e3, 100), ("base_yaw", 3e4, 6e3, 100)]:
        a = rob.add_actuator(name=name + "_act", target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=gain, kv=damp)
        a.ctrlrange = [-lim, lim]
        a.forcerange = [-5e3, 5e3]
    # the point between the two fingertip balls; the fingers point along the wrist's -z and close along its x
    rob.body(f"link_{ARM}_arm_6").add_site(name="tcp", pos=[0, 0, -0.226], size=[0.008, 0, 0], rgba=[1, 0, 0, 0])
    rob.body("link_head_2").add_camera(name="head_cam", pos=[0.06, 0, 0.03], xyaxes=[0, -1, 0, 0, 0, 1], fovy=70)
    frame = room.worldbody.add_frame(pos=[0, 0, 0])
    frame.attach_body(base, P, "")
    for b in ("base", "wheel_r", "wheel_l"):  # the base glides; it does not rub the floor or the walls
        e = room.add_exclude()
        e.bodyname1 = "world"
        e.bodyname2 = P + b
    room.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    room.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    room.option.impratio = 100
    room.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    room.option.iterations = ITERATIONS
    room.visual.global_.offwidth, room.visual.global_.offheight = 1920, 1080
    room.visual.quality.shadowsize = 4096
    room.visual.headlight.ambient = [0.42, 0.42, 0.42]
    room.visual.headlight.diffuse = [0.35, 0.35, 0.35]
    room.visual.headlight.specular = [0.05, 0.05, 0.05]
    if goal is not None:
        room.worldbody.add_geom(
            name="goal",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=goal[0],
            size=goal[1],
            rgba=[0.1, 0.85, 0.3, 0.28],
            contype=0,
            conaffinity=0,
            group=1,
        )
    from sbv.common.scene import fixed_size

    m = fixed_size(room).compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    q = {"base_x": start[0], "base_y": start[1], "base_yaw": start[2]}
    for n, v in q.items():
        d.qpos[m.jnt_qposadr[m.joint(P + n).id]] = v
    d.ctrl[:] = 0
    for n, v in q.items():
        d.ctrl[m.actuator(P + n + "_act").id] = v
    mujoco.mj_forward(m, d)
    return m, d


class Robot:
    """Joint and motor bookkeeping, plus inverse kinematics for the working arm with the torso helping."""

    def __init__(self, m, d):
        self.m, self.d = m, d
        jn = [f"torso_{i}" for i in range(6)] + [f"{ARM}_arm_{i}" for i in range(7)]
        self.joints = [m.joint(P + n).id for n in jn]
        self.qadr = np.array([m.jnt_qposadr[j] for j in self.joints])
        self.dadr = np.array([m.jnt_dofadr[j] for j in self.joints])
        self.act = {}  # joint id -> actuator id
        for a in range(m.nu):
            if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
                self.act[m.actuator_trnid[a, 0]] = a
        lim = np.array([m.actuator_ctrlrange[self.act[j]] for j in self.joints])
        self.lo, self.hi = lim[:, 0] + 0.02, lim[:, 1] - 0.02
        self.base = [self.act[m.joint(P + n).id] for n in ("base_x", "base_y", "base_yaw")]
        self.finger = m.actuator(P + f"{ARM}_finger_act").id
        self.head = [self.act[m.joint(P + n).id] for n in ("head_0", "head_1")]
        self.tcp = m.site(P + "tcp").id
        self.dk = mujoco.MjData(m)  # scratch copy for kinematics only

    def ik(self, base, pos, rot, q0, iters=400, nominal=None):
        """Torso and arm angles that put the fingertips at pos with orientation rot, base parked at (x, y, yaw)."""
        m, dk = self.m, self.dk
        dk.qpos[:] = self.d.qpos
        for n, v in zip(("base_x", "base_y", "base_yaw"), base):
            dk.qpos[m.jnt_qposadr[m.joint(P + n).id]] = v
        q = np.clip(np.array(q0, float), self.lo, self.hi)
        nom = q.copy() if nominal is None else np.array(nominal, float)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            dk.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, dk)
            mujoco.mj_comPos(m, dk)
            R = dk.site_xmat[self.tcp].reshape(3, 3)
            e = np.concatenate(
                [pos - dk.site_xpos[self.tcp], 0.5 * sum(np.cross(R[:, i], rot[:, i]) for i in range(3))]
            )
            if np.linalg.norm(e[:3]) < 1e-4 and np.linalg.norm(e[3:]) < 1e-3:
                break
            mujoco.mj_jacSite(m, dk, jp, jr, self.tcp)
            J = np.vstack([jp, jr])[:, self.dadr]
            Jp = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(6))
            dq = Jp @ e + (np.eye(len(q)) - Jp @ J) @ (0.05 * (nom - q))
            q = np.clip(q + np.clip(dq, -0.1, 0.1), self.lo, self.hi)
        dk.qpos[self.qadr] = q
        mujoco.mj_kinematics(m, dk)
        return q, float(np.linalg.norm(pos - dk.site_xpos[self.tcp]))


def down_grasp(closing):
    """Fingers pointing down, closing along the horizontal direction `closing`."""
    x = np.array([closing[0], closing[1], 0.0])
    x /= np.linalg.norm(x)
    z = np.array([0, 0, 1.0])
    return np.column_stack([x, np.cross(z, x), z])
