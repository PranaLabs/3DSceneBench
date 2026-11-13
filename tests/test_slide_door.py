"""A door that slides along its own face: the open step finds its handle on whichever side stands proud, the robot walks
it open past what an arm reaches, and the floor map after the step frees the doorway it opened."""

import tempfile, types, unittest
import mujoco, numpy as np
from sbv.common.scene import Scene, PHYSICS
from sbv.calibrate import room_bounds
from sbv.functionality import f2_calibrate as C
from sbv.functionality.f4_robot import plan as RP, tasks as T, activity as A
from sbv.functionality.units import robot_trial
from tests import rooms

DOOR = (
    '<body name="door__0" pos="2.5 1.0 1.0"><joint name="door_slide" type="slide" axis="0 1 0" range="0 1.3" damping="20" frictionloss="2"/>'
    '<geom name="door_panel" type="box" size="0.01 0.6 1.0" density="600"/>'
    '<geom name="door_bar" type="capsule" fromto="-0.05 -0.45 -0.15 -0.05 -0.45 0.15" size="0.012" density="2000"/>'
    '<geom name="door_bar_mount" type="box" pos="-0.03 -0.45 0" size="0.02 0.01 0.01" density="2000"/></body>'
)
# the room's floor reaches it
BOX = '<body name="box__0" pos="0.6 3.4 0.1"><freejoint/><geom type="box" size="0.1 0.1 0.1" mass="1"/></body>'
ROOM = (
    '<mujoco><compiler angle="radian" autolimits="true"/><default><geom friction="0.8 0.005 0.0001" density="600"/></default><worldbody>'
    '<geom name="floor" type="plane" size="0 0 0.1"/>'
    + "".join(
        rooms.box(n, p, h)
        for n, p, h in [
            ("wall_w", (-0.05, 2.0, 1.25), (0.05, 2.1, 1.25)),
            ("wall_e", (5.05, 2.0, 1.25), (0.05, 2.1, 1.25)),
            ("wall_s", (2.5, -0.05, 1.25), (2.6, 0.05, 1.25)),
            ("wall_n", (2.5, 4.05, 1.25), (2.6, 0.05, 1.25)),
        ]
    )
    + DOOR
    + BOX
    + "</worldbody></mujoco>"
)


class SlideDoor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        xml = rooms.write(tempfile.mkdtemp(), ROOM)
        cls.s = s = Scene(xml, dict(PHYSICS), probes=True, textures=False)
        s.settle()
        cls.b = b = room_bounds(s)
        cls.c = c = [(b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2]
        cls.ctx = C.context(s, c)

    def test_the_door_is_walked_open_by_its_handle(self):
        units, bad = T.named(
            self.s,
            self.ctx,
            dict(activity=[dict(words="x", start=[1.5, 1.2], steps=[dict(kind="open", object="door__0", amount=1.0)])]),
            self.c,
            self.b,
        )
        self.assertEqual(bad, [])
        u = units[0]
        if not RP.installed():
            self.skipTest("needs the RB-Y1 model (scripts/get_robot.sh)")
        u["params"]["near"] = RP.nearby(self.s, u["slots"])
        self.s.reset()
        out, mons = robot_trial("robot_activity")(self.s, u, None, 0)
        det = mons[0].details
        self.assertEqual(out, [True], det)
        self.assertGreaterEqual(det["steps"][0]["moved"], 0.8)

    def test_the_floor_after_frees_the_doorway(self):
        xs, ys, occ = RP.floor_map(self.s, self.b)
        i, j = int(np.abs(xs - 2.5).argmin()), int(np.abs(ys - 0.7).argmin())
        self.assertTrue(occ[i, j])  # the closed panel stands here
        d = mujoco.MjData(self.s.m)
        d.qpos[:] = self.s.q0
        d.qpos[self.s.m.jnt_qposadr[self.s.m.joint("door_slide").id]] = 1.2
        q0 = self.s.q0.copy()
        xs2, ys2, free = A.floor_after(self.s, types.SimpleNamespace(m=self.s.m, d=d), self.b)
        self.assertTrue(free[:, i, j].any())  # opened 1.2 m: the base fits where the panel stood
        np.testing.assert_array_equal(self.s.q0, q0)
        self.assertTrue(RP.floor_map(self.s, self.b)[2][i, j])  # the room's own start is back


if __name__ == "__main__":
    unittest.main()
