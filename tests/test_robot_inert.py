"""The object gate reaches the robot world: an object the F1 gate made inert (it blew the copy up) is inert in the robot
tasks' own model too, where it used to take part and could blow that one task up on its own."""

import tempfile, unittest
import numpy as np
from sbv.common.scene import Scene, PHYSICS, INERT_DAMPING
from sbv.functionality.f4_robot import plan as RP
from tests import rooms


@unittest.skipUnless(RP.installed(), "needs the RB-Y1 model (scripts/get_robot.sh)")
class RobotWorldInert(unittest.TestCase):
    def test_an_inert_object_touches_nothing_and_holds_still_in_the_robot_world(self):
        xml = rooms.write(tempfile.mkdtemp(), rooms.robot_room_xml())
        s = Scene(xml, dict(PHYSICS, inert=["cup__0"]))
        w = RP.World(xml, (0.5, 0.5, 0.0), textures=True, phys=s.phys)
        m = w.m
        cup = w.objects["cup__0"]
        gs = w.geoms("cup__0")
        self.assertTrue(gs)
        self.assertFalse(m.geom_contype[gs].any() or m.geom_conaffinity[gs].any())
        self.assertTrue((m.dof_damping[w.owner[m.dof_bodyid] == cup] == INERT_DAMPING).all())
        self.assertTrue(m.geom_contype[w.geoms("table__0")].any())  # the others as before
        plain = RP.World(xml, (0.5, 0.5, 0.0))
        self.assertTrue(plain.m.geom_contype[plain.geoms("cup__0")].any())
        z0 = w.d.xipos[cup, 2]
        w.settle()
        self.assertLess(abs(w.d.xipos[cup, 2] - z0), 1e-3)  # held where the copy put it


if __name__ == "__main__":
    unittest.main()
