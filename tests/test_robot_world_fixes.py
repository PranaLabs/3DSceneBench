import tempfile, unittest
import numpy as np
from sbv.common.scene import Scene, ARENA, PHYSICS
from sbv.functionality import f2_calibrate as C
from sbv.functionality.f4_robot import plan as RP, tasks as T, rby1
from sbv.calibrate import room_bounds
from tests import rooms


class RobotWorldFixes(unittest.TestCase):
    def test_a_copys_size_is_reset_to_the_fixed_arena(self):
        big = rooms.robot_room_xml().replace("<worldbody>", '<size njmax="40000" nconmax="16000"/>\n  <worldbody>')
        self.assertEqual(Scene(rooms.write(tempfile.mkdtemp(), big)).m.narena, ARENA)  # 92 GB asked, 1 GiB given
        self.assertEqual(rby1.ITERATIONS, 100)

    def test_a_task_watches_its_own_objects_and_surroundings(self):
        w = type("World", (), {})()
        w.objects = dict(cup__0=1, table__0=2, cabinet__0=3, bowl__0=4)
        unit = dict(slots=["cup__0", "cabinet__0"], params=dict(near=["table__0"]))
        self.assertEqual(
            RP.watched(w, unit, None, 1), dict(cabinet__0=3, table__0=2)
        )  # the bowl elsewhere is F2's business
        corr = dict(
            cup__0="cup__0", cabinet__0="cabinet__0", table__0="desk__0"
        )  # in a copy: its names, present ones only
        self.assertEqual(RP.watched(w, unit, corr, 1), dict(cabinet__0=3))
        self.assertEqual(
            RP.watched(w, dict(slots=["cup__0"], params={}), None, 1), dict(table__0=2, cabinet__0=3, bowl__0=4)
        )  # before near: all

    def test_a_named_task_may_start_from_its_own_floor_point(self):
        s = Scene(rooms.write(tempfile.mkdtemp(), rooms.robot_room_xml()), PHYSICS, probes=True, textures=False)
        bounds = room_bounds(s)
        ctx = C.context(s, [2.5, 2.0])
        spec = {
            "push": [
                {"object": "cup__0"},
                {"object": "cup__0", "start": [2.7, 2.7]},
                {"object": "cup__0", "start": [-3.0, -3.0]},
            ]
        }
        units, bad = T.named(s, ctx, spec, [2.5, 1.0, 0.0], bounds)
        self.assertEqual(len(units), 2)
        room, own = [u["params"]["start"] for u in units]
        self.assertLess(np.linalg.norm(np.subtract(room[:2], [2.5, 1.0])), 0.5)  # the room's start, as before
        self.assertLess(
            np.linalg.norm(np.subtract(own[:2], [2.7, 2.7])), 0.5
        )  # the task's own start, 1.3 m from the room's
        self.assertEqual(
            bad, [("robot_push|cup__0", "named task: no free floor for the robot within 0.5 m of the task's start")]
        )


if __name__ == "__main__":
    unittest.main()
