"""Pick and place: the goal is anywhere the robot reaches on the goal's top. Spots are the named one first, then a 10 cm
grid nearest it; a spot with something standing on it, or past the top's edge, is not one."""

import os, tempfile, unittest
import numpy as np
from sbv.common.scene import Scene
from sbv.functionality.f4_robot.pick_place import spots, top_z, SPOTS


def room(book=True):
    """A 1.1 m square table 74 cm high, a book lying at its centre, a cup on the floor beside it."""
    x = f"""<mujoco><worldbody><geom name="floor" type="plane" size="5 5 0.1"/>
<body name="table__0"><geom type="box" pos="0 0 0.72" size="0.55 0.55 0.02"/><geom type="box" pos="0 0 0.35" size="0.04 0.04 0.35"/></body>
{'<body name="book__0" pos="0 0 0.755"><freejoint/><geom type="box" size="0.12 0.08 0.015" mass="0.5"/></body>' if book else ""}
<body name="cup__0" pos="1 1 0.05"><freejoint/><geom type="cylinder" size="0.035 0.05" mass="0.2"/></body>
</worldbody></mujoco>"""
    d = tempfile.mkdtemp(prefix="sbv_spots_")
    p = os.path.join(d, "scene.xml")
    with open(p, "w") as f:
        f.write(x)
    s = Scene(p, probes=True)
    s.settle()
    return s


class GoalSpots(unittest.TestCase):
    def test_the_named_spot_comes_first_when_clear(self):
        s = room(book=False)
        sp = spots(s, "table__0", [-0.17, 0.0], 0.035)
        self.assertTrue(np.allclose(sp[0][:2], [-0.17, 0.0]) and abs(sp[0][2] - 0.74) < 1e-3)
        self.assertEqual(len(sp), SPOTS)
        self.assertTrue(
            all(abs(x) <= 0.55 - 0.065 + 1e-9 and abs(y) <= 0.55 - 0.065 + 1e-9 for x, y, _ in sp)
        )  # inside the edge

    def test_the_spots_cover_the_whole_top(self):
        """Spread, not clustered at the named spot: on a clear 1.1 m table the spots reach both ends."""
        sp = spots(room(book=False), "table__0", [-0.17, 0.0], 0.035)
        xs, ys = [p[0] for p in sp], [p[1] for p in sp]
        self.assertGreater(max(xs) - min(xs), 0.7)
        self.assertGreater(max(ys) - min(ys), 0.7)
        self.assertTrue(np.allclose(sp[0][:2], [-0.17, 0.0]))  # the named spot still first

    def test_a_spot_with_something_on_it_is_skipped(self):
        s = room()
        sp = spots(s, "table__0", [0.0, 0.0], 0.035)
        self.assertFalse(np.allclose(sp[0][:2], [0.0, 0.0]))  # the book lies there
        self.assertTrue(all(abs(x) > 0.12 + 0.065 - 1e-9 or abs(y) > 0.08 + 0.065 - 1e-9 for x, y, _ in sp))

    def test_resting_on_the_top_anywhere(self):
        s = room()
        self.assertAlmostEqual(
            top_z(s, "table__0", (0.4, -0.4), 1.04), 0.74, places=3
        )  # far from the named spot: on the top
        self.assertIsNone(top_z(s, "table__0", (0.0, 0.0), 1.04))  # on the book, not on the table
        self.assertIsNone(top_z(s, "table__0", (0.7, 0.0), 1.04))  # past the edge


if __name__ == "__main__":
    unittest.main()
