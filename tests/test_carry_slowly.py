"""The robot drives more slowly while its hand holds something: 0.25 m/s instead of 0.35, so a cup carried 7 m through a
doorway no longer slips in the grip."""

import math, unittest
from types import SimpleNamespace
from sbv.functionality.f4_robot.pick_place import Executor, SPEED, CARRY


class CarrySlowly(unittest.TestCase):
    def test_a_carried_leg_takes_longer(self):
        ex = Executor.__new__(Executor)
        ex.base = (0.0, 0.0, 0.0)
        took = []
        ex.go = lambda seconds, base=None, **k: (took.append(seconds), setattr(ex, "base", base))
        ex.drive([(0, 0, 0), (2.0, 0, 0)])
        ex.base = (0.0, 0.0, 0.0)
        ex.drive([(0, 0, 0), (2.0, 0, 0)], CARRY)
        self.assertEqual((CARRY, SPEED), (0.25, 0.35))
        self.assertAlmostEqual(took[0], 2.0 / SPEED)
        self.assertAlmostEqual(took[1], 2.0 / CARRY)


if __name__ == "__main__":
    unittest.main()
