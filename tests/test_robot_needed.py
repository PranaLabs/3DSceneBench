"""What a robot task needs: the objects the robot touched, or drove past within 1 m of the task's own objects, in the
reference room; only these cancel the task in a copy that lacks them. A case from before keeps its 30 cm "near" list."""

import unittest
from types import SimpleNamespace
import numpy as np
from sbv.functionality.f4_robot.plan import passed_by, missing_near

BOXES = {
    "chair__0": ((1.0, 0.0, 0.0), (1.5, 0.5, 0.9)),  # beside the base's path, on the floor
    "wall_art__0": ((0.9, -0.3, 1.0), (1.6, -0.25, 1.6)),  # as close, but on the wall above the base
    "plant__0": ((4.0, 4.0, 0.0), (4.3, 4.3, 1.0)),  # far away
    "cup__0": ((9.0, 9.0, 0.7), (9.1, 9.1, 0.8)),
}  # far, but the hand touched it
ROOTS = {n: i + 1 for i, n in enumerate(BOXES)}


class Needed(unittest.TestCase):
    def test_touched_or_driven_past(self):
        w = SimpleNamespace(objects=ROOTS, aabb=lambda n: tuple(np.array(v, float) for v in BOXES[n]))
        ex = SimpleNamespace(touched={ROOTS["cup__0"]}, track=[(0.0, 0.0), (0.5, 0.0)], base=(0.5, 0.0, 0.0))
        self.assertEqual(passed_by(w, ex), ["chair__0", "cup__0"])

    def test_driven_past_counts_only_near_the_task(self):
        w = SimpleNamespace(objects=ROOTS, aabb=lambda n: tuple(np.array(v, float) for v in BOXES[n]))
        ex = SimpleNamespace(touched={ROOTS["cup__0"]}, track=[(0.0, 0.0), (0.5, 0.0)], base=(0.5, 0.0, 0.0))
        self.assertEqual(
            passed_by(w, ex, ["plant__0"]), ["cup__0"]
        )  # the chair is on the way, 3 m from the task: not needed
        self.assertEqual(
            passed_by(w, ex, ["wall_art__0"]), ["chair__0", "cup__0"]
        )  # the chair beside the task's object: needed
        start = [(np.array([1.6, 0.0, 0.7]), np.array([1.7, 0.1, 0.8]))]  # the task's object started by the chair
        self.assertEqual(
            passed_by(w, ex, ["plant__0"], start), ["chair__0", "cup__0"]
        )  # and was carried 3 m away: still needed

    def test_only_what_is_needed_cancels(self):
        unit = dict(params=dict(near=["chair__0", "wall_art__0"], needed=["chair__0"]))
        self.assertIsNone(missing_near(unit, {"chair__0": "chair__3"}))  # the picture is missing: no matter
        self.assertIn("chair__0", missing_near(unit, {"wall_art__0": "wall_art__0"}))  # the chair is: cancelled
        self.assertIn(
            "wall_art__0", missing_near(dict(params=dict(near=["wall_art__0"])), {})
        )  # a case from before: as it was


if __name__ == "__main__":
    unittest.main()
