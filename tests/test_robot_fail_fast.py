"""A robot candidate whose nominal run fails skips its jittered runs; a task that passes still runs every check."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from sbv import calibrate

EDITS = [(0, {"mass": 1.3, "friction": 0.15}), (0, {"mass": 0.7, "friction": -0.15})]


class Monitor:
    owners = ["cup"]
    pen = {"cup": 0.001}

    def __init__(self, ok):
        self.details = dict(success=ok, **({} if ok else dict(reason="physical task failure")))

    def crossed(self, owners):
        return []

    def report(self, owners):
        return {}


class RobotFailFast(unittest.TestCase):
    def job(self, fail_seed=None, fail_edit=False, error=False):
        calls = []

        def trial(scene, unit, corr, seed, edit=None):
            calls.append((seed, copy.deepcopy(edit)))
            if error:
                raise RuntimeError("native execution failed")
            ok = seed != fail_seed and not (fail_edit and edit is not None)
            return [ok], [Monitor(ok)]

        unit = {"id": "robot|cup|table", "kind": "robot", "slots": ["cup", "table"], "params": {}}
        with (
            patch.object(calibrate, "S", SimpleNamespace(reset=lambda: None)),
            patch.object(calibrate, "OPT", {"jitter": 3}),
            patch("sbv.functionality.units.robot_trial", return_value=trial),
            patch("sbv.functionality.f4_robot.plan.nearby", return_value=[]),
        ):
            return calibrate._robot_job(unit), calls

    def test_nominal_failure_keeps_the_reason_and_runs_nothing_else(self):
        (uid, u, why), calls = self.job(fail_seed=0)
        self.assertEqual((uid, u, why), ("robot|cup|table", None, "reference task fails: physical task failure"))
        self.assertEqual(calls, [(0, None)])

    def test_a_task_that_passes_runs_every_jitter_and_both_physics_edits(self):
        (_, u, why), calls = self.job()
        self.assertIsNone(why)
        self.assertFalse(u["hint_only"])
        self.assertEqual(calls, [(s, None) for s in range(4)] + EDITS)

    def test_a_jitter_failure_still_runs_every_seed_and_refuses_the_task(self):
        (_, u, why), calls = self.job(fail_seed=2)
        self.assertIsNone(u)
        self.assertEqual(why, "task outcome changes under 1 mm jitter")
        self.assertEqual(calls, [(s, None) for s in range(4)])

    def test_a_physics_edit_failure_marks_the_task_hint_only(self):
        (_, u, _), calls = self.job(fail_edit=True)
        self.assertTrue(u["hint_only"])
        self.assertEqual(calls, [(s, None) for s in range(4)] + EDITS[:1])

    def test_an_error_in_the_nominal_run_is_not_hidden(self):
        with self.assertRaisesRegex(RuntimeError, "native execution failed"):
            self.job(error=True)


if __name__ == "__main__":
    unittest.main()
