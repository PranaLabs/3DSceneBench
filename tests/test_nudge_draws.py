"""The nudge test of robot-task calibration (calibrate.needs_and_nudges): five draws each at 5 and 10 cm; a task is
refused only when more than one of the five fails at 5 cm, so one unlucky draw no longer refuses it; the 10 cm share is
kept as its margin."""

import os, tempfile, unittest
from types import SimpleNamespace
from unittest import mock
from sbv import calibrate as CB
from sbv.common.scene import Scene, PHYSICS
from sbv.functionality.f4_robot import pick_place, plan as RP
from tests import rooms


def nudge(results_5cm, results_10cm=(True,) * 5):
    """needs_and_nudges on a task that needs one free chair, with the nudged runs' outcomes scripted."""
    unit = dict(slots=["cup__0", "table__0"], params={})
    det = dict(passed_by=["chair__0", "cup__0"])
    scene = SimpleNamespace(free_dof=lambda n: 0)
    runs = {0.05: list(results_5cm), 0.1: list(results_10cm)}
    with mock.patch.object(CB, "_nudged", lambda sc, u, tr, names, sigma, seed: runs[sigma].pop(0)):
        why = CB.needs_and_nudges(scene, unit, det, trial=None)
    return why, unit["params"]


class NudgeDraws(unittest.TestCase):
    def test_five_draws_and_one_may_fail(self):
        self.assertEqual([len(v) for v in CB.NUDGES.values()], [5, 5])
        why, p = nudge([True, False, True, True, True])  # one unlucky draw: kept
        self.assertIsNone(why)
        self.assertEqual(p["needed"], ["chair__0"])
        self.assertEqual(p["nudge"]["0.05"], 0.8)
        why, _ = nudge([True, False, True, False, True])  # two of five: refused
        self.assertIn("fails when the free objects", why)
        self.assertIn("60% of runs pass", why)

    def test_the_10_cm_share_is_a_margin_only(self):
        why, p = nudge([True] * 5, [False, False, True, False, False])
        self.assertIsNone(why)
        self.assertAlmostEqual(p["nudge"]["0.1"], 0.2)


@unittest.skipUnless(RP.installed(), "needs the RB-Y1 model (scripts/get_robot.sh)")
class NudgeInPlace(unittest.TestCase):
    """A nudge moves the objects at the reference room's start and changes nothing else (calibrate._nudged). it ran on a
    copy of the file written beside it through MjSpec.to_xml: 6-digit numbers, and None for every draw in a room folder
    it could not write to."""

    @classmethod
    def setUpClass(cls):
        xml = rooms.robot_room_xml().replace(
            'friction="0.8 ', 'friction="0.812345678 '
        )  # more digits than to_xml writes
        cls.dir = tempfile.mkdtemp()
        cls.xml = rooms.write(cls.dir, xml, functions=False)  # its folder holds scene.xml only
        cls.scene = s = Scene(cls.xml, PHYSICS, probes=True, textures=False)
        s.settle()
        cls.unit = dict(
            id="robot|cup__0|cabinet__0",
            kind="robot",
            slots=["cup__0", "cabinet__0"],
            params=dict(
                rel=[0.0, 0.0], start=[2.5, 1.0, 0.0], bounds=CB.room_bounds(s), isolate=True, near=["table__0"]
            ),
        )

    def nudged(self, sigma):
        seen = []

        def trial(*a):
            r = pick_place.trial(*a)
            seen.append((a[0].xml, r[1][0].details))
            return r

        return CB._nudged(self.scene, self.unit, trial, ["table__0"], sigma, 1), seen

    def test_a_zero_nudge_is_the_nominal_run_of_the_true_file(self):
        nominal = pick_place.trial(self.scene, self.unit, None, 0)[1][0].details
        ok, seen = self.nudged(0.0)
        self.assertTrue(ok)
        self.assertEqual(seen, [(self.xml, nominal)])

    def test_a_room_folder_it_cannot_write_to_is_still_nudged(self):
        os.chmod(self.dir, 0o555)
        self.addCleanup(os.chmod, self.dir, 0o755)
        ok, seen = self.nudged(0.05)
        self.assertIsNotNone(ok)
        self.assertEqual(len(seen), 1)
        self.assertEqual(sorted(os.listdir(self.dir)), ["scene.xml"])


if __name__ == "__main__":
    unittest.main()
