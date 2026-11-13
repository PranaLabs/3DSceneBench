"""Tolerance policies (sbv/common/tolerances.py): calibration accepts a task judge's empty policy and changes nothing;
a policy that names units is refused before any work starts, never silently ignored. Run from the repository root:
    python -m unittest tests.test_tolerances -v
"""

import json, os, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from sbv.common import tolerances as Tol  # noqa: E402


def write(value):
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(value, f)
    return path


class Policy(unittest.TestCase):
    def test_load_reads_objects(self):
        self.assertEqual(Tol.load(write({})), {})
        self.assertEqual(Tol.load(write({"units": {"F2.1": {"tol": 0.01}}})), {"units": {"F2.1": {"tol": 0.01}}})

    def test_load_refuses_other_shapes(self):
        for bad in ([], "units", {"units": ["F2.1"]}):
            with self.assertRaises(ValueError):
                Tol.load(write(bad))

    def test_empty(self):
        for policy in (None, {}, {"units": {}}):
            self.assertTrue(Tol.empty(policy))
        for policy in ({"units": {"F2.1": {}}}, {"note": "wider pour bar"}):
            self.assertFalse(Tol.empty(policy))

    def test_only_an_empty_policy_is_supported(self):
        Tol.require_supported({})
        with self.assertRaisesRegex(ValueError, "F2.1"):
            Tol.require_supported({"units": {"F2.1": {"tol": 0.01}}})


class Calibration(unittest.TestCase):
    def test_build_refuses_a_policy_before_reading_the_room(self):
        from sbv import calibrate

        with self.assertRaisesRegex(ValueError, "does not apply tolerance policies"):
            calibrate.build("/no/such/room", tolerances={"units": {"F2.1": {"tol": 0.01}}})

    def test_build_accepts_an_empty_policy(self):
        from sbv import calibrate

        with self.assertRaises(
            (Exception, SystemExit)
        ) as caught:  # gets past the policy, then fails on the missing room
            # (calibrate.resources stops it with SystemExit)
            calibrate.build("/no/such/room", tolerances={})
        self.assertNotIn("tolerance", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
