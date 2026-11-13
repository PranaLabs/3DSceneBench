"""A failed or non-finite calibration run can never freeze an infinite tolerance."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from sbv.nonrigid import verify
from tests.test_nonrigid import COFFEE, side


class CalibrationErrors(unittest.TestCase):
    def test_a_failed_or_nonfinite_true_run_refuses_the_calibration(self):
        root = side([COFFEE])
        for failure in ({"error": "timeout"}, {"readout": float("nan"), "finite": True}):
            out = Path(tempfile.mkdtemp()) / "case.json"

            def run(xml, obj, action, scale, work, video=None, threads=None):
                if Path(work).name == "base":
                    return failure
                return dict(readout=0.5, held_at_rest=1.0, finite=True, seconds=0.1)

            with patch.object(verify, "_one", side_effect=run):
                with self.assertRaisesRegex(SystemExit, "base run"):
                    verify.calibrate(root, str(out), procs=1)
            self.assertFalse(out.exists())

    def test_invalid_sensitivity_is_not_usable_response_evidence(self):
        root = side([COFFEE])
        out = Path(tempfile.mkdtemp()) / "case.json"

        def run(xml, obj, action, scale, work, video=None, threads=None):
            if scale:
                return dict(readout=100.0, finite=False, seconds=0.1)
            return dict(readout=0.5, held_at_rest=1.0, finite=True, seconds=0.1)

        with patch.object(verify, "_one", side_effect=run):
            units = verify.calibrate(root, str(out), procs=1)
        self.assertFalse(units[0]["decidable"])
        self.assertEqual(units[0]["sensitivity"], {"half": None, "double": None})
        self.assertEqual(set(units[0]["sensitivity_errors"]), {"half", "double"})

    def test_bad_legacy_tolerance_is_rejected_before_scoring(self):
        path = Path(tempfile.mkdtemp()) / "case.json"
        path.write_text(json.dumps({"units": [{"id": "S1.bad", "tolerance": float("inf"), "truth": 0.5}]}))
        with self.assertRaisesRegex(ValueError, "recalibration"):
            verify.evaluate(str(path), "/missing", "/unused")


if __name__ == "__main__":
    unittest.main()
