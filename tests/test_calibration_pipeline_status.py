"""Pipeline status distinguishes failed physics from interrupted execution."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from sbv.common.calibration_jobs import Journal, CalibrationIncomplete, read, identity
from tests.test_calibration_bounded_jobs import work


class PipelineStatus(unittest.TestCase):
    def test_timeout_error_and_physical_rejection_are_distinct(self):
        from sbv import pipeline
        from tests import rooms

        with tempfile.TemporaryDirectory() as tmp:
            room = str(Path(rooms.write(str(Path(tmp) / "room"))).parent)
            for error, status, retry in [
                (CalibrationIncomplete("timeout", "box", "slow"), "timeout", True),
                (RuntimeError("engine problem"), "error", True),
                (SystemExit("no working task"), "failed", False),
            ]:
                with patch.object(pipeline.calibrate, "build", side_effect=error):
                    result = pipeline.check(room, str(Path(tmp) / "out"))
                self.assertFalse(result["ready"])
                self.assertEqual(result["status"], status)
                self.assertEqual(result["retry"], retry)


class RunCache(unittest.TestCase):
    def test_external_asset_change_invalidates_completed_room_check(self):
        from sbv import pipeline
        from sbv.common import case as K

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            room = work / "rooms" / "room"
            room.mkdir(parents=True)
            asset = work / "asset.obj"
            asset.write_text("old")
            (room / "scene.xml").write_text('<mujoco><asset><mesh file="../../asset.obj"/></asset></mujoco>')
            out = work / "results" / "room"
            out.mkdir(parents=True)
            old = dict(
                ready=False,
                status="failed",
                retry=False,
                fingerprint=K.fingerprint(str(room)),
                version=pipeline.version(),
                inputs=identity(room, {}),
                reason="fixture",
                artifacts={},
            )
            (out / "check.json").write_text(json.dumps(old))
            with patch.object(pipeline, "check", return_value=old) as check:
                pipeline.run(str(work), log=lambda *a: None)
                check.assert_not_called()
                asset.write_text("new")
                pipeline.run(str(work), log=lambda *a: None)
                self.assertEqual(check.call_count, 1)

    def test_invalidation_never_reuses_completed_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            with Journal(tmp, {"fixture": 1}, lambda *a: None) as j:
                j.jobs("objects", work, [("a", dict(value=1))])
                p = next(j.root.glob("objects/*.json"))
                j.invalidate()
                self.assertEqual(read(p)["status"], "invalidated")
                self.assertEqual(j.jobs("objects", work, [("a", dict(value=1))]), [{"value": 1}])
                self.assertEqual(read(p)["status"], "completed")
