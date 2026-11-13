"""Completed calibration trials survive staging and retry; live scoring bypasses the cache."""

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from sbv.package_tasks import calibration_cache as C
from tests.test_package_tasks import room_with


class CalibrationReuse(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        _, room = room_with()
        self.room = Path(room)
        self.addCleanup(shutil.rmtree, self.room.parent, True)
        self.entry = str(self.room / "package_tasks/push.py")
        self.ctx = dict(
            scene_dir=str(self.room),
            scene_xml=str(self.room / "scene.xml"),
            nonrigid=None,
            objects={"book__0": "book__0"},
            params={"force": 1.0},
            seed=0,
        )
        self.calls = 0
        self.work = self.root / "work"
        self.env = patch.dict(os.environ, SBV_PACKAGE_CACHE=str(self.root / "cache"))
        self.env.start()
        self.addCleanup(self.env.stop)

    def execute(self):
        self.calls += 1
        shutil.rmtree(self.work, ignore_errors=True)
        self.work.mkdir()
        out = dict(
            stages=[["moved", True]],
            score=1.0,
            success=True,
            finite=True,
            jitter_applied=False,
            metrics={"x": 0.1},
            why="measured",
            seconds=1.25,
        )
        (self.work / "result.json").write_text(json.dumps(out))
        (self.work / "stdout.log").write_text("real original log")
        return out

    def trial(self):
        return C.completed(self.entry, "mujoco", self.ctx, str(self.work), 120, 1, sys.executable, self.execute)

    def test_completed_result_and_logs_reused_with_provenance(self):
        a = self.trial()
        b = self.trial()
        self.assertEqual(self.calls, 1)
        self.assertEqual(a, b)
        self.assertEqual((self.work / "stdout.log").read_text(), "real original log")
        self.assertTrue((self.work / "calibration-reuse.json").is_file())

    def test_staged_folder_reuses_same_executable_inputs(self):
        self.trial()
        dest = self.root / "staged"
        shutil.copytree(self.room, dest)
        self.room = dest
        self.entry = str(dest / "package_tasks/push.py")
        self.ctx.update(scene_dir=str(dest), scene_xml=str(dest / "scene.xml"))
        self.trial()
        self.assertEqual(self.calls, 1)

    def test_negative_scene_keeps_truth_code_identity_across_staging(self):
        wrong = self.root / "wrong"
        shutil.copytree(self.room, wrong)
        self.ctx.update(scene_dir=str(wrong), scene_xml=str(wrong / "scene.xml"))
        first = C.identity(self.entry, "mujoco", self.ctx, 120, 1, sys.executable, code_room=str(self.room))
        staged = self.root / "different/deeper/truth"
        staged.parent.mkdir(parents=True)
        shutil.copytree(self.room, staged)
        second = C.identity(
            str(staged / "package_tasks/push.py"), "mujoco", self.ctx, 120, 1, sys.executable, code_room=str(staged)
        )
        self.assertEqual(first, second)

    def test_parameter_and_seed_changes_invalidate(self):
        self.trial()
        self.ctx["params"]["force"] = 2.0
        self.trial()
        self.ctx["seed"] = 1
        self.trial()
        self.assertEqual(self.calls, 3)

    def test_model_and_entry_changes_invalidate(self):
        self.trial()
        p = self.room / "scene.xml"
        p.write_text(p.read_text().replace('density="700"', 'density="1400"'))
        self.trial()
        p = Path(self.entry)
        p.write_text(p.read_text() + "\n# controller changed\n")
        self.trial()
        self.assertEqual(self.calls, 3)

    def test_edited_evidence_invalidates(self):
        self.trial()
        p = next((self.root / "cache").glob("*/run/stdout.log"))
        p.write_text("tampered")
        self.trial()
        self.assertEqual(self.calls, 2)

    def test_errors_are_not_cached(self):
        def failed():
            self.calls += 1
            return {"error": "transport stopped"}

        for _ in range(2):
            C.completed(self.entry, "mujoco", self.ctx, str(self.work), 120, 1, sys.executable, failed)
        self.assertEqual(self.calls, 2)
        self.assertFalse((self.root / "cache").exists())

    def test_inputs_changed_during_execution_are_not_stored(self):
        def changing():
            result = self.execute()
            p = self.room / "scene.xml"
            p.write_text(p.read_text().replace('density="700"', 'density="1400"'))
            return result

        C.completed(self.entry, "mujoco", self.ctx, str(self.work), 120, 1, sys.executable, changing)
        self.assertFalse((self.root / "cache").exists())

    def test_disabled_cache_always_executes(self):
        with patch.dict(os.environ, SBV_PACKAGE_CACHE=""):
            self.trial()
            self.trial()
        self.assertEqual(self.calls, 2)

    def test_native_calibration_reuses_completed_positive_and_negative_runs(self):
        from sbv.package_tasks import verify as V

        out = str(self.root / "package_case.json")
        with patch.object(V, "run_one", wraps=V.run_one) as run:
            original = V.calibrate(str(self.room), out, procs=2)
            count = run.call_count
            repeated = V.calibrate(str(self.room), out, procs=2)
        self.assertEqual(count, 4)
        self.assertEqual(run.call_count, count)
        self.assertEqual(original, repeated)

    def test_evaluation_keeps_direct_execution(self):
        from sbv.package_tasks import verify as V

        out = self.root / "case.json"
        V.calibrate(str(self.room), str(out), procs=1)
        report = self.root / "report.json"
        report.write_text(json.dumps({"geometry": {}, "correspondence": {"book__0": "book__0"}}))
        with patch.object(C, "completed", side_effect=AssertionError("live grading must execute")):
            V.evaluate(str(out), str(self.room), str(report), str(self.root / "eval.json"), procs=1)
