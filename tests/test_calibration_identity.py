"""Checkpoint identity and validation reject stale, corrupt and incomplete work."""

import json
import os
import time
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from sbv.common.calibration_jobs import Journal, read, identity
from tests.test_calibration_bounded_jobs import work


class CacheIdentity(unittest.TestCase):
    def test_external_assets_and_settings_invalidate(self):
        from sbv.functionality.f4_robot import rby1

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            room = p / "room"
            room.mkdir()
            asset = p / "external.obj"
            asset.write_text("first")
            (room / "scene.xml").write_text('<mujoco><asset><mesh file="../external.obj"/></asset></mujoco>')
            with patch.object(rby1, "ROBOT_DIR", str(p / "absent")):
                a = identity(room, {"draws": 10})
                asset.write_text("second")
                b = identity(room, {"draws": 10})
                self.assertNotEqual(a, b)
                self.assertNotEqual(b, identity(room, {"draws": 9}))

    def test_a_room_word_that_changes_meaning_invalidates_and_an_added_word_does_not(self):
        from sbv.functionality.f4_robot import rby1
        from sbv.common import scene as Words

        with tempfile.TemporaryDirectory() as tmp:
            room = Path(tmp) / "room"
            room.mkdir()
            (room / "scene.xml").write_text(
                '<mujoco><worldbody><body name="cup__0"><geom size="0.1"/></body></worldbody></mujoco>'
            )
            with patch.object(rby1, "ROBOT_DIR", str(Path(tmp) / "absent")):
                a = identity(room, {})
                self.assertEqual(a["words"], {"cup": ["cup", "kitchen and dining"]})
                with patch.dict(Words.TOPIC, {"zorbleflux": "decor"}):
                    self.assertEqual(identity(room, {}), a)  # a word added to the list
                with patch.dict(Words.ALIAS, {"cup": "bowl"}):
                    self.assertNotEqual(identity(room, {}), a)  # cup now scored as bowl
                with patch.dict(Words.TOPIC, {"cup": "decor"}):
                    self.assertNotEqual(identity(room, {}), a)  # cup under another topic

    def test_missing_external_asset_refuses_resume(self):
        from sbv.common.calibration_jobs import asset_dependencies

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "scene.xml").write_text('<mujoco><asset><mesh file="absent.obj"/></asset></mujoco>')
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                asset_dependencies(p)


class ResumeValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.logs = []

    def journal(self, **kw):
        return Journal(self.tmp.name, {"fixture": 1}, self.logs.append, **kw)

    def test_corrupt_result_and_changed_arguments_are_not_reused(self):
        with self.journal() as j:
            j.jobs("objects", work, [("a", dict(value=1))])
            p = next(j.root.glob("objects/*.json"))
            r = read(p)
            r["result"] = {"value": 900}
            p.write_text(json.dumps(r))
            self.assertEqual(j.jobs("objects", work, [("a", dict(value=1))]), [{"value": 1}])
            self.assertEqual(j.jobs("objects", work, [("a", dict(value=2))]), [{"value": 2}])
            p.write_text("{broken")
            self.assertEqual(j.jobs("objects", work, [("a", dict(value=1))]), [{"value": 1}])

    def test_identity_and_no_resume(self):
        with self.journal() as j:
            j.jobs("objects", work, [("a", dict(value=1))])
            root = j.root
            p = next(root.glob("objects/*.json"))
            stamp = p.stat().st_mtime_ns
        with self.journal(resume=False) as j:
            j.jobs("objects", work, [("a", dict(value=1))])
        self.assertNotEqual(stamp, p.stat().st_mtime_ns)
        with Journal(self.tmp.name, {"fixture": 2}, self.logs.append) as j:
            self.assertNotEqual(root, j.root)

    def test_duplicate_ids_and_lock(self):
        with self.journal() as j:
            with self.assertRaisesRegex(RuntimeError, "already in use"):
                with self.journal():
                    pass
            with self.assertRaisesRegex(ValueError, "duplicate"):
                j.jobs("objects", work, [("a", {}), ("a", {})])

    def test_progress_reaches_a_redirected_log(self):
        # Run like `python -m sbv.calibrate ... > calibrate.log`: stdout is a file, so print() buffers it.
        import subprocess, sys

        code = (
            "import time; from sbv.common.calibration_jobs import Journal\n"
            "j = Journal(%r, {'room': 'x'}, timeout=60)\n"
            % self.tmp.name
            + "j.log('first progress line'); time.sleep(30)"
        )
        with open(Path(self.tmp.name) / "out.log", "w") as out:
            p = subprocess.Popen([sys.executable, "-c", code], stdout=out, cwd=str(Path(__file__).parents[1]))
        try:
            deadline = time.time() + 20
            while time.time() < deadline and "first progress line" not in (Path(self.tmp.name) / "out.log").read_text():
                time.sleep(0.2)
            self.assertIn("first progress line", (Path(self.tmp.name) / "out.log").read_text())
        finally:
            p.kill()
            p.wait()
