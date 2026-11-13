"""Task iteration reuses rigid measurements without weakening full qualification identity."""

import json
import shutil
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from sbv.common.calibration_inputs import rigid_identity, room_identity, source_identity, import_legacy
from sbv.common.calibration_jobs import identity, Journal, encoded
from tests.test_calibration_bounded_jobs import work


class RigidInputs(unittest.TestCase):
    def room(self, tmp):
        root = Path(tmp) / "room"
        root.mkdir()
        (root / "scene.xml").write_text("<mujoco><worldbody/></mujoco>")
        (root / "package_tasks").mkdir()
        (root / "package_tasks/action.py").write_text("def run(ctx): return {}\n")
        spec = {"package": [dict(id="open", entry="package_tasks/action.py", params={"force": 5})]}
        (root / "tasks.json").write_text(json.dumps(spec))
        return root, spec

    def test_controller_and_parameters_do_not_invalidate_rigid_but_invalidate_full(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, spec = self.room(tmp)
            full = identity(root, {})
            rigid = rigid_identity(root, {})
            (root / "package_tasks/action.py").write_text('def run(ctx): return {"changed": True}\n')
            spec["package"][0]["params"]["force"] = 8
            (root / "tasks.json").write_text(json.dumps(spec))
            self.assertEqual(rigid_identity(root, {}), rigid)
            self.assertNotEqual(identity(root, {}), full)
            spec["push"] = [dict(object="cup__0")]
            (root / "tasks.json").write_text(json.dumps(spec))
            self.assertNotEqual(rigid_identity(root, {}), rigid)

    def test_actual_physics_and_shared_model_assets_remain_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self.room(tmp)
            # An unusual MJCF include inside the controller folder must never be hidden.
            (root / "package_tasks/part.py").write_text("<mujoco><worldbody/></mujoco>")
            (root / "scene.xml").write_text('<mujoco><include file="package_tasks/part.py"/></mujoco>')
            a = room_identity(root)
            (root / "package_tasks/part.py").write_text("<mujoco><worldbody><body/></worldbody></mujoco>")
            self.assertNotEqual(a, room_identity(root))
            a = rigid_identity(root, {})
            (root / "physics.json").write_text('{"mass": 2}')
            self.assertEqual(a, rigid_identity(root, {}))  # unused authoring metadata is not physical input
            (root / "package_tasks/part.py").write_text(
                '<mujoco><worldbody><body><geom size=".1"/></body></worldbody></mujoco>'
            )
            self.assertNotEqual(a, rigid_identity(root, {}))
            self.assertNotEqual(rigid_identity(root, {}), rigid_identity(root, {"jitter": 2}))

    def test_identity_import_is_normalized_but_real_calibration_changes_are_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "calibrate.py"
            p.write_text("from sbv.common.calibration_jobs import Journal, identity, note\ndef run(): return 1\n")
            a = source_identity(root)
            p.write_text(
                "from sbv.common.calibration_jobs import Journal, note\nfrom sbv.common.calibration_inputs import rigid_identity as identity\ndef run(): return 1\n"
            )
            self.assertEqual(a, source_identity(root))
            p.write_text(p.read_text().replace("return 1", "return 2"))
            self.assertNotEqual(a, source_identity(root))

    def test_real_build_reuses_rigid_jobs_after_package_edit(self):
        from sbv import calibrate
        from tests import rooms

        xml = '<mujoco><worldbody><geom type="plane" size="0 0 .1"/><body name="box__0" pos="0 0 .05"><freejoint/><geom type="box" size=".05 .05 .05"/></body><body name="anchor__0" pos="2 2 .3"><geom type="box" size=".05 .05 .05"/></body></worldbody></mujoco>'
        with tempfile.TemporaryDirectory() as tmp:
            room = Path(rooms.write(str(Path(tmp) / "room"), xml)).parent
            task = room / "action.py"
            task.write_text("def run(ctx): return {}\n")
            (room / "tasks.json").write_text(json.dumps({"package": [dict(entry="action.py")]}))
            logs = []
            args = dict(
                procs=1,
                draws=3,
                need=2,
                jitter=1,
                robot_tasks=0,
                push_tasks=0,
                require_tasks=False,
                checkpoint=str(Path(tmp) / "cache"),
                log=logs.append,
            )
            first = calibrate.build(str(room), **args)
            task.write_text('def run(ctx): return {"different": True}\n')
            logs.clear()
            second = calibrate.build(str(room), **args)
            self.assertTrue(any("objects: 2/2 completed (reused)" in s for s in logs), logs)
            first["calibration"].pop("seconds")
            second["calibration"].pop("seconds")
            self.assertEqual(encoded(first), encoded(second))

    def test_legacy_import_preserves_measured_records_and_rejects_changed_inputs(self):
        import sbv

        with tempfile.TemporaryDirectory() as tmp:
            room, _ = self.room(tmp)
            cache = Path(tmp) / "cache"
            snapshot = Path(tmp) / "snapshot"
            source = Path(sbv.__file__).parent
            for p in source.rglob("*.py"):
                q = snapshot / p.relative_to(source)
                q.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, q)
            with Journal(cache, identity(room, {})) as old:
                old.jobs("objects", work, [("measured", {"value": 4})])
                record = next(old.root.glob("objects/*.json")).read_bytes()
            row = import_legacy(room, {}, cache, snapshot)
            self.assertEqual(row["records_copied"], 1)
            self.assertEqual(next(Path(row["destination"]).glob("objects/*.json")).read_bytes(), record)
            with Journal(cache, rigid_identity(room, {})) as new:
                with patch("sbv.common.calibration_jobs._execute", side_effect=AssertionError("must reuse")):
                    self.assertEqual(new.jobs("objects", work, [("measured", {"value": 4})]), [{"value": 4}])
            (room / "physics.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "legacy inputs"):
                import_legacy(room, {}, cache, snapshot)
            p = snapshot / "functionality/f3_functions.py"
            p.write_text(p.read_text() + "\nPHYSICS_CHANGED = True\n")
            with self.assertRaisesRegex(ValueError, "execution code differs"):
                import_legacy(room, {}, cache, snapshot)


if __name__ == "__main__":
    unittest.main()
