"""Regression fixtures for stale reuse; no full-room simulations or model downloads."""

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from sbv.common import dependencies as D, case as K, calibration_jobs as J, calibration_inputs as I, scene as S
from sbv import pipeline as P, add_tasks as AT
from sbv.appearance import views as A
from sbv.package_tasks import calibration_cache as C, verify as V


class Dependencies(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.room = self.root / "room"
        self.room.mkdir()
        self.xml = self.room / "scene.xml"
        self.xml.write_text('<mujoco><worldbody><include file="part.xml"/></worldbody></mujoco>')
        self.part = self.room / "part.xml"
        self.part.write_text('<mujoco><body name="cup__0"><geom size=".1"/></body></mujoco>')

    def edit(self):
        self.part.write_text(self.part.read_text().replace(".1", ".2"))

    def case(self):
        return dict(
            schema=K.SCHEMA,
            reference=str(self.xml),
            _dir=str(self.root),
            reference_sha256=K.file_sha256(self.xml),
            reference_files=D.model_files(self.xml),
            appearance=dict(cameras=[], input_cameras=[], resolution=64, samples=1),
            physics={},
            room=dict(bounds=[[0, 0, 0], [1, 1, 1]]),
            objects=[],
        )

    def test_frozen_reference_checks_includes_and_legacy_binds_retained_hashes(self):
        case = self.case()
        self.assertEqual(K.reference_path(case), str(self.xml))
        old = copy.deepcopy(case)
        old.pop("reference_files")
        self.assertEqual(
            K.reference_path(old), str(self.xml)
        )  # a case from before the manifest still loads (scene hash only)
        K.bind_reference(old, case["reference_files"])
        self.assertEqual(K.reference_path(old), str(self.xml))
        self.edit()
        with self.assertRaisesRegex(ValueError, "dependencies changed"):
            K.reference_path(case)
        with self.assertRaisesRegex(ValueError, "retained"):
            K.bind_reference(old, case["reference_files"])

    def test_included_words_and_nested_include_contents_are_hashed(self):
        before = I.rigid_identity(self.room, {})
        self.assertIn("cup", before["words"])
        with patch.dict(S.ALIAS, {"cup": "bowl"}):
            self.assertNotEqual(before, I.rigid_identity(self.room, {}))
        self.edit()
        self.assertNotEqual(before, I.rigid_identity(self.room, {}))

    def test_runtime_environment_and_external_overrides_invalidate(self):
        before = J.identity(self.room, {})
        with patch.dict(os.environ, SBV_LIGHT="42"):
            self.assertNotEqual(before, J.identity(self.room, {}))
        rigid = I.rigid_identity(self.room, {})
        import numpy as np

        with patch.object(np, "__version__", "changed"):
            self.assertNotEqual(rigid, I.rigid_identity(self.room, {}))
        config = self.root / "functions.json"
        config.write_text("{}")
        settings = dict(functions=str(config))
        before = J.identity(self.room, settings)
        config.write_text('{"changed":1}')
        self.assertNotEqual(before, J.identity(self.room, settings))

    def test_notes_and_render_environment_do_not_invalidate_rigid(self):
        before = I.rigid_identity(self.room, {})
        (self.room / "notes.md").write_text("editorial note")
        self.assertEqual(before, I.rigid_identity(self.room, {}))
        with patch.dict(os.environ, SBV_LIGHT="42"):
            self.assertEqual(before, I.rigid_identity(self.room, {}))

    def test_shared_scene_manifest_uses_contents_even_with_same_size_and_mtime(self):
        before = S._files(str(self.xml))
        stat = self.part.stat()
        self.edit()
        os.utime(self.part, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertNotEqual(before, S._files(str(self.xml)))

    def test_render_keys_cover_helper_code_and_runtime_but_ignore_notes(self):
        case = self.case()
        keys = lambda: (A.input_key(case), A.ref_key(case))
        from sbv.common import code_dependencies as CD

        before = keys()
        original = CD.sha
        with patch.object(
            CD, "sha", side_effect=lambda p: "changed" if str(p).endswith("geometry/g2_shape.py") else original(p)
        ):
            self.assertTrue(all(a != b for a, b in zip(before, keys())))
        with patch.object(D, "runtime", return_value={"bpy": "changed"}):
            self.assertTrue(all(a != b for a, b in zip(before, keys())))
        (self.room / "notes.md").write_text("editorial")
        self.assertEqual(before, keys())

    def test_render_external_assets_are_content_dependencies(self):
        external = self.root / "mesh.obj"
        external.write_text("first")
        self.xml.write_text('<mujoco><asset><mesh file="../mesh.obj"/></asset></mujoco>')
        case = self.case()
        before = (A.input_key(case), A.ref_key(case))
        external.write_text("second")
        # A frozen case now refuses the modified dependency, rather than returning an old key.
        with self.assertRaises(ValueError):
            A.ref_key(case)
        case = self.case()
        self.assertNotEqual(before, (A.input_key(case), A.ref_key(case)))

    def test_legacy_robot_keep_covers_includes_and_runtime(self):
        from types import SimpleNamespace

        with (
            patch.object(AT, "S", SimpleNamespace(xml=str(self.xml), phys={})),
            patch.object(AT, "UNITS", {"u": {"params": {}}}),
        ):
            before = AT.trial_name("u", "seed_0")
            self.edit()
            self.assertNotEqual(before, AT.trial_name("u", "seed_0"))

    def test_top_level_reuse_verifies_outputs_and_environment(self):
        out = self.root / "out"
        out.mkdir()
        K.save(self.case(), out / "case.json")
        (out / "self.json").write_text("{}")
        old = dict(
            ready=True,
            status="passed",
            inputs=J.identity(self.room, {}),
            version=P.version(),
            artifacts=P.result_files(out),
        )
        self.assertTrue(P.reusable(old, str(self.room), str(out)))
        with patch.dict(os.environ, SBV_LIGHT="42"):
            self.assertFalse(P.reusable(old, str(self.room), str(out)))
        (out / "case.json").write_text("{}")
        self.assertFalse(P.reusable(old, str(self.room), str(out)))
        old.pop("artifacts")
        self.assertFalse(P.reusable(old, str(self.room), str(out)))

    def test_package_config_is_not_silently_reused(self):
        entry = self.room / "task.py"
        entry.write_text(
            'import json\nfrom pathlib import Path\ndef run(ctx):\n v=json.loads((Path(ctx["scene_dir"])/"config.json").read_text())["v"]\n return {"stages":[["ready",True]],"finite":True,"metrics":{"v":v}}\n'
        )
        config = self.room / "config.json"
        config.write_text('{"v":1}')
        ctx = V.context(str(self.room), {}, {}, 0, files=["config.json"])
        with patch.dict(os.environ, SBV_PACKAGE_CACHE=str(self.root / "cache")):

            def trial(name):
                work = str(self.root / name)
                return C.completed(
                    str(entry),
                    "mujoco",
                    ctx,
                    work,
                    30,
                    1,
                    sys.executable,
                    lambda: V.run_one(str(entry), "mujoco", ctx, work, 30),
                )

            self.assertEqual(trial("a")["metrics"]["v"], 1)
            config.write_text('{"v":2}')
            self.assertEqual(trial("b")["metrics"]["v"], 2)
            self.assertFalse((self.root / "b/calibration-reuse.json").exists())
            self.assertEqual(trial("c")["metrics"]["v"], 2)
            self.assertTrue((self.root / "c/calibration-reuse.json").exists())

    def test_soft_cache_rejects_changed_model_threads_mesh_and_corruption(self):
        from sbv.nonrigid import verify as Nonrigid
        from types import SimpleNamespace

        mesh = self.room / "cloth.obj"
        mesh.write_text("first")
        obj = dict(kind="thin_shell", shape={"type": "mesh", "file": "cloth.obj"})  # only a mesh shape names a file
        result = SimpleNamespace(stdout='RESULT {"readout":1,"finite":true}', stderr="")
        work = str(self.root / "nonrigid")
        with (
            patch.object(D, "runtime", return_value={"fixture": "runtime"}),
            patch.object(Nonrigid.subprocess, "run", return_value=result) as engine,
        ):

            def trial(threads=1):
                return Nonrigid._one(str(self.xml), obj, {}, None, work, threads=threads)

            trial()
            trial()
            self.assertEqual(engine.call_count, 1)
            self.edit()
            trial()
            self.assertEqual(engine.call_count, 2)
            trial(2)
            self.assertEqual(engine.call_count, 3)
            mesh.write_text("second")
            trial(2)
            self.assertEqual(engine.call_count, 4)
            cache = Path(json.loads((self.root / ".nonrigid-cache/nonrigid-current.json").read_text())["original"])
            saved = json.loads(cache.read_text())
            saved["result"]["readout"] = 999
            cache.write_text(json.dumps(saved))
            trial(2)
            self.assertEqual(engine.call_count, 5)
            cache.write_text("{broken")
            trial(2)
            self.assertEqual(engine.call_count, 6)

    def test_frozen_soft_case_checks_shape_mesh_before_pairing(self):
        from sbv.nonrigid import verify as Nonrigid

        mesh = self.room / "cloth.obj"
        mesh.write_text("original")
        obj = dict(name="cloth__0", kind="thin_shell", shape={"file": "cloth.obj"})
        inputs = Nonrigid.reference_inputs(str(self.room), [obj])
        case = dict(
            room=str(self.room), inputs=inputs, units=[dict(id="S1.x", truth=1, tolerance=0.1, truth_object=obj)]
        )
        path = self.root / "nonrigid_case.json"
        path.write_text(json.dumps(case))
        mesh.write_text("changed")
        with self.assertRaisesRegex(ValueError, "reference inputs"):
            Nonrigid.evaluate(str(path), "/unused", "/unused")
