"""End-to-end authoring dispatch with counted engines; no fabricated simulation evidence."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from sbv import pipeline as P, qualification as Q
from sbv.appearance import cache_identity as M
from sbv.common import case as K
from sbv.common.scene import PHYSICS
from tests import rooms
from tests.test_appearance_floor import SCORES


class Qualification(unittest.TestCase):
    def test_appearance_edit_does_not_rerun_physics_and_no_resume_executes_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            room = Path(rooms.write(str(Path(tmp) / "room"))).parent
            out = Path(tmp) / "out"
            case = dict(
                schema=K.SCHEMA,
                physics=dict(PHYSICS),
                room=dict(bounds=[[0, 0, 0], [4, 3, 2.5]]),
                appearance=dict(resolution=64, samples=1, cameras=[], input_cameras=[]),
                objects=[],
                functionality=dict(units=[], excluded=[]),
            )
            calls = dict(calibration=0, physics=0, appearance=0, floor=0)

            def build(*a, **kw):
                calls["calibration"] += 1
                return copy.deepcopy(case)

            def rigid(*a, **kw):
                calls["physics"] += 1
                return dict(geometry={}, units={}, F1={"passed": True})

            def appearance(*a, **kw):
                calls["appearance"] += 1
                return dict(mean={k: 1 for k in P.App.METRICS})

            def floor(*a, **kw):
                calls["floor"] += 1
                return dict(siglip2=0.5)

            with (
                patch.object(P.calibrate, "build", side_effect=build),
                patch.object(P.evaluate, "evaluate", side_effect=rigid),
                patch.object(P.App, "evaluate", side_effect=appearance),
                patch.object(P.App, "floor", side_effect=floor),
                patch.object(P.App, "ray_depths", return_value=[]),
                patch.object(P.App, "near_cameras", return_value=[]),
                patch.object(P, "scores", return_value=dict(SCORES)),
                patch.object(M, "identity", return_value={"weights": "fixture"}),
            ):
                first = P.check(str(room), str(out))
                self.assertTrue(first["ready"], first)
                P.check(str(room), str(out))
                self.assertEqual(calls, dict(calibration=1, physics=1, appearance=1, floor=1))
                xml = room / "scene.xml"
                xml.write_text(xml.read_text().replace("<worldbody>", '<worldbody><light pos="0 0 2"/>'))
                P.check(str(room), str(out))
                self.assertEqual(calls, dict(calibration=1, physics=1, appearance=2, floor=2))
                P.check(str(room), str(out), resume=False)
                self.assertEqual(calls, dict(calibration=2, physics=2, appearance=3, floor=3))
                self.assertTrue(P.reusable(json.loads((out / "check.json").read_text()), str(room), str(out)))

    def test_unknown_metric_weights_do_not_promote_a_visual_pass_to_reusable(self):
        from sbv.common.calibration_jobs import identity

        with tempfile.TemporaryDirectory() as tmp:
            room = Path(rooms.write(str(Path(tmp) / "room"))).parent
            out = Path(tmp) / "out"
            out.mkdir()
            old = dict(
                ready=True,
                status="passed",
                inputs=identity(room, {}),
                version=P.version(),
                artifacts={},
                self={"siglip2": 1.0},
            )
            old["inputs"]["weights"] = None
            with patch.object(P, "identity", return_value=old["inputs"]):
                self.assertFalse(P.reusable(old, str(room), str(out)))
