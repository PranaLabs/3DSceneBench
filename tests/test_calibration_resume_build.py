"""Actual calibration keeps cameras and physics results across partial resume."""

import tempfile
import json
import unittest
from pathlib import Path
from sbv.common.calibration_jobs import Journal, encoded
from sbv.appearance.heldout import PROTOCOL


class ResumeBuild(unittest.TestCase):
    def test_real_case_resume_matches_fresh_calibration(self):
        from sbv import calibrate
        from tests import rooms
        from sbv.common.calibration_jobs import encoded

        xml = """<mujoco><worldbody><geom type="plane" size="0 0 .1"/>
        <body name="box__0" pos="0 0 .05"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
        <body name="anchor__0" pos="2 2 .3"><geom type="box" size=".05 .05 .05"/></body></worldbody></mujoco>"""
        with tempfile.TemporaryDirectory() as tmp:
            room = str(Path(rooms.write(str(Path(tmp) / "room"), xml)).parent)
            logs = []
            warned = lambda: any("exempt from placement" in s and "anchor__0" in s for s in logs)
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
            first = calibrate.build(room, **args)
            self.assertTrue(warned())
            logs.clear()
            prepared = next((Path(tmp) / "cache").glob("*/prepare/*.json"))
            self.assertTrue(
                json.loads(prepared.read_text())["stage"].startswith("warning: exempt from placement")
            )  # its last note
            second = calibrate.build(room, **args)
            self.assertTrue(warned())  # the room preparation is reused: still said
            self.assertTrue(any("objects: 2/2 completed (reused)" in s for s in logs))
            # The held-out cameras come from the room (heldout.build), not from a camera seed: the same sixteen after
            # resume.
            self.assertNotIn("camera_seed", first["appearance"])
            self.assertEqual(first["appearance"]["heldout_protocol"]["protocol"], PROTOCOL)
            self.assertEqual(first["appearance"]["heldout_protocol"], second["appearance"]["heldout_protocol"])
            first["calibration"].pop("seconds")
            second["calibration"].pop("seconds")
            self.assertEqual(encoded(first), encoded(second))
            # Losing one object result must preserve all other checkpoints and the case.
            object_records = list((Path(tmp) / "cache").glob("*/objects/*.json"))
            object_records[0].unlink()
            logs.clear()
            partial = calibrate.build(room, **args)
            partial["calibration"].pop("seconds")
            self.assertEqual(encoded(first), encoded(partial))
            self.assertTrue(any("objects: 1/2 completed (reused)" in s for s in logs))
