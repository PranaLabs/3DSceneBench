"""Calibration runs an object's local tests, declared holds, each declared opening and each joint as separate jobs, and
losing one joint's record reruns only that joint, with the same case."""

import json, tempfile, unittest
from pathlib import Path
from sbv import calibrate
from sbv.common.calibration_jobs import encoded
from tests.test_declared import walls

DOOR = (
    '<body name="door_{0}" pos="0 {1} .5"><joint name="hinge_{0}" type="hinge" axis="0 0 1" range="0 1.5" damping=".1"/>'
    '<geom type="box" pos=".12 0 0" size=".1 .01 .2" mass="1"/></body>'
)
ROOM = (
    '<mujoco><compiler angle="radian"/><worldbody><geom name="floor" type="plane" size="3 3 .1"/>'
    f'<body name="bin__0" pos="1.5 0 0"><freejoint/>{walls(0, 0.06, 0.2)}</body>'
    '<body name="bottle__8" pos="1.5 0 .05"><freejoint/><geom type="cylinder" size=".025 .04" mass=".05"/></body>'
    '<body name="cabinet__9" pos="-1.5 -1 0"><geom type="box" pos="-.05 0 .25" size=".03 .3 .25"/>'
    + DOOR.format("a", -0.15)
    + DOOR.format("b", 0.15)
    + "</body></worldbody></mujoco>"
)
FUNCTIONS = dict(
    containers={"bin__0": dict(holds=["bottle__8"], openings=[dict(pos=[0, 0, 0.21], dir=[0, 0, -1], radius=0.05)])}
)


class UnitJobs(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        room = self.tmp / "room"
        room.mkdir()
        self.room = str(room)
        (room / "scene.xml").write_text(ROOM)
        (room / "functions.json").write_text(json.dumps(FUNCTIONS))

    def build(self, procs, logs):
        case = calibrate.build(
            self.room,
            procs=procs,
            draws=1,
            need=1,
            jitter=1,
            robot_tasks=0,
            push_tasks=0,
            require_tasks=False,
            camera_seed=7,
            checkpoint=str(self.tmp / "checkpoint"),
            log=logs.append,
        )
        case["calibration"].pop("seconds")
        return encoded(case)

    def test_one_job_per_part_and_a_part_rerun_alone(self):
        logs = []
        case = self.build(3, logs)
        root = next((self.tmp / "checkpoint").glob("*/manifest.json")).parent
        jobs = json.loads((root / "objects-summary.json").read_text())["jobs"]
        self.assertEqual(
            sorted(jobs),
            [
                "bin__0|holds",
                "bin__0|opening|0",
                "bin__0|tests",
                "bottle__8|tests",
                "cabinet__9|joint|hinge_a",
                "cabinet__9|joint|hinge_b",
                "cabinet__9|tests",
            ],
        )
        for uid in ("joint|cabinet__9|hinge_b", "opening|bin__0|0", "holds|bin__0|bottle__8"):
            self.assertIn(uid, case)
        (root / jobs["cabinet__9|joint|hinge_b"]).unlink()
        logs.clear()
        self.assertEqual(self.build(1, logs), case)  # the lost joint reruns alone, in a fresh worker
        self.assertTrue(any("objects: 6/7 completed (reused)" in s for s in logs), logs)


if __name__ == "__main__":
    unittest.main()
