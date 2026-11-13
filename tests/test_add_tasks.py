"""sbv.add_tasks: a trial that shares the settle of its start and seed (plan.World.settle_steps, then a forked child)
gives the same details, bit for bit, as calibrate's own trial; and the tool accepts and labels the named tasks of the
robot room as calibrate does (expect, tol, hint_only, near, isolate)."""

import json, os, tempfile, unittest
from unittest import mock
from sbv import add_tasks as A, calibrate
from sbv.common.scene import Scene, PHYSICS
from sbv.functionality.f4_robot import plan as RP
from sbv.functionality.units import robot_trial
from tests import rooms

SPEC = {
    "pick_place": [{"object": "cup__0", "goal": "cabinet__0", "words": "Put the cup on the cabinet."}],
    "push": [{"object": "cup__0"}],
}


@unittest.skipUnless(RP.installed(), "needs the RB-Y1 model (scripts/get_robot.sh)")
class AddTasks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.xml = rooms.write(cls.dir, rooms.robot_room_xml())
        with open(os.path.join(cls.dir, "tasks.json"), "w") as f:
            json.dump(SPEC, f)
        cls.case = calibrate.build(
            cls.dir, procs=4, draws=2, need=2, jitter=1, log=lambda *a: None, robot_tasks=0, push_tasks=0, camera_seed=7
        )

    @mock.patch.dict(os.environ, {RP.FREEZE_ENV: "0"})  # the sharing as it is without frozen trials (plan.mode)
    def test_a_shared_settle_gives_the_same_trial(self):
        A.S = s = Scene(self.xml, PHYSICS, probes=True, textures=False)
        s.settle()
        u = next(u for u in self.case["functionality"]["units"] if u["kind"] == "robot")
        s.reset()
        plain = robot_trial("robot")(s, u, None, 1)[1][0].details
        st = A.world_start(u)
        w = RP.World(s.xml, st, s.textures, s.phys)
        w.jitter(1)
        x0, x1 = w.settle_steps()
        built = RP.World
        try:
            RP.World = lambda *a: w
            w.jitter = lambda seed: None
            w.settle_steps = lambda: (x0, x1)
            s.reset()
            shared = robot_trial("robot")(s, u, None, 1)[1][0].details
        finally:
            RP.World = built
        self.assertEqual(plain, shared)

    def test_frozen_trials_run_calibrate_s_path(self):
        with mock.patch.dict(os.environ, {RP.FREEZE_ENV: "1"}):
            RP._KEPT.clear()
            case = calibrate.build(
                self.dir,
                procs=4,
                draws=2,
                need=2,
                jitter=1,
                log=lambda *a: None,
                robot_tasks=0,
                push_tasks=0,
                camera_seed=7,
            )
            robot = {u["id"]: u for u in case["functionality"]["units"] if u["kind"].startswith("robot")}
            self.assertTrue(robot and all(u["params"]["freeze"] == RP.FROZEN for u in robot.values()))
            RP._KEPT.clear()
            got, dropped, rows = A.add(
                dict(case, calibration=dict(case["calibration"], jitter_seeds=1)),
                self.dir,
                SPEC,
                procs=4,
                log=lambda *a: None,
            )
        self.assertEqual({u["id"]: u for u in got}, {i: u for i, u in robot.items() if not u.get("hint_only")})
        self.assertTrue(all(len(r) == 4 for r in rows.values()))

    def test_the_tool_accepts_what_calibrate_accepts(self):
        robot = {u["id"]: u for u in self.case["functionality"]["units"] if u["kind"].startswith("robot")}
        self.assertTrue(robot)
        got, dropped, rows = A.add(
            dict(self.case, calibration=dict(self.case["calibration"], jitter_seeds=1)),
            self.dir,
            SPEC,
            procs=4,
            log=lambda *a: None,
        )
        self.assertEqual({u["id"]: u for u in got}, {i: u for i, u in robot.items() if not u.get("hint_only")})
        self.assertTrue(all(len(r) == 4 for r in rows.values()))  # seeds 0 and 1, two edits


if __name__ == "__main__":
    unittest.main()
