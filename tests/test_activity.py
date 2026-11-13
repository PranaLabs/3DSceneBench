"""An activity (tasks.json "activity"): a long-horizon task of several steps, done in order in one unbroken run of the
robot room. The reader resolves each step as its own kind would be, refuses the whole activity with the step's number
and reason, and needs words; the trial runs the steps without a reset and names the step that fails."""

import tempfile, unittest
from unittest import mock
from sbv.common.scene import Scene, PHYSICS
from sbv.calibrate import room_bounds
from sbv.functionality import f2_calibrate as C
from sbv.functionality.f4_robot import plan as RP, tasks as T, pick_place
from sbv.functionality.units import robot_trial
from tests import rooms

STEPS = [
    dict(kind="pick_place", object="cup__0", goal="cabinet__0"),
    dict(kind="push", object="cup__0", direction=[0, 1]),
]


class Activity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        xml = rooms.write(tempfile.mkdtemp(), rooms.robot_room_xml())
        cls.s = s = Scene(xml, dict(PHYSICS), probes=True, textures=False)
        s.settle()
        cls.b = b = room_bounds(s)
        cls.c = c = [(b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2]
        cls.ctx = C.context(s, c)

    def named(self, spec):
        return T.named(self.s, self.ctx, spec, self.c, self.b)

    def test_the_steps_are_resolved_like_their_own_kinds(self):
        units, bad = self.named(
            dict(activity=[dict(words="Put the cup on the cabinet, then push it along.", steps=STEPS)])
        )
        self.assertEqual(bad, [])
        u = units[0]
        self.assertEqual(
            (u["kind"], u["id"], u["slots"], u["words"]),
            (
                "robot_activity",
                "robot_activity|0",
                ["cup__0", "cabinet__0"],
                "Put the cup on the cabinet, then push it along.",
            ),
        )
        self.assertEqual(len(u["params"]["steps"][0]["rel"]), 2)
        self.assertEqual(u["params"]["steps"][1]["dir"], [0.0, 1.0])

    def test_refused_with_the_steps_number_and_reason(self):
        for spec, reason in [
            (dict(steps=STEPS), "needs words"),
            (dict(words="x", steps=[]), "at least one step"),
            (
                dict(words="x", steps=[STEPS[0], dict(kind="fly", object="cup__0")]),
                "step 2 (fly): unknown step kind 'fly'",
            ),
            (
                dict(words="x", steps=[dict(kind="pick_place", object="mug__9", goal="cabinet__0")]),
                "step 1 (pick_place): no object named mug__9",
            ),
        ]:
            units, bad = self.named(dict(activity=[spec]))
            self.assertEqual(units, [])
            self.assertIn(reason, bad[0][1])

    def test_a_missing_object_names_the_copy_gap(self):
        u = self.named(dict(activity=[dict(words="x", steps=STEPS)]))[0][0]
        u["params"]["near"] = []
        out, mons = robot_trial("robot_activity")(self.s, u, {"cup__0": "cup__0", "cabinet__0": None}, 0)
        self.assertEqual((out, mons[0].details["reason"]), ([False], "missing in the copy: cabinet__0"))

    @unittest.skipUnless(RP.installed(), "needs the RB-Y1 model (scripts/get_robot.sh)")
    def test_the_steps_run_in_one_go_and_a_failed_step_is_named(self):
        u = self.named(dict(activity=[dict(words="x", steps=STEPS)]))[0][0]
        u["params"]["near"] = RP.nearby(self.s, u["slots"])
        self.s.reset()
        out, mons = robot_trial("robot_activity")(self.s, u, None, 0)
        det = mons[0].details
        self.assertEqual(
            (out, [x["kind"] for x in det["steps"]], [x["success"] for x in det["steps"]]),
            ([True], ["pick_place", "push"], [True, True]),
            det,
        )
        self.assertGreater(det["steps"][1]["seconds"], 0)
        self.assertAlmostEqual(det["seconds"], det["steps"][0]["seconds"] + det["steps"][1]["seconds"], delta=0.2)
        u["params"]["steps"][0]["rel"] = [0.45, 0.0]  # a named spot past the cabinet's edge the robot
        self.s.reset()
        out, mons = robot_trial("robot_activity")(self.s, u, None, 0)
        det = mons[0].details  # sets it down elsewhere on the top
        self.assertEqual(out, [True], det)
        self.assertGreater(det["steps"][0]["spot_cm"], 0)
        with mock.patch.object(
            pick_place, "spots", lambda *a, **k: []
        ):  # no clear spot on the top: the named one, the cup falls
            self.s.reset()
            out, mons = robot_trial("robot_activity")(self.s, u, None, 0)
            det = mons[0].details
        self.assertEqual((out, det.get("failed_step"), len(det["steps"])), ([False], 1, 1))
        self.assertTrue(det["reason"].startswith("step 1 (pick_place): "), det["reason"])


if __name__ == "__main__":
    unittest.main()


class Credit(unittest.TestCase):
    def test_an_activity_scores_by_the_steps_done_in_order(self):
        from sbv.functionality.units import judge, SETTINGS, RULES
        from sbv.functionality import aggregate as AG

        class Mon:
            owners = [1]
            details = dict(
                success=False,
                started=True,
                steps=[
                    dict(kind="open", success=True),
                    dict(kind="pick_place", success=True),
                    dict(kind="close", success=False),
                ],
            )

            def ok_rules(self, owners, sink=True):
                return True

            def report(self, owners):
                return {}

        unit = dict(
            id="robot_activity|0",
            kind="robot_activity",
            slots=["a"],
            params=dict(steps=[{}, {}, {}]),
            expect=[True],
            rules=RULES,
            named=True,
        )
        r = judge(unit, [False], [Mon()])
        self.assertEqual(r["passed"]["input_only"], False)
        self.assertAlmostEqual(r["credit"]["input_only"], 2 / 3)
        self.assertEqual(
            r["stage_points"], [11 * 12 / 16, 12.0]
        )  # open 5 + pick 6 of open 5 + pick 6 + close 5, capped at 12
        self.assertAlmostEqual(r["stage_score"], 11 / 16, places=3)  # the third step did not start
        lv = AG.levels(1.0, [unit], {"robot_activity|0": [r]})
        self.assertAlmostEqual(lv["input_only"]["interaction"][0], 11 / 16, places=3)
        self.assertEqual(
            AG.level_points(1.0, [unit], {"robot_activity|0": [r]})["input_only"]["interaction"], [[8.25, 12.0]]
        )
        bad = Mon()
        bad.ok_rules = lambda owners, sink=True: False
        self.assertEqual(
            judge(unit, [False], [bad])["credit"]["input_only"], 0.0
        )  # a run that broke a rule earns nothing
        self.assertEqual(AG.value(dict(passed={s: True for s in SETTINGS}), "passed", "input_only"), 1.0)
