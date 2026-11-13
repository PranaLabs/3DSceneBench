"""Open and close steps (F4.4): the reader finds the part by name and needs a handle; the robot pulls the drawer open
by its bar handle and pushes it shut, and swings the door open by its knob, each as a step of an activity."""

import tempfile, unittest
from sbv.common.scene import Scene, PHYSICS
from sbv.calibrate import room_bounds
from sbv.functionality import f2_calibrate as C
from sbv.functionality.f4_robot import plan as RP, tasks as T
from sbv.functionality.units import robot_trial
from tests import rooms


def room(knob=True):
    xml = rooms.write(tempfile.mkdtemp(), rooms.cabinet_room_xml(knob))
    s = Scene(xml, dict(PHYSICS), probes=True, textures=False)
    s.settle()
    b = room_bounds(s)
    c = [(b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2]
    return s, C.context(s, c), c, b


def activity(s, ctx, c, b, steps):
    units, bad = T.named(s, ctx, dict(activity=[dict(words="x", steps=steps)]), c, b)
    return units, bad


class OpenClose(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s, cls.ctx, cls.c, cls.b = room()

    def test_the_part_is_found_by_name_and_needs_a_handle(self):
        units, bad = activity(
            self.s,
            self.ctx,
            self.c,
            self.b,
            [
                dict(kind="open", object="cabinet__0", part="drawer"),
                dict(kind="close", object="cabinet__0", part="drawer_slide"),
            ],
        )
        self.assertEqual(bad, [])
        ref = units[0]["params"]["steps"][0]["ref"]
        self.assertEqual((ref["name"], ref["type"], ref["sgn"]), ("drawer_slide", 2, -1.0))
        for step, reason in [
            (dict(kind="open", object="cabinet__0"), "has 2 moving parts"),
            (dict(kind="open", object="cabinet__0", part="lid"), "no part named 'lid'"),
            (dict(kind="open", object="wall_e"), "no object named wall_e"),
        ]:
            units, bad = activity(self.s, self.ctx, self.c, self.b, [step])
            self.assertEqual(units, [])
            self.assertIn(reason, bad[0][1])
        s2, ctx2, c2, b2 = room(knob=False)
        units, bad = activity(s2, ctx2, c2, b2, [dict(kind="open", object="cabinet__0", part="door")])
        self.assertEqual(units, [])
        self.assertIn("no handle or knob", bad[0][1])

    @unittest.skipUnless(RP.installed(), "needs the RB-Y1 model (scripts/get_robot.sh)")
    def test_the_drawer_opens_and_closes_in_one_run(self):
        u = activity(
            self.s,
            self.ctx,
            self.c,
            self.b,
            [
                dict(kind="open", object="cabinet__0", part="drawer"),
                dict(kind="close", object="cabinet__0", part="drawer"),
            ],
        )[0][0]
        u["params"]["near"] = RP.nearby(self.s, u["slots"])
        self.s.reset()
        out, mons = robot_trial("robot_activity")(self.s, u, None, 0)
        det = mons[0].details
        self.assertEqual((out, [x["success"] for x in det["steps"]]), ([True], [True, True]), det)
        self.assertGreaterEqual(det["steps"][0]["moved"], 0.8 * 0.25)
        self.assertLessEqual(det["steps"][1]["from_closed"], 0.02)

    @unittest.skipUnless(RP.installed(), "needs the RB-Y1 model (scripts/get_robot.sh)")
    def test_the_door_swings_open_by_its_knob(self):
        u = activity(
            self.s, self.ctx, self.c, self.b, [dict(kind="open", object="cabinet__0", part="door", amount=0.9)]
        )[0][0]
        u["params"]["near"] = RP.nearby(self.s, u["slots"])
        self.s.reset()
        out, mons = robot_trial("robot_activity")(self.s, u, None, 0)
        det = mons[0].details
        self.assertEqual(out, [True], det)
        self.assertGreaterEqual(det["steps"][0]["moved"], 0.72)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(RP.installed(), "needs the RB-Y1 model (scripts/get_robot.sh)")
class Calibrated(unittest.TestCase):
    def test_calibration_keeps_the_activity_and_the_instruction_says_it(self):
        import json, os
        from sbv import calibrate
        from scenebench_harbor.make_tasks import room_tasks

        d = tempfile.mkdtemp()
        rooms.write(d, rooms.cabinet_room_xml())
        with open(os.path.join(d, "tasks.json"), "w") as f:
            json.dump(
                dict(
                    activity=[
                        dict(
                            words="Pull the drawer open by its handle, then push it shut.",
                            steps=[
                                dict(kind="open", object="cabinet__0", part="drawer"),
                                dict(kind="close", object="cabinet__0", part="drawer"),
                            ],
                        )
                    ]
                ),
                f,
            )
        case = calibrate.build(
            d, procs=4, draws=2, need=2, jitter=1, log=lambda *a: None, robot_tasks=0, push_tasks=0, camera_seed=7
        )
        u = [u for u in case["functionality"]["units"] if u["kind"] == "robot_activity"]
        self.assertEqual(len(u), 1, case["functionality"]["excluded"])
        self.assertEqual((u[0]["expect"], u[0]["named"], u[0]["params"]["near"]), ([True], True, []))
        self.assertIn("- Pull the drawer open by its handle, then push it shut.", room_tasks(case))
