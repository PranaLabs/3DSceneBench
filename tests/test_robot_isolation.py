"""Robot tasks in isolation (f4_robot/plan.World.settle): the robot room with a crate far away. The crate sleeps, and
the case and the room's score against itself are the same with isolation on and off."""

import copy, os, tempfile, unittest
import mujoco
from sbv import calibrate, evaluate
from sbv.functionality.f4_robot import plan as RP
from tests import rooms

FAR = '<body name="crate__1" pos="0.5 3.5 0.15"><freejoint/><geom type="box" size=".15 .15 .15"/></body>'


@unittest.skipUnless(RP.installed(), "RB-Y1 model not installed (scripts/get_robot.sh)")
class RobotIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.xml = rooms.write(tempfile.mkdtemp(), rooms.robot_room_xml().replace("</worldbody>", FAR + "</worldbody>"))

    def test_what_the_task_cannot_reach_sleeps_and_the_robot_never(self):
        w = RP.World(self.xml, (2.5, 1.0, 0.0))
        w.settle({"cup__0"})
        m, d = w.m, w.d
        tree = lambda b: m.body_treeid[b]
        for _ in range(10):
            mujoco.mj_step(m, d)
        self.assertGreaterEqual(d.tree_asleep[tree(w.objects["crate__1"])], 0)
        self.assertTrue(
            all(
                d.tree_asleep[tree(b)] < 0
                for b in w.robot | {w.objects["cup__0"], w.objects["table__0"]}
                if tree(b) >= 0
            )
        )

    def test_the_same_case_and_score_with_and_without_isolation(self):
        cases, units = {}, {}
        for iso in (True, False):
            c = calibrate.build(
                os.path.dirname(self.xml),
                procs=4,
                draws=2,
                need=2,
                jitter=1,
                log=lambda *a: None,
                robot_tasks=1,
                push_tasks=1,
                camera_seed=7,
                isolate_robot=iso,
            )
            r = evaluate.evaluate(dict(c, reference=self.xml), self.xml, procs=4, log=lambda *a: None)
            f = copy.deepcopy(c["functionality"])
            robot = [u for u in f["units"] if u["kind"].startswith("robot")]
            self.assertTrue(
                robot and all(u["params"].pop("isolate", False) == iso and u.pop("tol") for u in robot)
            )  # tol: see below
            for u in robot:
                u["params"].pop("nudge", None)  # nudge: see below
            cases[iso] = f["units"], f["excluded"]
            self.assertEqual(r["functionality"]["physics_hint"]["functionality"], 100.0)
            units[iso] = {
                u["id"]: [
                    (x["passed"], {k: v for k, v in x["details"].items() if k != "push_force_N"})
                    for x in r["units"][u["id"]]
                ]
                for u in robot
            }
        self.assertEqual(cases[True], cases[False])
        self.assertEqual(units[True], units[False])
        # Not compared: contact-force readings (the push force, the depth behind a robot unit's tol, informational under
        # the run rules). A far island asleep changes the solver's rounding; 1 mm jitter alone moves the push force from
        # 1.33 to 1.65 N. Nor the nudge shares (the share of five nudged reruns that pass, informational at 10 cm): the
        # same rounding flips one borderline 10 cm draw, 0.8 against 0.6. Whether a task is kept or refused still must
        # agree: the units and the excluded list are compared.
