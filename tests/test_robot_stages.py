"""Robot tasks by stage: the share of a task's stages passed in order each checkpoint is one point, so an activity
scores its checkpoints passed over all its checkpoints; a task that never started scores 0."""

import unittest
from sbv.functionality.f4_robot.plan import stages, task_score, task_points

PICK = dict(kind="robot", params={})
PUSH = dict(kind="robot_push", params={})
SETTLED = dict(released=True, still=True, others_moved={}, on_time=True)


class Stages(unittest.TestCase):
    def test_pick_and_place(self):
        self.assertEqual(task_score(PICK, dict(success=True, started=True)), 1.0)
        self.assertEqual(
            task_score(PICK, dict(success=False, reason="target or goal object missing")), 0.0
        )  # never started
        self.assertAlmostEqual(
            task_score(PICK, dict(success=False, started=True, reason="no reachable base pose reaches the object")), 0.0
        )  # no free stage for starting
        goal_out = dict(
            success=False, started=True, reason="no reachable base pose reaches the goal", grasp=True, lift=True
        )
        self.assertAlmostEqual(task_score(PICK, goal_out), 3 / 6)  # picked it up
        dropped = dict(success=False, started=True, grasp=True, lift=True, in_goal=False, upright=True, **SETTLED)
        self.assertAlmostEqual(task_score(PICK, dropped), 4 / 6)

    def test_push(self):
        tipped = dict(success=False, started=True, push_force_N=0.4, at_mark=True, upright=False, **SETTLED)
        self.assertAlmostEqual(task_score(PUSH, tipped), 3 / 5)  # at the mark, then fell over
        self.assertAlmostEqual(
            task_score(PUSH, dict(success=False, started=True, reason="no reachable base pose reaches the push path")),
            0.0,
        )

    def test_an_activity_counts_every_checkpoint_once(self):  # one point per checkpoint
        unit = dict(
            kind="robot_activity", params=dict(steps=[dict(kind="press"), dict(kind="open"), dict(kind="pick_place")])
        )
        door = dict(kind="open", success=False, moved=0.66, amount=2.0, done=False, **SETTLED)  # a third of the way
        self.assertEqual([x for _, x in stages("open", door)], [True, True, False, False, True])  # 2 leading of 5
        run = dict(success=False, started=True, steps=[dict(kind="press", success=True), door])
        got, total = task_points(unit, run)  # 5 of 3 + 5 + 6 = 14, scaled to the 12-point cap
        self.assertEqual((round(got, 4), total), (round(5 * 12 / 14, 4), 12.0))  # the third step never ran
        self.assertAlmostEqual(task_score(unit, run), 5 / 14)
        self.assertAlmostEqual(
            task_score(
                unit,
                dict(
                    success=False,
                    started=True,
                    steps=[dict(kind="open", success=False, reason="no handle or knob stands proud")],
                ),
            ),
            0.0,
        )

    def test_a_task_is_worth_its_checkpoints(self):
        self.assertEqual(task_points(PICK, dict(success=True, started=True)), (6, 6))
        self.assertEqual(
            task_points(PUSH, dict(success=False, reason="target or goal object missing")), (0, 5)
        )  # never started
        self.assertEqual(task_points(dict(kind="robot_shelve", params={}), dict(success=True, started=True)), (6, 6))

    def test_an_activity_is_worth_at_most_12_points(
        self,
    ):
        piano = dict(kind="robot_activity", params=dict(steps=[dict(kind="press")] * 9))
        self.assertEqual(
            task_points(piano, dict(success=True, started=True, steps=[dict(kind="press", success=True)] * 9)),
            (12.0, 12.0),
        )
        three = [dict(kind="press", success=True)] * 3 + [dict(kind="press", success=False)]
        self.assertEqual(
            task_points(piano, dict(success=False, started=True, steps=three)), (9 * 12 / 27, 12.0)
        )  # 9 of 27 checkpoints
        patio = dict(kind="robot_activity", params=dict(steps=[dict(kind="open"), dict(kind="pick_place")]))
        self.assertEqual(
            task_points(
                patio,
                dict(
                    success=True,
                    started=True,
                    steps=[dict(kind="open", success=True), dict(kind="pick_place", success=True)],
                ),
            ),
            (11, 11),
        )

    def test_shelving_has_six_checkpoints(self):  # before, one
        shelve = dict(kind="robot_shelve", params={})
        at = lambda reason, **k: task_points(shelve, dict(success=False, started=True, reason=reason, **k))[0]
        self.assertEqual(
            [
                at("the book does not lie flat"),
                at("the spine overhangs the book below by only 1.2 cm"),
                at("no reachable base pose reaches the book's spine"),
                at("no reachable base pose reaches the bay"),
            ],
            [0, 1, 2, 3],
        )
        fell = dict(in_goal=True, upright=False, spine_out=True, **SETTLED)
        self.assertEqual(at("", **fell), 4)  # in the bay, but not upright
        self.assertEqual(at("", **dict(fell, upright=True, still=False)), 5)  # stands, but still moving


if __name__ == "__main__":
    unittest.main()
