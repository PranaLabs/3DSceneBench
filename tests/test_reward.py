"""One reward: every part on 0 to 100, equal weights; four levels, the room's robot tasks in interaction; non-rigid
units in their parts."""

import unittest
from sbv import reward as R
from sbv.functionality.aggregate import levels
from sbv.functionality.units import level


def geometry(v=1.0):
    return dict(
        G1=dict(inventory=dict(f1=v)),
        G2=dict(fscore=v, pose_size=dict(score=v)),
        G3=dict(relations=dict(score=v), placement=dict(score=v), navigation=dict(score=v)),
    )


class Reward(unittest.TestCase):
    def test_appearance_scores_turn_to_0_to_100(self):
        self.assertEqual(
            R.appearance(dict(siglip2=1.0, dinov3=0.0, lpips_vgg=0.25)),
            dict(siglip2_100=100.0, dinov3_100=50.0, lpips_100=75.0),
        )
        self.assertEqual(R.appearance(dict(lpips_vgg=1.7)), dict(lpips_100=0.0))

    def test_the_rooms_floor_is_0_and_the_true_room_100(self):
        floor = dict(siglip2=0.9, dinov3=0.6, lpips_vgg=0.25)
        self.assertEqual(
            R.appearance(dict(siglip2=1.0, dinov3=1.0, lpips_vgg=0.0), floor),
            dict(siglip2_100=100.0, dinov3_100=100.0, lpips_100=100.0),
        )
        self.assertEqual(
            R.appearance(floor, floor), dict(siglip2_100=0.0, dinov3_100=0.0, lpips_100=0.0)
        )  # an empty room
        self.assertEqual(
            R.appearance(dict(siglip2=0.95, dinov3=0.8, lpips_vgg=0.125), floor),
            dict(siglip2_100=50.0, dinov3_100=50.0, lpips_100=50.0),
        )
        self.assertEqual(
            R.appearance(dict(siglip2=0.7, dinov3=0.2, lpips_vgg=0.4), floor),
            dict(siglip2_100=0.0, dinov3_100=0.0, lpips_100=0.0),
        )  # worse than empty: 0
        self.assertEqual(
            R.appearance(dict(siglip2=0.995), dict(siglip2=0.99)), dict(siglip2_100=10.0)
        )  # a floor near 1: the span stays 0.05
        self.assertEqual(
            R.appearance(dict(siglip2=0.995), dict(siglip2=None)), dict(siglip2_100=99.8)
        )  # no floor for a metric: as before
        from sbv import pipeline

        rep = dict(
            geometry=geometry(1.0),
            levels={},
            functionality=dict(
                physics_hint=dict(functionality=None, layers={}), input_only=dict(functionality=None, layers={})
            ),
            units={},
            F1=dict(passed=True),
            appearance=dict(mean=floor, floor=floor),
        )
        self.assertEqual((pipeline.scores(rep)["reward_appearance"], pipeline.scores(rep)["siglip2_100"]), (0.0, 0.0))

    def test_units_go_to_their_level_and_every_robot_task_is_interaction(self):
        self.assertEqual(level(dict(kind="tilt")), "response")
        self.assertEqual(level(dict(kind="edge")), "balance")
        self.assertEqual(level(dict(kind="robot")), "interaction")
        self.assertEqual(level(dict(kind="robot", named=True)), "interaction")
        self.assertEqual(
            [level(dict(kind=k, named=True)) for k in ("robot_shelve", "robot_activity")], ["interaction"] * 2
        )
        self.assertNotIn("tasks", R.PARTS)

    def test_equal_weights_over_the_parts_the_room_has(self):
        units = [
            dict(id="push|a", kind="push"),
            dict(id="push|b", kind="push", hint_only=True),
            dict(id="joint|c|j", kind="joint"),
        ]
        results = {
            "push|a": [dict(passed=dict(physics_hint=True, input_only=True))],
            "push|b": [dict(passed=dict(physics_hint=False, input_only=False))],
            "joint|c|j": [dict(passed=dict(physics_hint=False, input_only=False))],
        }
        rep = dict(
            geometry=geometry(0.5),
            appearance=dict(mean=dict(siglip2=1.0, dinov3=1.0, lpips_vgg=0.0)),
            levels=levels(1.0, units, results),
        )
        rw, p = R.reward(
            rep,
            nonrigid_rows=[
                dict(id="contain.tea", kind="contain", passed=True),
                dict(id="S1.tea", kind="fluid", passed=False),
            ],
        )
        # input only: push|b is hint-only and left out; response = push|a (1) and the spill (0); balance = containment
        # (1)
        self.assertEqual(
            p, dict(appearance=100.0, geometry=50.0, balance=100.0, response=50.0, affordance=0.0, interaction=None)
        )
        self.assertEqual(rw, 66.67)  # A 100, G 50, F (100 + 50 + 0) / 3 = 50: (100 + 50 + 50) / 3
        self.assertEqual(R.reward(rep, setting="physics_hint")[1]["response"], 50.0)  # push|a 1, push|b 0

    def test_lpips_is_reported_but_not_scored(self):
        rep = dict(
            geometry=geometry(1.0), levels={}, appearance=dict(mean=dict(siglip2=1.0, dinov3=0.0, lpips_vgg=1.0))
        )
        self.assertEqual(R.parts(rep)["appearance"], 75.0)  # SigLIP-2 100, DINOv3 50; LPIPS 0 left out
        self.assertEqual(R.appearance(rep["appearance"]["mean"])["lpips_100"], 0.0)

    def test_navigation_is_reported_but_not_scored(self):
        g = geometry(1.0)
        g["G3"]["navigation"]["score"] = 0.0  # no goal reachable: reported, geometry unchanged
        self.assertEqual(R.geometry(g), 100.0)
        g = geometry(0.5)
        g["G3"]["navigation"]["score"] = 1.0
        self.assertEqual(R.geometry(g), 50.0)

    def test_a_soft_unit_that_cannot_decide_does_not_count_and_a_graded_one_counts_its_score(self):
        rows = [
            dict(id="S1.tea", kind="fluid", passed=True, score=0.75, counted=True),
            dict(id="S1.coat", kind="thin_shell", passed=False, score=None, counted=False),
            dict(id="S1.old", kind="soft_body", passed=True, decidable=False),
        ]
        p = R.parts(dict(geometry=geometry(1.0), levels={}), nonrigid_rows=rows)
        self.assertEqual(p["response"], 75.0)

    def test_soft_rows_count_in_their_parts(self):
        rows = [
            dict(id="S0.tea", kind="amount", passed=True),
            dict(id="S0.pillow", kind="size", passed=False),
            dict(id="contain.tea", kind="contain", passed=False),
            dict(id="S1.tea", kind="fluid", passed=True),
            dict(id="S1.pillow", kind="soft_body", passed=True),
            dict(id="S1.coat", kind="thin_shell", passed=False),
        ]
        self.assertEqual(
            [R.nonrigid_part(r) for r in rows], ["geometry", "geometry", "balance", "response", "response", "response"]
        )
        self.assertEqual(
            [R.nonrigid_part(dict(id=i)) for i in ("S0.tea", "S1.tea")], ["geometry", "response"]
        )  # rows from before kinds were written
        # pose-and-size: 2 rigid objects, 1 passed, plus the amount (passed) and the size (failed): 2 of 4; the other
        # four scored parts 1
        g = geometry(1.0)
        g["G2"]["pose_size"] = dict(score=0.5, objects=[dict(passed=True), dict(passed=False)])
        p = R.parts(dict(geometry=g, levels={}), nonrigid_rows=rows)
        self.assertEqual(
            (p["geometry"], p["balance"], p["response"]), (round(100 * (4 + 0.5) / 5, 1), 0.0, round(100 * 2 / 3, 1))
        )
        self.assertEqual(
            R.geometry(geometry(1.0), [1.0, 0.0]), 95.0
        )  # no per-object rows: the share 1 averaged with theirs 0.5 gives 0.75, then over five

    def test_a_level_is_its_points_earned_over_its_points_possible(self):
        rep = dict(
            geometry=geometry(1.0),
            levels=dict(input_only=dict(interaction=[0.5, 1.0])),
            level_points=dict(
                input_only=dict(interaction=[[3.0, 6.0], [5.0, 5.0]], affordance=[[2.74, 5.48], [1.0, 1.0]])
            ),
        )
        p = R.parts(rep)
        self.assertEqual((p["interaction"], p["affordance"]), (round(100 * 8 / 11, 1), round(100 * 3.74 / 6.48, 1)))
        self.assertEqual(
            R.parts(dict(rep, level_points=None))["interaction"], 75.0
        )  # a report from before points: one point per unit
        nonrigid = [
            dict(id="S1.tea", kind="fluid", passed=True, score=0.75, counted=True),
            dict(id="S1.coat", kind="thin_shell", passed=False, score=None, counted=False),
        ]
        self.assertEqual(
            R.parts(dict(geometry=geometry(1.0), levels={}), nonrigid_rows=nonrigid)["response"], 75.0
        )  # graded and not-counted non-rigid rows

    def test_the_reference_bar_reads_its_own_levels_and_the_f_score_is_the_levels_mean(self):
        units = [dict(id="push|a", kind="push"), dict(id="joint|c|j", kind="joint")]
        results = {
            "push|a": [
                dict(
                    passed=dict(physics_hint=True, input_only=True),
                    passed_reference_bar=dict(physics_hint=False, input_only=False),
                )
            ],
            "joint|c|j": [dict(passed=dict(physics_hint=True, input_only=True))],
        }  # a unit without bars counts the same at both
        rep = dict(
            geometry=geometry(1.0),
            levels=levels(1.0, units, results),
            levels_reference_bar=levels(1.0, units, results, bar=1),
        )
        self.assertEqual(
            (R.parts(rep)["response"], R.parts(rep, bar=1)["response"], R.parts(rep, bar=1)["affordance"]),
            (100.0, 0.0, 100.0),
        )
        self.assertEqual((R.functionality(R.parts(rep)), R.functionality(R.parts(rep, bar=1))), (100.0, 50.0))
        from sbv import pipeline

        s = pipeline.scores(dict(rep, units=results, F1=dict(passed=True)))
        self.assertEqual(
            (
                s["parse"],
                s["sim_ready"],
                s["reward_functionality"],
                s["reward_functionality_reference_bar"],
                s["inventory"],
            ),
            (1.0, 1.0, 100.0, 50.0, 100.0),
        )
        self.assertFalse({"F1", "F2", "functionality_input", "G1", "G2_pose", "nonrigid"} & set(s))
        s0 = pipeline.scores(dict(geometry=None, levels={}, units={}, F1=dict(passed=False)))
        self.assertEqual((s0["parse"], s0["sim_ready"], s0["reward"], s0["reward_functionality"]), (0.0, 0.0, 0.0, 0.0))

    def test_a_copy_that_does_not_load_scores_0(self):
        self.assertEqual(R.reward(dict(geometry=None, levels={}))[0], 0.0)
        units = [dict(id="push|a", kind="push")]
        results = {"push|a": [dict(passed=dict(physics_hint=True, input_only=True))]}
        self.assertEqual(levels(0.0, units, results)["input_only"]["response"], [0.0])  # the gate failed: every unit 0

    def test_a_copy_that_cannot_render_scores_0_on_appearance_and_a_grader_failure_leaves_it_out(self):
        rep = dict(geometry=geometry(1.0), levels={})
        self.assertIsNone(R.parts(dict(rep, appearance=dict(status="metrics failed: no weights")))["appearance"])
        self.assertEqual(
            R.parts(dict(rep, appearance=dict(status="render failed: bad mesh", copy_render_failed=True)))[
                "appearance"
            ],
            0.0,
        )
        self.assertEqual(
            R.reward(dict(rep, appearance=dict(status="render failed", copy_render_failed=True)))[0], 50.0
        )  # (0 + 100) / 2

    def test_the_reward_reads_a_scope_and_a_report_from_before_scopes_its_own_numbers(self):
        rep = dict(
            geometry=geometry(0.5),
            levels={},
            scopes=dict(full=dict(geometry=geometry(1.0), levels={}), visible=dict(geometry=geometry(0.2), levels={})),
        )
        self.assertEqual([R.reward(rep, scope=k)[0] for k in ("full", "visible")], [100.0, 20.0])
        self.assertEqual(R.reward(dict(rep, scopes=None))[0], 50.0)


if __name__ == "__main__":
    unittest.main()
