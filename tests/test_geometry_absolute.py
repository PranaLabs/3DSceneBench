"""The geometry columns of the leaderboard: shares and medians in units a reader can check, from the same scope rows as
the scores."""

import unittest
from sbv import scope as SC, pipeline


def report():
    """True chair a, table b, lamp c; the copy pairs x to a and y to b, and also built a vase z and a lamp w beyond its
    gate."""
    case = dict(
        objects=[dict(slot=s) for s in "abc"],
        geometry=dict(
            inventory=dict(permitted_extras=[]),
            pose_size=[dict(slot=s, position_tol=0.1, size_bars=[1.25, 2.0]) for s in "abc"],
        ),
    )
    pose = [
        dict(slot="a", present=True, position_error=0.10, size_ratio=1.1, passed=True, passed_reference_bar=True),
        dict(slot="b", present=True, position_error=0.30, size_ratio=1.5, passed=False, passed_reference_bar=True),
        dict(slot="c", present=False, passed=False, passed_reference_bar=False),
    ]
    place = [
        dict(object="x", support=True, non_penetration=True, in_bounds=True, passed=True),
        dict(object="y", support=False, non_penetration=True, in_bounds=True, passed=False),
        dict(object="z", support=True, non_penetration=False, in_bounds=True, passed=False),
        dict(object="w", support=True, non_penetration=True, in_bounds=False, passed=False),
        dict(object="c", missing=True, support=False, non_penetration=False, in_bounds=False, passed=False),
    ]
    rep = dict(
        geometry=dict(
            G2=dict(
                per_object=dict(a=dict(fscore=0.9, chamfer=0.02), b=dict(fscore=0.7, chamfer=0.04)),
                pose_size=dict(objects=pose),
            ),
            G3=dict(relations=dict(relations=[]), placement=dict(objects=place), navigation=dict(routes=[])),
        )
    )
    cats = dict(x="chair", y="table", z="vase", w="lamp")
    return case, rep, cats


class Absolute(unittest.TestCase):
    def test_columns_count_units_paired_rows_and_built_objects(self):
        case, rep, cats = report()
        out = SC.readouts(case, rep, [], [], cats, dict(a="x", b="y"), {}, [], {}, 1.0, {"a", "b", "c"})
        a = out["geometry"]["absolute"]
        self.assertEqual((a["missing"], a["required"], a["extra"], a["built"]), (1, 3, 2, 4))
        self.assertAlmostEqual(a["missing_pct"], 100 / 3)
        self.assertEqual(a["extra_pct"], 50.0)
        self.assertAlmostEqual(a["position_error_median_cm"], 20.0)
        self.assertEqual(a["size_over_bar_pct"], 50.0)  # paired objects only
        self.assertAlmostEqual(a["surface_deviation_median_pct"], 3.0)
        self.assertEqual(
            (a["unsupported_pct"], a["penetrating_pct"], a["out_of_bounds_pct"]), (25.0, 25.0, 25.0)
        )  # the missing lamp is no row here
        self.assertAlmostEqual(
            out["geometry"]["G3"]["placement"]["score"], 1 / 5
        )  # the score still counts it as a failed row

    def test_reward_json_carries_the_columns_rounded(self):
        case, rep, cats = report()
        sc = SC.readouts(case, rep, [], [], cats, dict(a="x", b="y"), {}, [], {}, 1.0, {"a", "b", "c"})
        r = dict(rep, scopes=dict(full=sc, visible=sc), units={}, F1=dict(passed=True))
        s = pipeline.scores(r)
        self.assertEqual(
            (s["missing_pct"], s["extra_pct"], s["position_error_median_cm"], s["unsupported_pct"]),
            (33.3, 50.0, 20.0, 25.0),
        )
        self.assertNotIn("missing", s)  # counts stay in the report, the shares go to reward.json

    def test_the_reward_and_the_report_carry_the_verifier_version(self):
        import json, os, tempfile
        import sbv

        case, rep, cats = report()
        sc = SC.readouts(case, rep, [], [], cats, dict(a="x", b="y"), {}, [], {}, 1.0, {"a", "b", "c"})
        tmp = tempfile.mkdtemp(prefix="sbv_version_")
        path, out = os.path.join(tmp, "report.json"), os.path.join(tmp, "reward.json")
        pipeline.write(path, dict(rep, scopes=dict(full=sc, visible=sc), units={}, F1=dict(passed=True)))
        self.assertEqual(pipeline.reward_file(path, out), 0)
        reward = json.load(open(out))
        self.assertEqual(reward["verifier_version"], float(sbv.VERSION))  # a number: Harbor takes nothing else
        self.assertTrue(all(isinstance(v, (int, float)) for v in reward.values()))
        self.assertEqual(json.load(open(path))["verifier"]["version"], sbv.VERSION)


if __name__ == "__main__":
    unittest.main()
