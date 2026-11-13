"""Non-rigid units in G2 pose-and-size (sbv/reward.py pose_share()): each fluid amount and soft body size unit is one
more object of weight 1 beside the rigid objects, which carry their geometry weights (sbv/scope.py weights()), in one
share: (sum of w_i * pass_i + sum of nonrigid) / (sum of w_i + number of non-rigid units)."""

import unittest
from sbv import reward as R, scope as SC

NONRIGID = [dict(id="S0.tea", kind="amount", passed=True), dict(id="S0.cushion", kind="size", passed=False)]
POOLED = (15 + 2 + 1) / (
    20 + 2 + 2
)  # 15 of 20 weight-1 objects pass, the weight-2 one passes; one non-rigid unit of two passes


def geometry(pose_size, v=1.0, inventory=None):
    return dict(
        G1=dict(inventory=dict(f1=v, **(inventory or {}))),
        G2=dict(fscore=v, pose_size=pose_size),
        G3=dict(relations=dict(score=v), placement=dict(score=v), navigation=dict(score=v)),
    )


def rows():
    """20 rigid objects of weight 1, 15 passing, and one of weight 2 (a robot task moves it), passing."""
    return [dict(slot=f"o{i}__0", passed=i < 15, weight=1.0) for i in range(20)] + [
        dict(slot="can__0", passed=True, weight=2.0)
    ]


class PoseSoftWeight(unittest.TestCase):
    def test_weighted_rows_and_soft_units_pool_into_one_share(self):
        g = geometry(dict(score=None, objects=rows()))
        self.assertAlmostEqual(R.pose_share(g, [1.0, 0.0]), POOLED)  # 18 / 24 = 0.75
        self.assertEqual(R.geometry(g, [1.0, 0.0]), round(100 * (4 + POOLED) / 5, 1))
        self.assertEqual(R.parts(dict(geometry=g, levels={}), nonrigid_rows=NONRIGID)["geometry"], 95.0)
        unweighted = [
            {k: v for k, v in r.items() if k != "weight"} for r in rows()
        ]  # rows without a weight count 1 each
        self.assertAlmostEqual(
            R.pose_share(geometry(dict(score=None, objects=unweighted)), [1.0, 0.0]), (16 + 1) / (21 + 2)
        )

    def test_the_scope_path_pools_its_weight_sums(self):
        ps = dict(score=17 / 22, score_reference_bar=17 / 22, weight=22.0, passed_weight=17.0)
        rep = dict(
            geometry=geometry(dict(score=0.0)),
            levels={},
            scopes=dict(full=dict(geometry=geometry(ps), levels={}), visible=dict(geometry=geometry(ps), levels={})),
        )
        self.assertAlmostEqual(R.pose_share(rep["scopes"]["full"]["geometry"], [1.0, 0.0]), POOLED)
        for scope in ("full", "visible"):
            for bar in (0, 1):
                self.assertEqual(R.parts(rep, NONRIGID, scope=scope, bar=bar)["geometry"], 95.0)
        self.assertEqual(
            R.parts(rep, scope="full")["geometry"], round(100 * (4 + 17 / 22) / 5, 1)
        )  # no non-rigid units: the rigid share

    def test_scope_readouts_write_the_weight_sums(self):
        slots = [f"o{i}__0" for i in range(20)] + ["can__0"]
        case = dict(
            objects=[dict(slot=s) for s in slots],
            functionality=dict(units=[dict(id="robot|can__0|o0__0", kind="robot", slots=["can__0"])]),
            geometry=dict(
                inventory=dict(permitted_extras=[]),
                pose_size=[dict(slot=s, position_tol=0.1, size_bars=[1.25, 2.0]) for s in slots],
            ),
        )
        ok = lambda s: s == "can__0" or int(s[1:-3]) < 15
        pose = [
            dict(slot=s, present=True, position_error=0.0, size_ratio=1.0, passed=ok(s), passed_reference_bar=ok(s))
            for s in slots
        ]
        rep = dict(
            geometry=dict(
                G2=dict(per_object={s: dict(fscore=1.0, chamfer=0.0) for s in slots}, pose_size=dict(objects=pose)),
                G3=dict(relations=dict(relations=[]), placement=dict(objects=[]), navigation=dict(routes=[])),
            )
        )
        corr = {s: s for s in slots}
        cats = {s: s[:-3] for s in slots}
        ps = SC.readouts(case, rep, [], [], cats, corr, {}, [], {}, 1.0, set(slots))["geometry"]["G2"]["pose_size"]
        self.assertEqual((ps["weight"], ps["passed_weight"]), (22.0, 17.0))
        self.assertAlmostEqual(ps["score"], 17 / 22)
        self.assertAlmostEqual(R.pose_share(geometry(ps), [1.0, 0.0]), POOLED)

    def test_a_scope_from_before_the_weight_sums_reads_g1s_true_weight(self):
        g = geometry(dict(score=17 / 22), inventory=dict(true=21, true_weight=22.0))
        self.assertAlmostEqual(R.pose_share(g, [1.0, 0.0]), POOLED)
        g = geometry(dict(score=15 / 20), inventory=dict(true=20))  # before weights: the count
        self.assertAlmostEqual(R.pose_share(g, [1.0, 0.0]), (15 + 1) / (20 + 2))

    def test_an_old_report_without_any_size_still_scores(self):
        g = geometry(dict(score=1.0))
        self.assertAlmostEqual(
            R.pose_share(g, [1.0, 0.0]), 0.75
        )  # the rigid share 1 and the non-rigid share 0.5, averaged
        self.assertEqual(R.geometry(g, [1.0, 0.0]), 95.0)
        self.assertEqual(
            R.pose_share(geometry(dict(score=None)), [1.0, 0.0]), 0.5
        )  # no rigid object: the non-rigid units alone
        self.assertEqual(R.pose_share(g), 1.0)  # no non-rigid unit: the rigid share


if __name__ == "__main__":
    unittest.main()
