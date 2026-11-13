"""Groups (sbv/groups.py): a single object standing beside a group stays a single object; a group word's leftovers are
one object in G1 (sbv/scope.py), and in G3 placement an extra is a row only when it fails."""

import math
import unittest
from types import SimpleNamespace
import numpy as np
from sbv import groups as GR, scope as SC


def room():
    """A tray of six vials and one more vial on the table beside it, inside the tray group's grown region."""
    objs = [
        dict(slot="table__0", category="table", center=[0, 0, 0.4], dims=[1.2, 0.8, 0.8]),
        dict(slot="tray__0", category="tray", center=[0, 0, 0.81], dims=[0.4, 0.3, 0.02]),
        dict(slot="vial__0", category="vial", center=[0.0, 0.15, 0.84], dims=[0.02, 0.02, 0.08]),
    ]
    objs += [
        dict(slot=f"vial__{i}", category="vial", center=[0.05 * i - 0.2, 0.0, 0.86], dims=[0.02, 0.02, 0.08])
        for i in range(1, 7)
    ]
    pairs = [("table__0", "tray__0"), ("table__0", "vial__0")] + [("tray__0", f"vial__{i}") for i in range(1, 7)]
    return dict(objects=objs, support_pairs=pairs)


def copy_of(case):
    """The reference room itself as the copy: same names, words and centres."""
    O = {o["slot"]: o for o in case["objects"]}
    return SimpleNamespace(
        objects=list(O),
        cat={n: o["category"] for n, o in O.items()},
        joints={n: [] for n in O},
        center=lambda n: np.asarray(O[n]["center"], float),
        aabb=lambda n: (
            np.subtract(O[n]["center"], np.divide(O[n]["dims"], 2)),
            np.add(O[n]["center"], np.divide(O[n]["dims"], 2)),
        ),
    )


class Groups(unittest.TestCase):
    def test_a_single_next_to_a_group_stays_single(self):
        case = room()
        G = GR.groups(case)
        S = copy_of(case)
        self.assertEqual([(g["id"], g["n"]) for g in G], [("group|vial|tray__0", 6)])
        self.assertTrue(GR.inside(case["objects"][2]["center"], G[0]))  # inside the grown region all the same
        corr, by = GR.pairing(case["objects"], S, G)
        self.assertEqual(sorted(by["group|vial|tray__0"]), [f"vial__{i}" for i in range(1, 7)])
        self.assertEqual(corr, {o["slot"]: o["slot"] for o in case["objects"]})
        self.assertEqual(GR.score(G, S, case["objects"], settle=False, by=by)["groups"][0]["count"], 1.0)

    def test_a_group_counts_one_plus_ln_of_its_members(self):
        case = dict(room(), geometry=dict(inventory={}, pose_size=[]))
        G = GR.groups(case)
        S = copy_of(
            dict(case, objects=[o for o in case["objects"] if o["slot"] not in {f"vial__{i}" for i in range(1, 7)}])
        )  # no vials in the tray
        corr, by = GR.pairing(case["objects"], S, G)
        grows = GR.score(G, S, case["objects"], settle=False, by=by)["groups"]
        rep = dict(
            geometry=dict(
                G2=dict(per_object={}, pose_size=dict(objects=[])),
                G3=dict(relations=dict(relations=[]), placement=dict(objects=[]), navigation=dict(routes=[])),
            )
        )
        g = SC.readouts(case, rep, G, grows, S.cat, corr, by, [], {}, 1.0, {o["slot"] for o in case["objects"]})[
            "geometry"
        ]
        self.assertAlmostEqual(
            g["G1"]["inventory"]["f1"], 2 * 3 / (3 + 3 + 1 + math.log(6))
        )  # the three singles found, the six vials, 2.79, not
        self.assertEqual(g["G1"]["inventory"]["true"], 4)  # counts stay counts: three singles and one group
        self.assertEqual(g["G2"]["pose_size"]["score"], 0.0)  # the group's pose row, the only one here

    def test_an_extra_is_a_placement_row_only_when_it_fails(self):
        """300 copy objects (placement rows and words only) that stand for no reference object. Named vial, a group
        word, they are one extra in G1 and one unit in the reported shares. In placement an extra is a row only when it
        fails, under any word: cubes that pass cannot lift placement, and a failing one counts once whatever it is
        called. On the slope (a row between 0 and 1) an extra costs its shortfall and adds nothing."""
        case = dict(room(), geometry=dict(inventory={}, pose_size=[]))
        G = GR.groups(case)
        S = copy_of(case)
        slots = list(S.objects)
        corr, by = GR.pairing(case["objects"], S, G)
        grows = GR.score(G, S, case["objects"], settle=False, by=by)["groups"]
        pad = [f"vial__p{i}" for i in range(300)]

        def readout(fail, word="vial", partial=None):  # partial: a failing row's score on placement()'s slope
            cats = dict(S.cat, **{n: word for n in pad})
            rows = [
                dict(object=n, passed=n not in fail, support=n not in fail, non_penetration=True, in_bounds=True)
                for n in slots + pad
            ]
            for r in rows:
                if partial is not None and not r["passed"]:
                    r["score"] = partial.get(r["object"], 0.0)
            rep = dict(
                geometry=dict(
                    G2=dict(per_object={}, pose_size=dict(objects=[])),
                    G3=dict(relations=dict(relations=[]), placement=dict(objects=rows), navigation=dict(routes=[])),
                )
            )
            return SC.readouts(case, rep, G, grows, cats, corr, by, [], {}, 1.0, set(slots))["geometry"]

        self.assertEqual(readout(set())["G1"]["inventory"]["copy"], 5)  # table, tray, vial__0, the group, the cubes
        for word in ("vial", "book"):  # a group word, and a word with no group
            self.assertEqual(
                readout({"tray__0"}, word)["G3"]["placement"]["score"], 2 / 3
            )  # table, tray, vial__0; was 302 / 303
        self.assertEqual(readout({"vial__0"})["G3"]["placement"]["score"], 2 / 3)  # the single vial keeps its own row
        bad = {"tray__0"} | {f"vial__p{i}" for i in range(40)}  # 40 failing cubes: 40 failing rows, under any word
        self.assertEqual({readout(bad, w)["G3"]["placement"]["score"] for w in ("vial", "book")}, {2 / 43})
        self.assertEqual(
            readout({"tray__0", "vial__p7"})["absolute"]["unsupported_pct"], 50.0
        )  # leftovers: one unit in the shares
        floating = {"tray__0"} | set(pad)  # 300 cubes 1.01 cm up: each row 0.995 on the slope
        score = readout(floating, "book", {n: 0.995 for n in pad})["G3"]["placement"]["score"]
        self.assertAlmostEqual(score, 2 / (3 + 300 * 0.005))  # each costs its 0.005 shortfall, and adds nothing
        self.assertLess(score, 2 / 3)  # (as rows of their own they would lift it to 0.99)


if __name__ == "__main__":
    unittest.main()
