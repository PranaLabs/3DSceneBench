"""Geometry weights by object (sbv/scope.py weights(): an object weighs as much as scoring depends on it, 2 for the
objects a robot task moves or places on, 1 + ln(n) for n moving parts, else 1, the largest that applies; a group weighs
once, as its heaviest member; every geometry part is a weighted mean, and the absolute columns stay counts."""

import math
import unittest
from sbv import groups as GR, scope as SC
from tests.test_groups import room, copy_of

L4, L2 = 1 + math.log(4), 1 + math.log(2)


def units():
    """A pick-and-place of the can onto the shelf with the lamp nearby (near adds no weight), a cabinet with four
    drawers, a shelf with two doors, and a push on the book (a unit that adds no weight)."""
    return [
        dict(id="robot|can__0|shelf__0", kind="robot", slots=["can__0", "shelf__0"], params=dict(near=["lamp__0"])),
        *[dict(id=f"joint|cabinet__0|drawer{i}", kind="joint", slots=["cabinet__0"]) for i in range(4)],
        *[dict(id=f"joint|shelf__0|door{i}", kind="joint", slots=["shelf__0"]) for i in range(2)],
        dict(id="push|book__0", kind="push", slots=["book__0"]),
    ]


class ObjectWeights(unittest.TestCase):
    def test_weights_follow_what_scoring_needs(self):
        case = dict(
            objects=[dict(slot=s) for s in ("can__0", "shelf__0", "lamp__0", "cabinet__0", "book__0")],
            functionality=dict(units=units()),
        )
        w = SC.weights(case)
        self.assertEqual(
            {s: round(v, 6) for s, v in w.items()},
            dict(can__0=2.0, shelf__0=2.0, lamp__0=1.0, cabinet__0=round(L4, 6), book__0=1.0),
        )  # the shelf: 2 beats 1.69; near adds nothing
        self.assertEqual(SC.weights(dict(objects=case["objects"])), dict.fromkeys(w, 1.0))  # a case with no units

    def test_every_geometry_part_is_a_weighted_mean(self):
        slots = (
            "can__0",
            "cabinet__0",
            "book__0",
            "lamp__0",
            "shelf__0",
        )  # weights 2, 2.39, 1, 1, 2; the copy pairs x, y, z
        case = dict(
            objects=[dict(slot=s) for s in slots],
            functionality=dict(units=units()),  # to the first three and adds a vase v
            geometry=dict(
                inventory=dict(permitted_extras=[]),
                pose_size=[dict(slot=s, position_tol=0.1, size_bars=[1.25, 2.0]) for s in slots],
            ),
        )
        corr, cats = dict(can__0="x", cabinet__0="y", book__0="z"), dict(x="can", y="cabinet", z="book", v="vase")
        pose = [
            dict(
                slot="can__0", present=True, position_error=0.05, size_ratio=1.1, passed=True, passed_reference_bar=True
            ),
            dict(
                slot="cabinet__0",
                present=True,
                position_error=0.40,
                size_ratio=1.1,
                passed=False,
                passed_reference_bar=False,
            ),
            dict(
                slot="book__0",
                present=True,
                position_error=0.02,
                size_ratio=1.0,
                passed=True,
                passed_reference_bar=True,
            ),
        ] + [dict(slot=s, present=False, passed=False, passed_reference_bar=False) for s in ("lamp__0", "shelf__0")]
        rel = [
            dict(a="can__0", b="shelf__0", passed=True),
            dict(a="book__0", b="cabinet__0", passed=False),
            dict(a="cabinet__0", passed=True),
        ]
        place = [
            dict(object="x", support=True, non_penetration=True, in_bounds=True, passed=True, score=1.0),
            dict(object="y", support=False, non_penetration=True, in_bounds=True, passed=False, score=0.5),
            dict(object="z", support=False, non_penetration=True, in_bounds=True, passed=False, score=0.0),
            dict(object="v", support=False, non_penetration=True, in_bounds=True, passed=False, score=0.25),
        ] + [
            dict(object=s, missing=True, support=False, non_penetration=False, in_bounds=False, passed=False, score=0.0)
            for s in ("lamp__0", "shelf__0")
        ]
        rep = dict(
            geometry=dict(
                G2=dict(
                    per_object=dict(
                        can__0=dict(fscore=0.9, chamfer=0.01),
                        cabinet__0=dict(fscore=0.5, chamfer=0.03),
                        book__0=dict(fscore=0.2, chamfer=0.05),
                    ),
                    pose_size=dict(objects=pose),
                ),
                G3=dict(
                    relations=dict(relations=rel),
                    placement=dict(objects=place),
                    navigation=dict(routes=[dict(goal="can__0", passed=False), dict(goal="cabinet__0", passed=True)]),
                ),
            )
        )
        out = SC.readouts(case, rep, [], [], cats, corr, {}, [], {}, 1.0, set(slots))
        g = out["geometry"]
        M, T = 2 + L4 + 1, 2 + L4 + 1 + 1 + 2
        self.assertAlmostEqual(g["G1"]["inventory"]["f1"], 2 * M / (M + 1 + T))  # was 2 x 3 / (4 + 5) = 0.667
        self.assertEqual([g["G1"]["inventory"][k] for k in ("paired", "copy", "true")], [3, 4, 5])  # counts stay counts
        self.assertAlmostEqual(g["G2"]["fscore"], (2 * 0.9 + L4 * 0.5 + 0.2) / M)
        self.assertAlmostEqual(g["G2"]["pose_size"]["score"], (2 + 1) / T)
        self.assertAlmostEqual(g["G3"]["relations"]["score"], (2 + L4) / (2 + 1 + L4))  # by each relation's object a
        self.assertAlmostEqual(
            g["G3"]["placement"]["score"], (2 + 0.5 * L4) / (T + 0.75)
        )  # the vase costs its shortfall, 0.75
        self.assertAlmostEqual(g["G3"]["navigation"]["score"], L4 / (2 + L4))  # by each route's goal
        a = g["absolute"]
        self.assertEqual((a["missing"], a["required"], a["extra"], a["built"], a["missing_pct"]), (2, 5, 1, 4, 40.0))
        self.assertEqual(out["C"], 3 / 5)

    def test_a_group_weighs_once_the_larger_of_its_points_and_its_heaviest_member(self):
        robot = dict(
            id="robot|vial__2|table__0", kind="robot", slots=["vial__2"], params=dict(near=["tray__0"])
        )  # the robot moves a vial in the tray
        case = dict(room(), functionality=dict(units=[robot]))
        G = GR.groups(case)
        case["geometry"] = dict(
            inventory={},
            pose_size=[dict(slot=o["slot"], position_tol=0.1, size_bars=[1.25, 2.0]) for o in case["objects"]],
        )
        S = copy_of(
            dict(case, objects=[o for o in case["objects"] if o["slot"] not in {f"vial__{i}" for i in range(1, 7)}])
        )  # no vials in the tray
        corr, by = GR.pairing(case["objects"], S, G)
        grows = GR.score(G, S, case["objects"], settle=False, by=by)["groups"]
        pose = [
            dict(slot=s, present=True, position_error=0.0, size_ratio=1.0, passed=True, passed_reference_bar=True)
            for s in ("table__0", "tray__0", "vial__0")
        ]
        rep = dict(
            geometry=dict(
                G2=dict(per_object={}, pose_size=dict(objects=pose)),
                G3=dict(relations=dict(relations=[]), placement=dict(objects=[]), navigation=dict(routes=[])),
            )
        )
        g = SC.readouts(case, rep, G, grows, S.cat, corr, by, [], {}, 1.0, {o["slot"] for o in case["objects"]})[
            "geometry"
        ]
        pts = 1 + math.log(6)  # the group's points: 2.79
        self.assertAlmostEqual(
            g["G1"]["inventory"]["f1"], 2 * 3 / (3 + 0 + 3 + pts)
        )  # table, tray (only near), vial__0 1 each; the group,
        self.assertAlmostEqual(
            g["G2"]["pose_size"]["score"], 3 / (3 + pts)
        )  # missing, weighs once: max(2.79, its robot vial's 2)
        case["functionality"]["units"] = []
        g = SC.readouts(case, rep, G, grows, S.cat, corr, by, [], {}, 1.0, {o["slot"] for o in case["objects"]})[
            "geometry"
        ]
        self.assertAlmostEqual(
            g["G1"]["inventory"]["f1"], 2 * 3 / (3 + 0 + 3 + pts)
        )  # every weight 1: main's group points
        case["functionality"]["units"] = [
            dict(id=f"joint|vial__2|j{i}", kind="joint", slots=["vial__2"]) for i in range(8)
        ]
        g = SC.readouts(case, rep, G, grows, S.cat, corr, by, [], {}, 1.0, {o["slot"] for o in case["objects"]})[
            "geometry"
        ]
        self.assertAlmostEqual(
            g["G1"]["inventory"]["f1"], 2 * 3 / (3 + 0 + 3 + 1 + math.log(8))
        )  # a member heavier than the points:
        self.assertAlmostEqual(g["G2"]["pose_size"]["score"], 3 / (3 + 1 + math.log(8)))  # 8 moving parts, 3.08, wins


if __name__ == "__main__":
    unittest.main()
