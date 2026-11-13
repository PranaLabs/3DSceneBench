"""Geometric matching (sbv/common/case.py correspond_geometric): one Hungarian over all objects inside each reference
object's gate and a 3x size bar; the word only breaks ties; merges stay missing, splits pair their largest piece; a case
selects it with pairing.matcher = "geometric". Run from the repository root:  python -m unittest
tests.test_geometric_match -v"""

import json, os, tempfile, unittest
from sbv.common import case as K
from tests.test_place_pairing import scene, slot


def geo(slots, s):
    return K.correspond(slots, s, matcher="geometric")


class Geometric(unittest.TestCase):
    def test_the_word_never_vetoes(self):
        s = scene({"pillow__0": ((0, 0, 0.25), (0.5, 0.5, 0.5)), "zorbleflux__0": ((2.1, 0, 0.5), (0.1, 0.1, 0.1))})
        slots = [
            slot("large_plant__0", (0, 0, 0.25), (0.5, 0.5, 0.5)),
            slot("trinket__0", (2, 0, 0.5), (0.1, 0.1, 0.1)),
        ]
        self.assertEqual(geo(slots, s), {"large_plant__0": "pillow__0", "trinket__0": "zorbleflux__0"})
        self.assertEqual(K.correspond(slots, s), {"trinket__0": "zorbleflux__0"})  # the word matcher: other topic, lost

    def test_gate_and_size_bar(self):
        cup = [slot("cup__0", (0, 0, 0.8), (0.08, 0.08, 0.1))]  # gate max(0.3 m, 0.15 m) = 0.3 m
        for at, size, found in ((0.25, 0.1, True), (0.35, 0.1, False), (0.0, 0.5, False), (0.0, 0.04, True)):
            self.assertEqual(bool(geo(cup, scene({"bowl__0": ((at, 0, 0.8), (size, size, size))}))), found, (at, size))

    def test_size_decides_between_objects_in_one_place_not_the_word(self):
        s = scene(
            {"cup__0": ((0, 0, 0.5), (0.25, 0.25, 0.25)), "box__0": ((0, 0, 0.5), (0.1, 0.1, 0.1))}
        )  # words swapped
        slots = [slot("box__1", (0, 0, 0.5), (0.25, 0.25, 0.25)), slot("cup__1", (0, 0, 0.5), (0.1, 0.1, 0.1))]
        self.assertEqual(geo(slots, s), {"box__1": "cup__0", "cup__1": "box__0"})

    def test_another_word_needs_a_closer_place(self):
        counter = [slot("counter__0", (0, 0, 0.45), (2.0, 0.6, 0.9))]  # gate 2.27 m; another word: 1 m
        for word, at, found in (("counter", 1.5, True), ("fryer", 1.5, False), ("fryer", 0.9, True)):
            self.assertEqual(
                bool(geo(counter, scene({f"{word}__0": ((at, 0, 0.45), (0.6, 2.0, 0.9))}))), found, (word, at)
            )

    def test_size_within_3x_on_each_axis(self):
        tray = [slot("tray__0", (0, 0, 0.9), (0.58, 0.28, 0.05))]  # a utensil has the tray's diagonal
        self.assertEqual(geo(tray, scene({"utensil__0": ((0, 0, 0.9), (0.1, 0.1, 0.62))})), {})
        self.assertEqual(geo(tray, scene({"plate__0": ((0, 0, 0.9), (0.3, 0.5, 0.02))})), {"tray__0": "plate__0"})

    def test_merge_stays_missing_and_is_reported(self):
        s = scene({"fryer__0": ((0, 0, 0.5), (0.6, 0.6, 1.0))})
        slots = [slot("fryer__0", (0, 0, 0.5), (0.6, 0.6, 1.0))] + [
            slot(f"fries__{i}", (0.1 * i - 0.1, 0, 0.8), (0.05, 0.05, 0.1)) for i in range(3)
        ]
        c = geo(slots, s)
        self.assertEqual(c, {"fryer__0": "fryer__0"})
        self.assertEqual(K.match_report(slots, s, c)["merges"], {"fryer__0": ["fries__0", "fries__1", "fries__2"]})

    def test_split_pairs_the_largest_piece(self):
        legs = {
            f"desk__{i + 1}": ((x, y, 0.36), (0.04, 0.04, 0.72))
            for i, (x, y) in enumerate([(-0.55, -0.35), (-0.55, 0.35), (0.55, -0.35), (0.55, 0.35)])
        }
        s = scene({"desk__0": ((0, 0, 0.74), (1.2, 0.8, 0.04)), **legs})  # a top and four legs, one word
        slots = [slot("table__0", (0, 0, 0.38), (1.2, 0.8, 0.76))]
        c = geo(slots, s)
        self.assertEqual(c, {"table__0": "desk__0"})
        self.assertEqual(K.match_report(slots, s, c)["splits"]["table__0"][0], "desk__0")

    def test_contents_are_not_pieces(self):
        fries = {
            f"fries__{i}": ((0.07 * (i % 3) - 0.07, 0.05 * (i // 3) - 0.05, 0.05), (0.01, 0.01, 0.08)) for i in range(9)
        }
        self.assertEqual(geo([slot("food_box__0", (0, 0, 0.05), (0.2, 0.16, 0.1))], scene(fries)), {})  # the union fits

    def test_the_case_selects_the_matcher(self):
        s = scene({"pillow__0": ((0, 0, 0.25), (0.5, 0.5, 0.5))})
        o = slot("large_plant__0", (0, 0, 0.25), (0.5, 0.5, 0.5))
        p = os.path.join(tempfile.mkdtemp(prefix="sbv_geo_"), "case.json")
        json.dump(dict(schema=K.SCHEMA, objects=[o], pairing=dict(matcher="geometric")), open(p, "w"))
        c = K.load(p)
        self.assertEqual(K.correspond(c["objects"], s), {"large_plant__0": "pillow__0"})
        K.save({k: v for k, v in c.items() if not k.startswith("_")}, p)
        self.assertNotIn("_matcher", json.load(open(p))["objects"][0])  # load() adds it, save() drops it


if __name__ == "__main__":
    unittest.main()
