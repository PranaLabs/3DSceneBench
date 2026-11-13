"""Pairing by place after pairing by word (sbv/common/case.py correspond()): a reference object still unpaired takes a
leftover copy object of any word whose centre is within max(the true box diagonal, 0.15 m) and, seen from above, within
max(the true footprint diagonal, 0.15 m), whose box diagonal is within 3x, and whose word is under the same topic of the
word list;
the inventory counts the pairs, and the word is reported apart. Run from the repository root:
    python -m unittest tests.test_place_pairing -v
"""

import os, sys, unittest
from types import SimpleNamespace
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from sbv.common import case as K  # noqa: E402
from sbv.geometry import g1_inventory as Inv  # noqa: E402


def scene(objs):
    """name -> (centre, size): a stand-in Scene with boxes only."""
    c = {n: np.array(v[0], float) for n, v in objs.items()}
    h = {n: np.array(v[1], float) / 2 for n, v in objs.items()}
    return SimpleNamespace(
        objects=objs,
        cat={n: n.split("__")[0] for n in objs},
        center=lambda n: c[n],
        aabb=lambda n: (c[n] - h[n], c[n] + h[n]),
        joints={n: [] for n in objs},
    )


def slot(name, centre, dims, role="free"):
    return dict(slot=name, category=name.split("__")[0], center=centre, dims=dims, role=role)


class PlaceStep(unittest.TestCase):
    def test_another_word_in_the_right_place_and_size_is_found(self):
        s = scene(
            {
                "figurine__0": ((1.05, 0, 0.5), (0.1, 0.1, 0.12)),
                "book__0": ((3.0, 0, 0.5), (0.2, 0.3, 0.05)),
                "cup__0": ((5.0, 0, 0.8), (0.08, 0.08, 0.1)),
            }
        )
        slots = [
            slot("trinket__0", (1.0, 0, 0.5), (0.1, 0.1, 0.1)),
            slot("book__0", (3.1, 0, 0.5), (0.2, 0.3, 0.05)),
            slot("cup__0", (0.0, 0, 0.8), (0.08, 0.08, 0.1)),
        ]
        # The copy's cup is 5 m from the true one: outside its gate, so missing; the figurine is the trinket by place
        self.assertEqual(K.correspond(slots, s), {"book__0": "book__0", "trinket__0": "figurine__0"})

    def test_the_distance_floor_and_the_size_limit(self):
        small = slot("trinket__0", (0, 0, 0.5), (0.03, 0.03, 0.03))  # box diagonal 5 cm: the 15 cm floor applies
        for at, size, found in (
            ((0.12, 0, 0.5), (0.03, 0.03, 0.03), True),
            ((0.2, 0, 0.5), (0.03, 0.03, 0.03), False),
            ((0.05, 0, 0.5), (0.08, 0.08, 0.08), True),
            ((0.05, 0, 0.5), (0.1, 0.1, 0.1), False),
            ((0.05, 0, 0.5), (0.011, 0.011, 0.011), True),
            ((0.05, 0, 0.5), (0.009, 0.009, 0.009), False),
        ):
            got = K.correspond([small], scene({"figurine__0": (at, size)}))
            self.assertEqual(bool(got), found, (at, size))  # sizes 2.7x and 0.37x in; 3.3x and 0.3x out

    def test_a_tall_object_reaches_only_its_footprint_sideways(self):
        lamp = slot("floor_lamp__1", (0, 0, 0.6), (0.3, 0.3, 1.2))  # box diagonal 1.27 m, footprint diagonal 0.42 m
        beside = scene({"table_lamp__1": ((1.1, 0, 0.9), (0.28, 0.28, 0.28))})  # 1.14 m away, size 0.38x: in by both
        self.assertEqual(K.correspond([lamp], beside), {})  # 3D rules, out by the footprint
        above = scene({"lamp__0": ((0.3, 0, 1.0), (0.3, 0.3, 0.9))})
        self.assertEqual(K.correspond([lamp], above), {"floor_lamp__1": "lamp__0"})

    def test_only_words_of_one_topic_pair_by_place(self):
        # each pair is in the right place and size; the pillow is bedding and the book office, the trinkets decor
        s = scene(
            {
                "pillow__2": ((0, 0, 0.25), (0.5, 0.5, 0.3)),
                "book__8": ((2.1, 0, 0.5), (0.2, 0.24, 0.03)),
                "figurine__0": ((4.05, 0, 0.5), (0.1, 0.1, 0.12)),
                "zorbleflux__0": ((6.05, 0, 0.5), (0.1, 0.1, 0.1)),
            }
        )
        slots = [
            slot("large_plant__0", (0, 0, 0.25), (0.5, 0.5, 0.5)),
            slot("trinket__0", (2, 0, 0.5), (0.12, 0.1, 0.1)),
            slot("trinket__1", (4, 0, 0.5), (0.1, 0.1, 0.1)),
            slot("trinket__2", (6, 0, 0.5), (0.1, 0.1, 0.1)),
        ]
        self.assertEqual(K.correspond(slots, s), {"trinket__1": "figurine__0", "trinket__2": "zorbleflux__0"})  # a word
        # not on the list may be anything

    def test_one_to_one_by_smallest_total_distance(self):
        s = scene({"figurine__0": ((0.0, 0, 0.5), (0.1, 0.1, 0.1)), "figurine__1": ((0.1, 0, 0.5), (0.1, 0.1, 0.1))})
        slots = [
            slot("trinket__0", (0.02, 0, 0.5), (0.1, 0.1, 0.1)),
            slot("trinket__1", (0.12, 0, 0.5), (0.1, 0.1, 0.1)),
            slot("trinket__2", (0.06, 0, 0.5), (0.1, 0.1, 0.1)),
        ]
        got = K.correspond(slots, s)
        self.assertEqual(len(got), 2)
        self.assertEqual(len(set(got.values())), 2)
        self.assertEqual((got.get("trinket__0"), got.get("trinket__1")), ("figurine__0", "figurine__1"))


class InventoryAndWords(unittest.TestCase):
    def test_the_inventory_counts_pairs_and_the_word_is_reported_apart(self):
        s = scene(
            {
                "figurine__0": ((1.0, 0, 0.5), (0.1, 0.1, 0.1)),
                "book__0": ((3.0, 0, 0.5), (0.2, 0.3, 0.05)),
                "lamp__0": ((9.0, 0, 0.5), (0.3, 0.3, 0.5)),
            }
        )
        slots = [
            slot("trinket__0", (1.0, 0, 0.5), (0.1, 0.1, 0.1)),
            slot("book__0", (3.0, 0, 0.5), (0.2, 0.3, 0.05)),
            slot("plant__0", (6.0, 0, 0.5), (0.5, 0.5, 1.0)),
        ]
        corr = K.correspond(slots, s)
        spec = dict(required=dict(trinket=1, book=1, plant=1))
        inv = Inv.inventory(spec, s, corr, slots)  # M = 2 found, 3 copy objects, 3 true: F1 = 4/6
        self.assertAlmostEqual(inv["f1"], 4 / 6)
        self.assertEqual(inv["per_category"]["trinket"], dict(required=1, generated=0, matched=1))
        w = Inv.words(slots, s, corr)
        self.assertAlmostEqual(w["accuracy"], 0.5)
        self.assertEqual(
            w["wrong"], [dict(slot="trinket__0", word="trinket", copy="figurine__0", copy_word="figurine")]
        )
        self.assertIsNone(Inv.words(slots, s, {})["accuracy"])

    def test_permitted_extras_count_only_when_paired(self):
        s = scene({"book__0": ((0, 0, 0.5), (0.2, 0.3, 0.05)), "book__1": ((2, 0, 0.5), (0.2, 0.3, 0.05))})
        slots = [slot("book__0", (0, 0, 0.5), (0.2, 0.3, 0.05))]
        inv = Inv.inventory(dict(required=dict(book=1), permitted_extras=["book"]), s, K.correspond(slots, s), slots)
        self.assertEqual(inv["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
