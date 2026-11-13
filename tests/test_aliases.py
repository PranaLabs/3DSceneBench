"""One kind of thing, several words: an alias in sbv/common/categories.txt is scored as its main word, in the copy and
in the case, and the instruction and the format checker say so. Run from the repository root:
    python -m unittest tests.test_aliases -v
"""

import json, os, subprocess, sys, tempfile, unittest
from types import SimpleNamespace
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from scenebench_harbor import make_tasks as M  # noqa: E402
from sbv.common import case as K  # noqa: E402
from sbv.common.scene import word_groups, main_word, category  # noqa: E402
from sbv.geometry import g1_inventory as Inv  # noqa: E402
from tests import rooms  # noqa: E402


class Aliases(unittest.TestCase):
    def test_each_word_once_and_aliases_map_to_their_main_word(self):
        groups = word_groups()
        words = [w for m, al in groups.items() for w in [m] + al]
        self.assertEqual(len(words), len(set(words)))  # no word in two groups, no alias that is a main word
        self.assertNotIn("bookshelf", groups)
        self.assertIn("ornament", groups)
        self.assertFalse([w for w in words if "/" in w])
        self.assertEqual(groups["bookcase"], ["bookshelf"])
        self.assertIn("shelf", groups)
        self.assertNotIn("book_stack", groups)
        self.assertEqual(
            [main_word(w) for w in ("bookshelf", "mug", "trinket", "bookcase", "vase")],
            ["bookcase", "cup", "ornament", "bookcase", "vase"],
        )
        self.assertEqual(
            [category(n) for n in ("bookshelf__0", "mug__3", "cup_holder__0")], ["bookcase", "cup", "cup_holder"]
        )
        self.assertEqual(category("mug_left", vocabulary=["mug", "table"]), "cup")
        for w in ("vase", "pot", "glass", "shelf", "office_chair", "floor_lamp"):
            self.assertEqual(main_word(w), w)  # kept apart

    def test_an_old_case_reads_its_true_words_as_main_words(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "case.json")
        K.save(
            dict(
                schema=K.SCHEMA,
                objects=[dict(slot="trinket__0", category="trinket"), dict(slot="ornament__0", category="ornament")],
                geometry=dict(
                    inventory=dict(required=dict(trinket=1, ornament=1), permitted_extras=["mug"]),
                    count_constraints=[dict(category="figurine", op="at_most", n=3)],
                ),
            ),
            path,
        )
        case = K.load(path)
        g = case["geometry"]
        self.assertEqual([o["category"] for o in case["objects"]], ["ornament", "ornament"])
        self.assertEqual(
            (g["inventory"]["required"], g["inventory"]["permitted_extras"], g["count_constraints"][0]["category"]),
            ({"ornament": 2}, ["cup"], "ornament"),
        )
        with open(path) as f:
            self.assertEqual(json.load(f)["case_hash"], case["case_hash"])  # the hash is the file's, as before

    def test_an_alias_is_paired_and_counted_as_its_main_word(self):
        at = {"figurine__0": (1.0, 0, 0.5), "bookshelf__0": (3.0, 0, 1.0), "vase__0": (1.05, 0, 0.5)}
        scene = SimpleNamespace(
            objects=at, cat={n: category(n) for n in at}, center=lambda n: np.array(at[n]), joints={n: [] for n in at}
        )
        slots = [
            dict(slot="trinket__0", category=main_word("trinket"), center=(1.0, 0, 0.5)),
            dict(slot="bookcase__0", category="bookcase", center=(3.0, 0, 1.0)),
        ]
        corr = K.correspond(slots, scene)
        self.assertEqual(corr, {"trinket__0": "figurine__0", "bookcase__0": "bookshelf__0"})
        self.assertAlmostEqual(
            Inv.inventory(dict(required=dict(ornament=1, bookcase=1)), scene, corr, slots)["f1"], 0.8
        )  # the vase is an extra


class Instruction(unittest.TestCase):
    def test_the_list_shows_each_alias_beside_its_main_word(self):
        text = M.word_file()
        self.assertIn("\n  bookcase (also: bookshelf)\n", text)
        self.assertIn("\n  cup (also: mug, teacup", text)
        self.assertIn("is scored as the first word", text)
        self.assertNotIn("\n  bookshelf", text)
        self.assertIn("bookshelf", M.categories())

    def test_every_kind_is_listed_once_under_its_topic(self):
        lines = M.word_file().splitlines()
        mains = [x.split()[0] for x in lines if x.startswith("  ")]
        self.assertEqual(sorted(mains), sorted(word_groups()))
        self.assertEqual(len(mains), len(set(mains)))
        self.assertLess(
            lines.index("  sofa (also: couch, settee, loveseat, sectional, sofa_bed, futon)"),
            lines.index("tables and desks"),
        )

    def test_the_format_checker_names_each_alias_and_unlisted_word(self):
        d = tempfile.mkdtemp()
        xml = rooms.write(os.path.join(d, "room"))
        with open(xml) as f:
            text = f.read()
        with open(xml, "w") as f:
            f.write(text.replace('"cup__0"', '"mug__0"').replace('"crate__0"', '"zorbleflux__0"'))
        words = os.path.join(d, "vocabulary.json")
        with open(words, "w") as f:
            json.dump(M.vocabulary(M.categories()), f)
        root = os.path.dirname(HERE)
        run = subprocess.run(
            [
                sys.executable,
                os.path.join(root, "checker", "check_scene.py"),
                os.path.dirname(xml),
                "--vocabulary",
                words,
            ],
            capture_output=True,
            text=True,
            cwd=root,
        )
        out = json.loads(run.stdout)
        self.assertEqual(
            (out["aliases"], out["words_not_on_the_list"]), (["mug__0 is scored as cup"], ["zorbleflux__0"])
        )


if __name__ == "__main__":
    unittest.main()
