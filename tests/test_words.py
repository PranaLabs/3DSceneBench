"""The shared word list only grows (sbv/words.py): every released word keeps its main word and its
topic, so adding words moves no released room's scores, and new words are plain names. Run from the repository root:
    python -m unittest tests.test_words -v
"""

import os, re, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from sbv import words as W  # noqa: E402
from sbv.common.scene import ARCH, CATEGORIES, category, is_arch  # noqa: E402

# ew ones still fail.
LISTED_WALL_WORDS = {"wall_isolator", "wall_shelf"}

OLD = "# seating\nchair sofa\n# decor\nvase plant\n\n# groups\nsofa/couch\n"


def write(d, text):
    path = os.path.join(d, f"list{len(os.listdir(d))}.txt")
    with open(path, "w") as f:
        f.write(text)
    return path


class OnlyGrows(unittest.TestCase):
    def test_no_released_word_changed(self):
        self.assertEqual(W.changes(), [])

    def test_adding_words_kinds_and_aliases_changes_nothing(self):
        d = tempfile.mkdtemp()
        old = write(d, OLD)
        new = write(
            d,
            "# seating\nchair sofa stool\n# decor\nvase plant\n# pets\npet_bed\n\n# groups\nsofa/couch/settee stool/bar_stool\n",
        )
        self.assertEqual((W.changes(old, new), W.added(old, new)), ([], ["bar_stool", "pet_bed", "settee", "stool"]))

    def test_each_kind_of_change_is_caught(self):
        d = tempfile.mkdtemp()
        old = write(d, OLD)
        self.assertEqual(W.changes(old, write(d, OLD.replace("vase plant", "vase"))), ["plant: removed"])
        self.assertEqual(
            W.changes(old, write(d, OLD.replace("sofa/couch", "sofa/couch/chair"))),
            ["chair: scored as sofa, was chair"],
        )
        self.assertEqual(W.changes(old, write(d, OLD.replace("sofa/couch", ""))), ["couch: removed"])
        self.assertEqual(
            W.changes(old, write(d, "# seating\nchair sofa vase\n# decor\nplant\n\n# groups\nsofa/couch\n")),
            ["vase: under topic 'seating', was 'decor'"],
        )

    def test_the_command_fails_on_a_change(self):
        d = tempfile.mkdtemp()
        old = write(d, OLD)
        self.assertEqual(W.main([old, write(d, OLD.replace("chair sofa", "chair sofa stool"))]), 0)
        self.assertEqual(W.main([old, write(d, OLD.replace("vase plant", "vase"))]), 1)


class WallWords(unittest.TestCase):
    def test_listed_wall_words_name_objects_not_walls(self):
        for w in sorted(LISTED_WALL_WORDS):
            for name in (f"{w}__0", f"{w}_2"):
                self.assertEqual(category(name), w, name)
                self.assertFalse(is_arch(name), name)
        self.assertTrue(is_arch("wall__0"))
        self.assertTrue(is_arch("wall_2"))  # the word wall itself is still the building


class NewWords(unittest.TestCase):
    def test_new_words_are_plain_names(self):
        new = W.added()
        self.assertEqual(
            [w for w in new if not re.fullmatch(r"[a-z][a-z_]*[a-z]", w) or "__" in w], []
        )  # no digits: n is the number
        self.assertEqual(
            [w for w in new if re.match(r"(wall|floor|ceiling)(_|$)", w) and w not in LISTED_WALL_WORDS], []
        )  # the harness refuses these
        self.assertEqual([w for w in new if w in ARCH or w.startswith("room_geometry")], [])  # these are the building

    def test_the_groups_block_holds_groups_only(self):
        """A word without a slash below the groups' comment would take that comment as its topic."""
        with open(CATEGORIES) as f:
            lines = f.read().splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith("# Kinds of thing with several words"))
        loose = [w for line in lines[start:] if not line.lstrip().startswith("#") for w in line.split() if "/" not in w]
        self.assertEqual(loose, [])


if __name__ == "__main__":
    unittest.main()
