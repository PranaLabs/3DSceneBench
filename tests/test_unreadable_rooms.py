"""The pipeline names every room it cannot check, instead of leaving it out of the summary without a word."""

import json, os, tempfile, unittest
from sbv import pipeline


class UnreadableRooms(unittest.TestCase):
    def test_each_unreadable_room_is_named_with_its_reason(self):
        work = tempfile.mkdtemp()
        rooms = os.path.join(work, "rooms")
        os.makedirs(os.path.join(rooms, "empty"))
        os.symlink("/no/such/folder", os.path.join(rooms, "linked"))
        with open(os.path.join(rooms, "notes.txt"), "w") as f:
            f.write("x")
        os.makedirs(os.path.join(rooms, ".hidden"))
        said = []
        rows = pipeline.run(work, log=said.append)
        notes = {r["room"]: r["note"] for r in rows}
        self.assertEqual(sorted(notes), ["empty", "linked", "notes.txt"])  # hidden entries stay out
        self.assertIn("no scene.xml", notes["empty"])
        self.assertIn("/no/such/folder", notes["linked"])
        self.assertIn("not a folder", notes["notes.txt"])
        self.assertTrue(all(n.startswith("skipped: ") for n in notes.values()))
        self.assertEqual(len([s for s in said if "skipped" in s]), 3)
        with open(os.path.join(work, "results", "summary.md")) as f:
            self.assertIn("linked", f.read())
        with open(os.path.join(work, "results", "summary.json")) as f:
            self.assertEqual(len(json.load(f)), 3)


if __name__ == "__main__":
    unittest.main()
