"""Export refuses a room with missing frozen files, or an obsolete task selection, before changing any output."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scenebench_harbor import make_tasks as M
from sbv.common import case as K
from tests import rooms
from tests.harbor_fixtures import metric_environment, receipt
from tests.test_levels import pictures


class ExportPreflight(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        self.out = self.work / "export"
        self.add_room("a")

    def add_room(self, name):
        rd = self.work / "rooms" / name
        res = self.work / "results" / name
        rooms.write(rd)
        res.mkdir(parents=True)
        K.save(
            dict(
                schema=K.SCHEMA,
                reference=f"../../rooms/{name}/scene.xml",
                vocabulary=["cup"],
                objects=[],
                functionality=dict(units=[]),
            ),
            res / "case.json",
        )
        receipt(self.work, name)

    def build(self, *args):
        with (
            metric_environment(),
            mock.patch.object(M.App, "input_pictures", pictures),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return M.main(["--work", str(self.work), "--out", str(self.out), "--pictures", "1", *args])

    def snapshot(self):
        return (
            {p.relative_to(self.out).as_posix(): p.read_bytes() for p in self.out.rglob("*") if p.is_file()}
            if self.out.exists()
            else {}
        )

    def refused_unchanged(self, *args, reason="frozen files incomplete"):
        before = self.snapshot()
        with self.assertRaisesRegex(SystemExit, reason):
            self.build(*args)
        self.assertEqual(self.snapshot(), before)
        if not before:
            self.assertFalse(self.out.exists())

    def change_check(self, fn):
        path = self.work / "results/a/check.json"
        record = json.loads(path.read_text())
        fn(record)
        path.write_text(json.dumps(record))

    def declare(self, axis, room="a"):
        path = self.work / "rooms" / room / ("nonrigid.json" if axis == "nonrigid" else "tasks.json")
        path.write_text('{"objects":[]}' if axis == "nonrigid" else '{"package":[]}')
        return path

    def test_declared_axis_without_its_frozen_case_refuses_fresh_export(self):
        for axis in ("nonrigid", "package"):
            with self.subTest(axis=axis):
                declaration = self.declare(axis)
                self.refused_unchanged()
                declaration.unlink()

    def test_unfinished_check_is_refused(self):
        for status in ("running", "interrupted"):
            with self.subTest(status=status):
                self.change_check(lambda r: r.update(status=status))
                self.refused_unchanged()

    def test_unreadable_case_is_refused(self):
        (self.work / "results/a/case.json").write_text("{")
        self.refused_unchanged()

    def test_room_frozen_on_earlier_code_still_exports(self):
        # The released rooms were frozen on earlier commits; some receipts carry no artifact list or self measurements.
        for change in ("version", "artifacts", "self"):
            with self.subTest(change=change):
                receipt(self.work, "a")
                self.change_check(lambda r: r.update(version="older") if change == "version" else r.pop(change))
                self.build()
                self.assertTrue((self.out / "private/3dscenebench-001/tests/case.json").is_file())

    def test_one_incomplete_room_refuses_the_whole_export(self):
        self.add_room("b")
        self.declare("nonrigid", "b")
        self.refused_unchanged()

    def test_all_rooms_checked_before_existing_export_or_id_map_changes(self):
        self.build()
        self.add_room("b")
        self.change_check(lambda r: r.update(ready=False))
        self.declare("nonrigid", "b")
        self.refused_unchanged()

    def test_removed_level_or_withdrawn_room_refuses_without_deleting_old_tasks(self):
        self.build("--levels", "l0", "l1")
        self.refused_unchanged("--levels", "l0", reason="stale generated tasks")
        self.change_check(lambda r: r.update(ready=False))
        self.refused_unchanged("--levels", "l0", "l1", reason="stale generated tasks")

    def test_unrelated_user_files_are_preserved_and_same_selection_can_be_rebuilt(self):
        self.build()
        notes = self.out / "public/notes"
        notes.mkdir()
        (notes / "keep.txt").write_text("user notes")
        self.build()
        self.assertEqual((notes / "keep.txt").read_text(), "user notes")

    def test_existing_export_needs_its_original_id_map(self):
        self.build()
        (self.out / "private/room_ids.json").unlink()
        self.refused_unchanged(reason="room ID map")

    def test_duplicate_chosen_room_ids_are_refused_before_writing(self):
        self.add_room("b")
        mapping = self.work / "original-ids.json"
        mapping.write_text('{"a":"001","b":"001"}')
        self.refused_unchanged("--room-ids", str(mapping), reason="duplicate room IDs")
        self.assertEqual(mapping.read_text(), '{"a":"001","b":"001"}')


if __name__ == "__main__":
    unittest.main()
