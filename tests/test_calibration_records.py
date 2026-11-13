"""Atomic job records, execution errors and exclusive checkpoint ownership."""

import tempfile
import unittest
from pathlib import Path
from sbv.common.calibration_jobs import Journal, _execute, atomic, note, read


def record(value):
    note("inside task")
    if value < 0:
        raise ValueError("bad value")
    return {"value": value}


class Records(unittest.TestCase):
    def test_completion_failure_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "record.json"
            self.assertEqual(_execute(record, 3, p, "key", "job", "phase"), {"value": 3})
            r = read(p)
            self.assertEqual(r["status"], "completed")
            self.assertEqual(r["stage"], "inside task")
            self.assertIn("result_hash", r)
            self.assertGreaterEqual(r["seconds"], 0)
            with self.assertRaises(ValueError):
                _execute(record, -1, p, "key", "job", "phase")
            self.assertEqual(read(p)["status"], "error")
            p.write_text("{broken")
            self.assertIsNone(read(p))
            atomic(p, {"ok": True})
            self.assertEqual(read(p), {"ok": True})

    def test_lock_identity_and_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with Journal(tmp, {"room": 1}, log=lambda s: None) as j:
                with self.assertRaisesRegex(RuntimeError, "already in use"):
                    with Journal(tmp, {"room": 2}):
                        pass
                p = j.root / "objects" / "job.json"
                atomic(p, {"status": "completed"})
                j.invalidate()
                self.assertEqual(read(p)["status"], "invalidated")
            with Journal(tmp, {"room": 2}, log=lambda s: None) as other:
                self.assertNotEqual(j.root, other.root)

    def test_invalid_timeout(self):
        for value in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                Journal(".", {}, timeout=value)
