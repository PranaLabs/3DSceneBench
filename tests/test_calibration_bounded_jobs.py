"""Bounded parallel jobs retain completed work and preserve result order."""

import os
import json
from pathlib import Path
import tempfile
import time
import unittest
from sbv.common.calibration_jobs import Journal, CalibrationIncomplete, note, read


def work(arg):
    note("doing " + str(arg["value"]))
    time.sleep(arg.get("delay", 0))
    if arg.get("error"):
        raise ValueError("deliberate error")
    return {"value": arg["value"]}


def slow_init():
    time.sleep(10)


def die(arg):
    os._exit(9)


class Jobs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.logs = []

    def journal(self, **kw):
        return Journal(self.tmp.name, {"fixture": 1}, self.logs.append, **kw)

    def test_out_of_order_completion_and_resume_preserve_order(self):
        jobs = [("slow", dict(value=1, delay=0.3)), ("fast", dict(value=2))]
        with self.journal() as j:
            first = j.jobs("objects", work, jobs, procs=2)
            stamps = {p.name: p.stat().st_mtime_ns for p in j.root.glob("objects/*.json")}
            again = j.jobs("objects", work, jobs, procs=2)
            self.assertEqual(stamps, {p.name: p.stat().st_mtime_ns for p in j.root.glob("objects/*.json")})
        self.assertEqual(first, again)
        self.assertEqual([x["value"] for x in first], [1, 2])
        completed = [s for s in self.logs if "completed:" in s]
        self.assertIn("fast", completed[0])
        self.assertTrue(any("2/2 completed (reused)" in s for s in self.logs))

    def test_timeout_is_not_a_result_and_completed_siblings_are_kept(self):
        jobs = [("fast", dict(value=1)), ("slow", dict(value=2, delay=10)), ("queued", dict(value=3, delay=10))]
        start = time.monotonic()
        with self.journal(timeout=2) as j:
            with self.assertRaises(CalibrationIncomplete):
                j.jobs("objects", work, jobs, procs=2)
            states = [read(p)["status"] for p in j.root.glob("objects/*.json")]
            self.assertIn("completed", states)
            self.assertIn("timeout", states)
        self.assertLess(time.monotonic() - start, 6)

    def test_initializer_timeout_is_bounded(self):
        with self.journal(timeout=0.3) as j:
            with self.assertRaises(CalibrationIncomplete):
                j.jobs("objects", work, [("a", dict(value=1))], initializer=slow_init)
            self.assertEqual(read(next(j.root.glob("objects/*.json")))["status"], "timeout")

    def test_errors_and_not_run_are_distinct_and_retried(self):
        with self.journal() as j:
            with self.assertRaisesRegex(ValueError, "deliberate"):
                j.jobs("objects", work, [("bad", dict(value=1, error=True)), ("later", dict(value=2))])
            rows = [read(p) for p in j.root.glob("objects/*.json")]
            self.assertEqual({r["status"] for r in rows}, {"error", "not_run"})
            # An interrupted record cannot be reused even if it contains an old result.
            for p in j.root.glob("objects/*.json"):
                r = read(p)
                r["status"] = "interrupted"
                p.write_text(json.dumps(r))
            self.assertEqual(j.jobs("objects", work, [("later", dict(value=2))]), [{"value": 2}])

    def test_worker_death_is_an_execution_error(self):
        from sbv.common.workers import WorkerDied

        with self.journal() as j:
            with self.assertRaises(WorkerDied):
                j.jobs("objects", die, [("a", {})])
            self.assertNotEqual(read(next(j.root.glob("objects/*.json")))["status"], "completed")
