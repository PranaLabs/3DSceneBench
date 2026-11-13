"""Cache behavior is checked by execution counts, not by matching its implementation."""

import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from sbv.common import dependencies as D
from sbv.common.phase_cache import cached
from sbv.package_tasks import verify as V
from tests.test_package_tasks import room_with


class Evidence(unittest.TestCase):
    def test_only_changed_phase_reexecutes_and_corrupt_artifact_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            inputs = {"rigid": 1, "appearance": 1}
            calls = []

            def phase(name):
                def compute():
                    calls.append(name)
                    (out / (name + ".txt")).write_text(str(inputs[name]))
                    return {"value": inputs[name]}

                with D.session():
                    return cached(out, name, lambda: {"input": inputs[name]}, compute, [name + ".txt"])

            for _ in range(2):
                phase("rigid")
                phase("appearance")
            self.assertEqual(calls, ["rigid", "appearance"])
            inputs["appearance"] = 2
            phase("rigid")
            phase("appearance")
            self.assertEqual(calls, ["rigid", "appearance", "appearance"])
            artifact = next((out / ".phase-cache/rigid").glob("*/files/rigid.txt"))
            artifact.write_text("tampered")
            phase("rigid")
            self.assertEqual(calls[-1], "rigid")
            self.assertEqual(len(calls), 4)

    def test_end_of_operation_revokes_only_changed_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            key = {"a": 1, "b": 1}
            with self.assertRaisesRegex(RuntimeError, "dependencies changed"):
                with D.session():
                    for name in ("a", "b"):
                        cached(out, name, lambda n=name: dict(value=key[n]), lambda: dict(ok=True))
                    key["b"] = 2
            self.assertEqual(len(list((out / ".phase-cache/a").glob("*/evidence.json"))), 1)
            self.assertEqual(len(list((out / ".phase-cache/b").glob("*/evidence.json"))), 0)

    def test_runtime_probe_shared_with_threads_but_refreshed_before_publication(self):
        from concurrent.futures import ThreadPoolExecutor
        import sys

        response = json.dumps({"fixture": "runtime"})
        with patch.object(D.subprocess, "check_output", return_value=response) as probe:
            with D.session():
                with ThreadPoolExecutor(4, initializer=D.use_session, initargs=(D.current_session(),)) as pool:
                    list(pool.map(lambda _: D.runtime(sys.executable, packages=("numpy",)), range(20)))
                D.published(None, invalidate=lambda: None)
            self.assertEqual(probe.call_count, 2)


class TaskIndependence(unittest.TestCase):
    def test_one_task_edit_reuses_other_tasks_and_live_grading_still_executes(self):
        temp, room = room_with()
        self.addCleanup(shutil.rmtree, temp, True)
        room = Path(room)
        root = Path(temp)
        tasks = json.loads((room / "tasks.json").read_text())
        first = tasks["package"][0]
        first["entry"] = "task_a/push.py"
        (room / "task_a").mkdir()
        shutil.copy2(room / "package_tasks/push.py", room / first["entry"])
        second = json.loads(json.dumps(first))
        second.update(id="other", entry="task_b/push.py")
        (room / "task_b").mkdir()
        shutil.copy2(room / first["entry"], room / second["entry"])
        tasks["package"].append(second)
        (room / "tasks.json").write_text(json.dumps(tasks))
        out = root / "case.json"
        report = root / "report.json"
        report.write_text(json.dumps({"geometry": {}, "correspondence": {"book__0": "book__0"}}))
        with (
            patch.dict(os.environ, SBV_PACKAGE_CACHE=str(root / "cache")),
            patch.object(V, "run_one", wraps=V.run_one) as run,
        ):
            V.calibrate(str(room), str(out), 2)
            self.assertEqual(run.call_count, 8)
            V.calibrate(str(room), str(out), 2)
            self.assertEqual(run.call_count, 8)
            first["params"]["force_n"] *= 1.01
            (room / "tasks.json").write_text(json.dumps(tasks))
            V.calibrate(str(room), str(out), 2)
            self.assertEqual(run.call_count, 12)
            (room / "notes.md").write_text("editorial change")
            first["words"] += " Carefully."
            (room / "tasks.json").write_text(json.dumps(tasks))
            V.calibrate(str(room), str(out), 2)
            self.assertEqual(run.call_count, 12)
            for _ in range(2):
                V.evaluate(str(out), str(room), str(report), str(root / "evaluation.json"), 2, qualification=True)
            self.assertEqual(run.call_count, 14)
            V.evaluate(str(out), str(room), str(report), str(root / "live.json"), 2)
            self.assertEqual(run.call_count, 16)
