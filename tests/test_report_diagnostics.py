"""Saved audit report shapes are diagnosed without changing grading artifacts."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "report_diagnostics.py"


class ReportDiagnostics(unittest.TestCase):
    def diagnose(self, report):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "report.json"
            path.write_text(json.dumps(report))
            # Adjacent grading artifacts must also remain untouched.
            (root / "reward.json").write_text('{"reward": 66.67}')
            (root / "check.json").write_text('{"ready": true, "status": "passed"}')
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(path)], capture_output=True, text=True, check=False
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual({p.name: p.read_bytes() for p in root.iterdir()}, before)
            self.assertIn("unchanged", result.stdout.lower())
            return result.stdout.lower()

    def test_reference_failure_is_incomplete_even_when_standalone_status_is_ok(self):
        text = self.diagnose({"status": "ok", "appearance": {"status": "reference no Blender runtime"}})
        self.assertIn("reference render", text)
        self.assertIn("incomplete", text)
        self.assertIn("runtime", text)
        self.assertIn("recorded status: ok", text)

    def test_copy_failure_does_not_attribute_a_worker_failure_to_the_submission(self):
        text = self.diagnose(
            {"status": "ok", "appearance": {"status": "Blender exited -9", "copy_render_failed": True}}
        )
        self.assertIn("copy_render_failed", text)
        self.assertIn("zero", text)
        self.assertIn("does not distinguish", text)
        self.assertIn("logs", text)

    def test_passed_qualification_with_missing_similarities_warns_without_changing_readiness(self):
        text = self.diagnose(
            {
                "status": "passed",
                "ready": True,
                "units": {"init": 3},
                "self": {"reward": 100.0, "siglip2": None, "dinov3": None},
            }
        )
        self.assertIn("qualification", text)
        self.assertIn("incomplete", text)
        self.assertIn("self.json", text)
        self.assertIn("recorded ready: true", text)

    def test_missing_appearance_does_not_claim_it_was_requested(self):
        text = self.diagnose({"status": "ok", "units": {}})
        self.assertIn("not recorded", text)
        self.assertIn("whether appearance was requested", text)

    def test_only_the_two_scored_similarities_determine_appearance_completeness(self):
        text = self.diagnose(
            {"status": "ok", "appearance": {"mean": {"siglip2": 0.9, "dinov3": 0.8, "lpips_vgg": None}}}
        )
        self.assertIn("both scored appearance similarities are recorded", text)
        self.assertNotIn("incomplete", text)

    def test_nonfinite_similarity_is_incomplete(self):
        text = self.diagnose({"status": "ok", "appearance": {"mean": {"siglip2": float("nan"), "dinov3": 0.8}}})
        self.assertIn("incomplete", text)
        self.assertIn("siglip2", text)

    def test_failed_later_repeat_reports_its_own_error(self):
        text = self.diagnose(
            {
                "status": "grader_failed",
                "failure": "unit exception",
                "units": {
                    "push": [
                        {"passed": {"physics_hint": True, "input_only": True}},
                        {
                            "passed": {"physics_hint": False, "input_only": False},
                            "grader_error": True,
                            "error": "RuntimeError: later repeat",
                        },
                    ]
                },
            }
        )
        self.assertIn("repeat 2", text)
        self.assertIn("runtimeerror: later repeat", text)

    def test_qualification_without_self_scores_handles_unit_counts(self):
        for extra in ({}, {"self": None}, {"self": []}):
            with self.subTest(extra=extra):
                text = self.diagnose({"status": "failed", "ready": False, "units": {"init": 3}, **extra})
                self.assertIn("qualification", text)
                self.assertIn("cannot determine", text)

    def test_malformed_nested_details_are_not_a_traceback(self):
        for report in (
            {"units": []},
            {"units": {"push": 3}, "appearance": []},
            {"units": {"push": [None, 2, "bad"]}, "appearance": {"mean": []}},
            {"F1": [], "appearance": {"mean": {"siglip2": True, "dinov3": "bad"}}},
        ):
            with self.subTest(report=report):
                self.diagnose(report)

    def test_failed_gate_without_appearance_explains_expected_skip(self):
        text = self.diagnose({"status": "ok", "F1": {"passed": False}, "units": {}})
        self.assertIn("failed f1 gate", text)
        self.assertIn("skip appearance", text)
        self.assertNotIn("appearance incomplete", text)

    def test_bad_inputs_do_not_prevent_inspecting_the_next_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            broken, array, good = (root / n for n in ("broken.json", "array.json", "good.json"))
            broken.write_text('{"status":')
            array.write_text("[]")
            good.write_text('{"status": "ok"}')
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(root / "missing.json"), str(broken), str(array), str(good)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("Recorded status: ok", result.stdout)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
