"""scripts/summarize.py prints a row per scene and a mean over the graded ones; a scene with no reward is named."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize.py"


class Summarize(unittest.TestCase):
    def test_table_and_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp)
            for name, r in (("3dscenebench-003-l0__a", (60, 80, 60, 40)), ("3dscenebench-019-l0__b", (40, 60, 40, 20))):
                d = job / name / "verifier"
                d.mkdir(parents=True)
                keys = ("reward", "reward_appearance", "reward_geometry", "reward_functionality")
                (d / "reward.json").write_text(json.dumps(dict(zip(keys, r))))
            (job / "3dscenebench-022-l0__c" / "verifier").mkdir(parents=True)  # the grader left no reward
            out = subprocess.run([sys.executable, str(SCRIPT), str(job)], capture_output=True, text=True, check=True)
        lines = out.stdout.splitlines()
        self.assertEqual(lines[1].split(), ["3dscenebench-003-l0", "60.0", "80.0", "60.0", "40.0"])
        self.assertIn("no scores", lines[3])
        self.assertEqual(lines[4].split()[:5], ["mean", "50.0", "70.0", "50.0", "30.0"])
        self.assertIn("(2 of 3 graded)", lines[4])

    def test_refuses_a_missing_folder(self):
        out = subprocess.run([sys.executable, str(SCRIPT), "/no/such/job"], capture_output=True, text=True)
        self.assertNotEqual(out.returncode, 0)


if __name__ == "__main__":
    unittest.main()
