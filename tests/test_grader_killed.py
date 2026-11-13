"""A compile check killed by SIGKILL (the out-of-memory killer) is the grader's failure, not "does not compile"."""

import subprocess, unittest
from unittest import mock
from sbv.functionality import f1_gate


class GraderKilled(unittest.TestCase):
    def test_sigkill_is_not_the_copys_fault(self):
        with mock.patch.object(f1_gate.subprocess, "run", return_value=subprocess.CompletedProcess([], -9, "", "")):
            with self.assertRaises(f1_gate.GraderKilled):
                f1_gate.compile_error("scene.xml")
        with mock.patch.object(
            f1_gate.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "XML Error: bad tag\n")
        ):
            self.assertEqual(
                f1_gate.compile_error("scene.xml"), "XML Error: bad tag"
            )  # the copy's own error, as before


if __name__ == "__main__":
    unittest.main()
