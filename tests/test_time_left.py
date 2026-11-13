"""time_left (checker/time_left.py), in every task container as /usr/local/bin/time_left: the time used and left of the
task's budget, counted from the container's start, so it never shows more time than the run really has."""

import contextlib, io, json, os, tempfile, unittest
from checker import time_left as T
from tests.test_budget import made, work_with


def fake_proc(boot, start_ticks):
    """A /proc with the boot time and a first process that started start_ticks after boot."""
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "1"))
    with open(os.path.join(d, "stat"), "w") as f:
        f.write(f"cpu  1 2 3\nbtime {boot}\nprocesses 9\n")
    fields = (
        ["S"] + ["0"] * 18 + [str(start_ticks)] + ["0"] * 10
    )  # field 22 of /proc/<pid>/stat: the start time in ticks
    with open(os.path.join(d, "1", "stat"), "w") as f:
        f.write("1 (sleep (a) b) " + " ".join(fields) + "\n")  # a name with ")" in it
    return d


class TimeLeft(unittest.TestCase):
    def test_the_container_start_comes_from_the_first_process(self):
        hz = os.sysconf("SC_CLK_TCK")
        self.assertAlmostEqual(T.container_start(fake_proc(1_000_000, 50 * hz)), 1_000_050.0)

    @unittest.skipUnless(os.path.exists("/proc/1/stat"), "needs Linux /proc")
    def test_the_real_proc_gives_a_start_in_the_past(self):
        import time

        self.assertLess(T.container_start(), time.time())

    def test_used_and_left(self):
        self.assertEqual(T.left(100.0, 100.0 + 72 * 60, 180), (72 * 60, 108 * 60))
        self.assertEqual(T.left(100.0, 100.0 + 200 * 60, 180)[1], 0.0)  # past the budget: nothing left, never negative
        text = T.report(100.0, 100.0 + 72 * 60 + 5, 180, 1_000_000)
        self.assertIn("Time used: 1 h 12 min of 3 h 00 min.", text)
        self.assertIn("Time left: at least 1 h 47 min.", text)
        self.assertIn("1,000,000 output tokens", text)
        self.assertIn("Time left: at least 4 min 30 s.", T.report(0.0, 175 * 60 + 30, 180))

    def test_the_command_reads_the_budget_beside_it(self):
        d = tempfile.mkdtemp()
        budget = os.path.join(d, "budget.json")
        with open(budget, "w") as f:
            json.dump(dict(minutes=180, output_tokens=1_000_000), f)
        proc = fake_proc(1_000_000, 0)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            T.main(["--seconds"], budget, proc, now=1_000_000 + 600)
        self.assertEqual(out.getvalue().strip(), str(170 * 60))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            T.main([], budget, proc, now=1_000_000 + 600)
        self.assertIn("Time left: at least 2 h 50 min.", out.getvalue())

    def test_every_task_container_gets_it_with_its_budget(self):
        read = made(work_with(), "--budget", "90m:500000")
        self.assertIn(
            "COPY check_scene.py vocabulary.json time_left.py budget.json /opt/scenebench/\n",
            read("public", "environment/Dockerfile"),
        )
        self.assertIn(
            "ln -s /opt/scenebench/time_left.py /usr/local/bin/time_left", read("public", "environment/Dockerfile")
        )
        self.assertEqual(json.loads(read("public", "environment/budget.json")), dict(minutes=90, output_tokens=500000))
        self.assertEqual(read("public", "environment/time_left.py"), open(T.__file__).read())
        self.assertIn("Run `time_left` to see", read("public", "instruction.md"))


if __name__ == "__main__":
    unittest.main()
