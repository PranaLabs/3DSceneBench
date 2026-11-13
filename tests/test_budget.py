"""A room's budget.json sets its grader's CPUs, memory and minutes in task.toml; the agent's budget is the same for
every task, whatever its level or room (work minutes and output tokens, --budget MINUTESm:TOKENS), and is written in
task.toml [metadata], in [agent] timeout_sec (the minutes plus 15 min of margin) and in the instruction; a wrong budget
stops the run."""

import contextlib, io, json, os, tempfile, tomllib, unittest
from unittest import mock
from scenebench_harbor import make_tasks as M
from sbv.common import case as K
from tests import rooms
from tests.test_levels import pictures  # one picture and a real camera, for the levels
from tests.harbor_fixtures import metric_environment, receipt

OBJECTS = [dict(slot=f"cup__{i}", category="cup") for i in range(3)]  # the levels need real ones: OBJECTS[:1]


def work_with(budget=None, objects=OBJECTS):
    work = tempfile.mkdtemp()
    res = os.path.join(work, "results", "room")
    os.makedirs(res)
    rooms.write(os.path.join(work, "rooms", "room"))
    if budget is not None:
        with open(os.path.join(work, "rooms", "room", "budget.json"), "w") as f:
            json.dump(budget, f)
    K.save(
        dict(
            schema=K.SCHEMA,
            reference="../../rooms/room/scene.xml",
            vocabulary=["cup", "table"],
            objects=objects,
            functionality=dict(units=[]),
        ),
        os.path.join(res, "case.json"),
    )
    receipt(work)
    return work


def made(work, *extra):
    out = os.path.join(work, "tasks")
    with metric_environment(), mock.patch.object(M.App, "input_pictures", pictures):
        M.main(["--work", work, "--out", out, "--pictures", "1", *extra])

    def read(side, name, level=""):
        task = next(e.path for e in os.scandir(os.path.join(out, side)) if e.is_dir() and e.name.endswith(level))
        with open(os.path.join(task, name)) as f:
            return f.read()

    return read


TEXT = (
    "You can work for up to 3 hours or generate up to 10,000,000 output tokens (your reasoning included), whichever\n"
    "comes first. Then the run stops, and the verifiers score the scene that is in `/output/scene` at that moment.\n"
    "Run `time_left` to see how much time is left. It never shows more than you have. You have 2 CPUs and 16 GB of memory."
)


class Budget(unittest.TestCase):
    def test_the_defaults_without_a_budget_file(self):
        read = made(work_with())
        toml = read("private", "task.toml")
        self.assertIn("cpus = 8\nmemory_mb = 8192\n", toml)
        self.assertIn("timeout_sec = 5400.0\n", toml)
        self.assertIn("[agent]\ntimeout_sec = 11700.0\n", toml)  # 3 h + 15 min
        self.assertIn(
            "### Budget\n\n" + TEXT + "\n\n## Input files", read("private", "instruction.md")
        )  # without levels too
        self.assertIn(
            "cpus = 1\nmemory_mb = 8192\n", read("public", "task.toml")
        )  # the public format check: 8 GB, whatever the room
        for side in ("public", "private"):
            self.assertIn("[environment]\ncpus = 2\nmemory_mb = 16384\n", read(side, "task.toml"))  # the agent: 16 GB

    def test_the_rooms_budget_reaches_the_private_grader_only(self):
        read = made(work_with(dict(grader=dict(cpus=16, memory_mb=32768, minutes=150))))
        toml = read("private", "task.toml")
        self.assertIn("cpus = 16\nmemory_mb = 32768\n", toml)
        self.assertIn("timeout_sec = 9000.0\n", toml)
        self.assertIn("cpus = 1\nmemory_mb = 8192\n", read("public", "task.toml"))
        self.assertNotIn("32768", read("public", "task.toml"))
        read = made(work_with(dict(grader=dict(minutes=120))))
        self.assertIn("cpus = 8\nmemory_mb = 8192\n", read("private", "task.toml"))  # the rest stays default

    def test_every_task_has_the_same_budget_in_three_places(self):
        read = made(work_with(objects=OBJECTS[:1]), "--levels", "l0", "l1")
        for lv in ("l0", "l1"):  # no difference by level
            for side in ("public", "private"):
                t = tomllib.loads(read(side, "task.toml", lv))
                self.assertEqual(
                    (t["metadata"]["budget_minutes"], t["metadata"]["budget_output_tokens"]), (180, 10_000_000)
                )
                self.assertEqual(t["agent"]["timeout_sec"], 180 * 60.0 + 15 * 60)  # Harbor's backstop only
                self.assertIn("### Budget\n\n" + TEXT + "\n\n", read(side, "instruction.md", lv))
            pub, priv = (tomllib.loads(read(side, "task.toml", lv)) for side in ("public", "private"))
            self.assertEqual(
                (pub["metadata"], pub["agent"], pub["task"]), (priv["metadata"], priv["agent"], priv["task"])
            )
        big = [dict(slot=f"cup__{i}", category="cup") for i in range(80)]  # nor by the room's size
        self.assertIn(
            "budget_minutes = 180\nbudget_output_tokens = 10000000\n",
            made(work_with(objects=big))("public", "task.toml"),
        )

    def test_one_budget_flag_for_every_task(self):
        read = made(work_with(objects=OBJECTS[:1]), "--levels", "l0", "l1", "--budget", "90m:250000")
        for lv in ("l0", "l1"):
            t = tomllib.loads(read("public", "task.toml", lv))
            self.assertEqual(
                (t["metadata"]["budget_minutes"], t["metadata"]["budget_output_tokens"], t["agent"]["timeout_sec"]),
                (90, 250000, 6300.0),
            )
            self.assertIn(
                "up to 90 minutes or generate up to 250,000 output tokens", read("public", "instruction.md", lv)
            )
        for bad in ("l1=120m:500000", "0m:1000", "60m", "60m:x"):
            err = io.StringIO()
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(err):
                made(work_with(), "--budget", bad)
            self.assertIn("MINUTESm:TOKENS", err.getvalue())
        for gone in ("--agent-hours", "--agent-hours-above"):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                made(work_with(), gone, "2")

    def test_a_wrong_budget_stops_the_run_with_the_reason(self):
        for budget, reason in [
            (dict(grader=dict(gpus=1)), "unknown field ['grader.gpus']"),
            (dict(agent=dict(hours=9)), "unknown field ['agent']"),
            (dict(grader=dict(cpus=0)), "grader.cpus must be a positive number"),
            (dict(grader=dict(minutes="90")), "grader.minutes must be a positive number"),
        ]:
            with self.assertRaises(SystemExit) as cm:
                made(work_with(budget))
            self.assertIn(reason, str(cm.exception))


if __name__ == "__main__":
    unittest.main()
