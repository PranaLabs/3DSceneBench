"""Harbor tasks for rooms with non-rigid objects: the public task gets the side file's section and check and never the
answer; the private grader gets the non-rigid case and runs the non-rigid verifiers last. Needs requirements/core.txt.
    python -m unittest tests.test_nonrigid_tasks -v
"""

import json, os, subprocess, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from scenebench_harbor import make_tasks as M  # noqa: E402
from tests import rooms  # noqa: E402

CUSHION = dict(
    name="cushion__0",
    kind="soft_body",
    on="chair__0",
    shape=dict(type="box", pos=[1.5, 0.62, 0.52], size=[0.4, 0.35, 0.07]),
    material=dict(youngs_kpa=40, poisson=0.3, density=50),
)


class SoftTasks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="sbv_soft_tasks_")
        room = os.path.dirname(rooms.write(os.path.join(cls.tmp, "room")))
        pics = os.path.join(cls.tmp, "pics")
        os.makedirs(pics)
        for f in ("input_0.png", "cameras.json"):
            open(os.path.join(pics, f), "w").close()
        cls.nonrigid = os.path.join(cls.tmp, "nonrigid_case.json")
        with open(cls.nonrigid, "w") as f:
            json.dump(
                dict(
                    schema="nonrigid_case/v0",
                    room="/abs/room",
                    threads="4",
                    units=[dict(id="S1.cushion__0", kind="soft_body"), dict(id="S0.cushion__0", kind="size")],
                ),
                f,
            )
        cls.task = lambda side, nonrigid: M.write_task(
            os.path.join(cls.tmp, side + str(bool(nonrigid))),
            "pranalabs",
            "t",
            "room",
            "1",
            pics,
            ["chair"],
            grader=(room, dict(schema="x"), "base", dict(cpus=8, memory_mb=None, minutes=90))
            if side == "private"
            else None,
            nonrigid=nonrigid,
        )
        for side in ("public", "private"):
            for nonrigid in (None, cls.nonrigid):
                cls.task(side, nonrigid)

    def read(self, *p):
        with open(os.path.join(self.tmp, *p)) as f:
            return f.read()

    def test_only_soft_rooms_get_the_section_and_the_check(self):
        text = self.read("publicTrue", "instruction.md")
        self.assertLess(text.index("## Output files"), text.index("### Non-rigid objects"), text.index("## Evaluation"))
        self.assertIn("| `/output/scene/nonrigid.json` | json | Yes, in this room |", text)
        self.assertNotIn("Non-rigid objects", self.read("publicFalse", "instruction.md"))
        self.assertIn("in MuJoCo and Genesis\n", text)
        self.assertIn("run in Genesis", text)  # two simulators, said in the title and the goal
        rigid = self.read("publicFalse", "instruction.md")
        self.assertIn("as one MuJoCo scene.", rigid)
        self.assertNotIn("nonrigid.json` |", rigid)
        self.assertTrue(
            rigid.startswith("# 3DSceneBench: build room room as a functional and reusable world in MuJoCo\n")
        )
        self.assertIn("| **Functionality**, in four levels", rigid)
        self.assertNotIn("Physics, in four levels", rigid)
        self.assertIn("--nonrigid", self.read("publicTrue", "tests", "test.sh"))
        self.assertNotIn("--nonrigid", self.read("publicFalse", "tests", "test.sh"))
        self.assertFalse(
            any("nonrigid_case" in f for _, _, fs in os.walk(os.path.join(self.tmp, "publicTrue")) for f in fs)
        )

    def test_the_private_grader_runs_the_soft_verifiers_last(self):
        self.assertEqual(json.loads(self.read("privateTrue", "tests", "nonrigid_case.json"))["room"], "room")
        self.assertIn("COPY case.json nonrigid_case.json /tests/", self.read("privateTrue", "tests", "Dockerfile"))
        test = self.read("privateTrue", "tests", "test.sh")
        self.assertGreater(test.index("sbv.nonrigid.verify evaluate"), test.index("sbv.evaluate"))
        self.assertIn(
            "timeout_sec = 7860.0", self.read("privateTrue", "task.toml")
        )  # 5400 + 4 x 600 + 60 for its one run
        self.assertNotIn("-m sbv.nonrigid.verify", self.read("privateFalse", "tests", "test.sh"))

    def test_thin_shell_lifts_get_grader_time_and_retired_swings_do_not(self):
        lift, swing = (
            dict(kind="thin_shell", action=dict(force_n=0.3)),
            dict(kind="thin_shell", action=dict(t_start=0.5)),
        )
        self.assertEqual(
            len(
                M.nonrigid_runs(
                    [dict(kind="fluid"), dict(kind="soft_body"), lift, dict(lift, kind="cloth"), dict(kind="size")]
                )
            ),
            4,
        )
        self.assertEqual(M.nonrigid_runs([swing, dict(swing, kind="cloth")]), [])  # an older case's swing is not run

    def test_each_run_gets_its_cap_and_one_retry_whatever_its_kind(self):
        for kind in ("fluid", "soft_body", "thin_shell"):  # cap max(600 s, 3 x the true run)
            u = dict(kind=kind, action=dict(force_n=0.3))
            self.assertEqual(M.nonrigid_seconds([dict(u, seconds=50)]), 4 * 600 + 60)
            self.assertEqual(M.nonrigid_seconds([dict(u, seconds=400)]), 4 * 1200 + 60)

    def test_the_format_check_reads_the_side_file(self):
        scene = os.path.join(self.tmp, "privateTrue", "solution", "room")

        def check(objs):
            if objs is not None:
                with open(os.path.join(scene, "nonrigid.json"), "w") as f:
                    json.dump(dict(objects=objs), f)
            elif os.path.exists(os.path.join(scene, "nonrigid.json")):
                os.remove(os.path.join(scene, "nonrigid.json"))
            r = os.path.join(self.tmp, "reward.json")
            result = subprocess.run(
                [sys.executable, os.path.join(ROOT, "checker", "check_scene.py"), scene, "--nonrigid", "--reward", r],
                capture_output=True,
            )
            with open(r) as f:
                return result.returncode, json.load(f)

        self.assertEqual(check([CUSHION]), (0, {"format_ok": 1.0, "nonrigid_ok": 1.0}))
        for invalid in ([dict(CUSHION, on="sofa__0")], None):
            with self.subTest(invalid=invalid):
                self.assertEqual(check(invalid), (1, {"format_ok": 1.0, "nonrigid_ok": 0.0}))


if __name__ == "__main__":
    unittest.main()
