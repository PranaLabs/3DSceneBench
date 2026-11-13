"""Package tasks (sbv/package_tasks, F4.7): the room's own task code is checked on the reference room, nominal and
jittered, every declared wrong copy must fail it, its code is frozen by hash, and a copy is scored on its matched
objects."""

import json, os, shutil, tempfile, unittest
from unittest import mock
from sbv import pipeline, reward as R
from sbv.package_tasks import schema as S, verify as V
from tests import rooms

# A scripted push in two stages: the book slid at least min_move_m, then it kept its height. A book twenty times too
# heavy does not slide.
PUSH = """
import mujoco, numpy as np
from sbv.package_tasks import api

def run(ctx):
    m = mujoco.MjModel.from_xml_path(ctx["scene_xml"]); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    moved = api.jitter(m, d, ctx["seed"]) if JITTER else False
    b = api.body(m, ctx["objects"]["book__0"]); p = ctx["params"]
    for _ in range(int(0.5 / m.opt.timestep)): mujoco.mj_step(m, d)
    start = d.xpos[b].copy()
    for _ in range(int(p["push_s"] / m.opt.timestep)):
        d.xfrc_applied[b, :3] = [-p["force_n"], 0, 0]; mujoco.mj_step(m, d)
    d.xfrc_applied[b] = 0
    for _ in range(int(1.0 / m.opt.timestep)): mujoco.mj_step(m, d)
    move = float(np.linalg.norm(d.xpos[b, :2] - start[:2])); dz = float(abs(d.xpos[b, 2] - start[2]))
    return dict(stages=[["slid", bool(move >= p["min_move_m"])], ["level", bool(dz < 0.02)]], finite=bool(np.isfinite(d.qpos).all()), jitter_applied=moved,
                metrics=dict(move_m=move, dz_m=dz), why=f"moved {move:.3f} m, {dz:.3f} m up or down")
"""
PARAMS = dict(force_n=9.0, push_s=0.3, min_move_m=0.08)
HEAVY = dict(name="heavy_book", why="a book twenty times too heavy does not slide", perturb=["--mass", "book__0", "20"])


def room_with(tasks=None, code=PUSH, jitter=True):
    d = tempfile.mkdtemp()
    room = os.path.join(d, "room")
    rooms.write(room)
    os.makedirs(os.path.join(room, "package_tasks"))
    with open(os.path.join(room, "package_tasks", "push.py"), "w") as f:
        f.write(code.replace("JITTER", str(jitter)))
    task = dict(
        id="push_book",
        words="Push the book on the table toward its middle.",
        entry="package_tasks/push.py",
        objects=["book__0"],
        controller="scripted",
        params=PARAMS,
        negatives=[HEAVY],
        timeout_s=120,
    )
    with open(os.path.join(room, "tasks.json"), "w") as f:
        json.dump(dict(package=tasks or [task]), f)
    return d, room


def problems(room, **change):
    d = json.load(open(os.path.join(room, "tasks.json")))
    d["package"][0].update(change)
    with open(os.path.join(room, "tasks.json"), "w") as f:
        json.dump(d, f)
    return S.load(room)[1]


class Schema(unittest.TestCase):
    def test_package_tasks_sit_beside_the_robot_tasks_in_tasks_json(self):
        _, room = room_with()
        d = json.load(open(os.path.join(room, "tasks.json")))
        d["push"] = [dict(object="cup__0")]
        with open(os.path.join(room, "tasks.json"), "w") as f:
            json.dump(d, f)
        self.assertTrue(S.has(room))
        self.assertEqual(S.load(room)[1], [])
        from sbv.functionality.f4_robot import tasks as T

        self.assertEqual(set(T.KINDS) & set(d), {"push"})  # the robot tasks read their own families only
        os.remove(os.path.join(room, "tasks.json"))
        self.assertFalse(S.has(room))

    def test_a_good_file_has_no_problems(self):
        _, room = room_with()
        tasks, bad = S.load(room)
        self.assertEqual(bad, [])
        self.assertEqual(tasks[0]["level"], "interaction")
        self.assertEqual(tasks[0]["jitter"], 2)

    def test_what_is_refused(self):
        _, room = room_with()
        self.assertTrue(problems(room, words="Push book__0."))  # object ids in the words
        self.assertTrue(problems(room, objects=["book__9"]))  # no such object
        self.assertTrue(problems(room, negatives=[]))  # a check no wrong copy fails
        self.assertTrue(problems(room, entry="../push.py"))  # code outside the room
        self.assertTrue(problems(room, controller="robot"))  # an unknown controller
        self.assertTrue(
            problems(room, negatives=[dict(HEAVY, perturb=["--remove", "book__0"])])
        )  # a trivial wrong copy
        self.assertTrue(problems(room, jitter=1))


class WrongCopies(unittest.TestCase):
    def test_range_cuts_a_joints_travel(self):
        from sbv import perturb
        from sbv.common.scene import Scene

        room = tempfile.mkdtemp()
        rooms.write(room)
        s = Scene(perturb.main([room, os.path.join(tempfile.mkdtemp(), "copy"), "--range", "door_hinge", "0", "0.4"]))
        j = s.m.joint("door_hinge")
        self.assertEqual(s.m.jnt_range[j.id].round(4).tolist(), [0.0, 0.4])
        self.assertTrue(s.m.jnt_limited[j.id])


class Calibrate(unittest.TestCase):
    def test_the_true_room_passes_and_its_wrong_copy_fails(self):
        d, room = room_with()
        out = os.path.join(d, "results", "package_case.json")
        os.makedirs(os.path.dirname(out))
        units = V.calibrate(room, out, procs=2)
        u = units[0]
        self.assertEqual(u["id"], "P1.push_book")
        self.assertEqual(u["kind"], "package")
        self.assertEqual(u["stages"], ["slid", "level"])
        self.assertEqual(u["negatives"][0]["failed_stage"], "slid")
        self.assertEqual(u["negatives"][0]["score"], 0.0)
        self.assertGreaterEqual(u["truth"]["metrics"]["move_m"], PARAMS["min_move_m"])
        self.assertEqual(len(u["jitter"]), 2)
        self.assertLess(u["negatives"][0]["metrics"]["move_m"], PARAMS["min_move_m"])
        self.assertEqual(u["code"], ["package_tasks/push.py"])
        self.assertEqual(json.load(open(out))["room"], "../room")

    def test_a_wrong_copy_that_passes_refuses_the_room(self):
        light = dict(name="same_book", why="the same book", perturb=["--mass", "book__0", "1"])
        d, room = room_with()
        problems(room, negatives=[light])
        with self.assertRaises(SystemExit) as e:
            V.calibrate(room, os.path.join(d, "package_case.json"), procs=2)
        self.assertIn("wrong copy same_book passes the check", str(e.exception))

    def test_a_task_that_ignores_the_jitter_is_refused(self):
        d, room = room_with(jitter=False)
        with self.assertRaises(SystemExit) as e:
            V.calibrate(room, os.path.join(d, "package_case.json"), procs=2)
        self.assertIn("moved nothing", str(e.exception))

    def test_a_task_the_true_room_fails_is_refused(self):
        d, room = room_with()
        problems(room, params=dict(PARAMS, min_move_m=5.0))
        with self.assertRaises(SystemExit) as e:
            V.calibrate(room, os.path.join(d, "package_case.json"), procs=2)
        self.assertIn("the reference room fails its nominal run", str(e.exception))


class Evaluate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d, cls.room = room_with()
        cls.case = os.path.join(cls.d, "results", "package_case.json")
        os.makedirs(os.path.dirname(cls.case))
        V.calibrate(cls.room, cls.case, procs=2)

    def score(self, copy, corr):
        w = tempfile.mkdtemp()
        rep = os.path.join(w, "report.json")
        with open(rep, "w") as f:
            json.dump(dict(geometry={}, correspondence=corr), f)
        return V.evaluate(self.case, copy, rep, os.path.join(w, "package_report.json"), procs=1)["units"][0]

    def test_a_faithful_copy_passes(self):
        row = self.score(self.room, {"book__0": "book__0"})
        self.assertTrue(row["passed"])
        self.assertEqual(row["score"], 1.0)

    def test_a_copy_scores_the_stages_passed_in_order(self):
        for stages, score in (([["slid", True], ["level", False]], 0.5), ([["slid", False], ["level", True]], 0.0)):
            res = dict(
                success=False,
                stages=stages,
                score=__import__("sbv.package_tasks.runner", fromlist=["lead"]).lead(stages),
                finite=True,
                jitter_applied=False,
                metrics={},
                why="",
                seconds=0.1,
            )
            with mock.patch.object(V, "run_one", return_value=res):
                row = self.score(self.room, {"book__0": "book__0"})
            self.assertEqual((row["passed"], row["score"]), (False, score))

    def test_other_stage_names_than_the_frozen_ones_score_0(self):
        res = dict(
            success=True,
            stages=[["slid", True]],
            score=1.0,
            finite=True,
            jitter_applied=False,
            metrics={},
            why="",
            seconds=0.1,
        )
        with mock.patch.object(V, "run_one", return_value=res):
            row = self.score(self.room, {"book__0": "book__0"})
        self.assertEqual((row["passed"], row["score"]), (False, 0.0))
        self.assertIn("not the frozen", row["why"])

    def test_the_copy_is_run_on_its_matched_object(self):
        copy = os.path.join(tempfile.mkdtemp(), "copy")
        shutil.copytree(self.room, copy, ignore=shutil.ignore_patterns("package_tasks*"))
        with open(os.path.join(copy, "scene.xml")) as f:
            xml = f.read()
        with open(os.path.join(copy, "scene.xml"), "w") as f:
            f.write(xml.replace('"book__0"', '"book__7"'))
        row = self.score(copy, {"book__0": "book__7"})
        self.assertTrue(row["passed"], row)
        self.assertEqual(row["objects"], {"book__0": "book__7"})

    def test_a_heavy_copy_fails(self):
        from sbv import perturb

        copy = os.path.join(tempfile.mkdtemp(), "copy")
        perturb.main([self.room, copy, "--mass", "book__0", "20"])
        self.assertFalse(self.score(copy, {"book__0": "book__0"})["passed"])

    def test_an_unmatched_object_fails_without_a_run(self):
        copy = os.path.join(tempfile.mkdtemp(), "copy")
        shutil.copytree(self.room, copy, ignore=shutil.ignore_patterns("package_tasks*"))
        with open(os.path.join(copy, "scene.xml")) as f:
            xml = f.read()
        with open(os.path.join(copy, "scene.xml"), "w") as f:
            f.write(xml.replace('"book__0"', '"vase__0"'))  # an unpaired role binds by category word
        row = self.score(copy, {})
        self.assertFalse(row["passed"])
        self.assertIn("no object in the copy matches", row["why"])
        self.assertNotIn("seconds", row)

    def test_a_copy_that_does_not_load_fails(self):
        w = tempfile.mkdtemp()
        rep = os.path.join(w, "report.json")
        with open(rep, "w") as f:
            json.dump(dict(geometry=None), f)
        self.assertFalse(V.evaluate(self.case, self.room, rep, os.path.join(w, "p.json"))["units"][0]["passed"])

    def test_changed_task_code_stops_the_grader(self):
        d, room = room_with()
        case = os.path.join(d, "package_case.json")
        V.calibrate(room, case, procs=2)
        with open(os.path.join(room, "package_tasks", "helper.py"), "w") as f:
            f.write("X = 1\n")  # a helper beside the entry counts
        w = tempfile.mkdtemp()
        rep = os.path.join(w, "report.json")
        with open(rep, "w") as f:
            json.dump(dict(geometry={}, correspondence={"book__0": "book__0"}), f)
        with self.assertRaises(ValueError):
            V.evaluate(case, room, rep, os.path.join(w, "p.json"))


class Contract(unittest.TestCase):
    def test_the_result_is_checked_and_scored_by_stages(self):
        from sbv.package_tasks.runner import checked

        r = checked(dict(stages=[["a", True], ["b", False], ["c", True]]))
        self.assertEqual((r["success"], round(r["score"], 3)), (False, 0.333))
        self.assertEqual(checked(dict(stages=[["a", True]], finite=False))["score"], 0.0)  # a non-finite run scores 0
        for bad in (
            dict(success=True),
            dict(stages=[]),
            dict(stages=[["a", 1]]),
            dict(stages=[["a", True], ["a", True]]),
            dict(stages=[["a", False]], success=True),
        ):
            with self.assertRaises(ValueError):
                checked(bad)


class Reward(unittest.TestCase):
    def test_a_package_row_counts_in_its_level(self):
        self.assertEqual(
            R.nonrigid_part(dict(id="P1.x", kind="package", level="interaction", passed=True)), "interaction"
        )
        self.assertEqual(
            R.nonrigid_part(dict(id="P1.x", kind="package", level="affordance", passed=True)), "affordance"
        )

    def test_a_package_row_counts_its_stage_share(self):
        rep = dict(
            geometry=dict(
                G1=dict(inventory=dict(f1=1.0)),
                G2=dict(fscore=1.0, pose_size=dict(score=1.0)),
                G3=dict(relations=dict(score=1.0), placement=dict(score=1.0), navigation=dict(score=1.0)),
            ),
            levels={},
        )
        p = R.parts(rep, [dict(id="P1.x", kind="package", level="interaction", passed=False, score=0.5)])
        self.assertEqual(p["interaction"], 50.0)

    def test_a_package_task_counts_one_point_per_stage_beside_a_robot_task(self):
        """Levels are scored by points: a robot task with 6 checkpoints that passed 1, and a package task with 4 stages
        that passed half of them: (1 + 2) / (6 + 4), as a robot task's checkpoints would count."""
        g = dict(
            G1=dict(inventory=dict(f1=1.0)),
            G2=dict(fscore=1.0, pose_size=dict(score=1.0)),
            G3=dict(relations=dict(score=1.0), placement=dict(score=1.0), navigation=dict(score=1.0)),
        )
        rep = dict(geometry=g, level_points=dict(input_only=dict(interaction=[[1.0, 6.0]])))
        row = dict(id="P1.x", kind="package", level="interaction", passed=False, score=0.5, points=4)
        self.assertEqual(R.parts(rep, [row])["interaction"], 30.0)
        self.assertEqual(
            R.parts(rep, [dict(row, points=None)])["interaction"], round(100 * 1.5 / 7, 1)
        )  # a row without points: 1

    def test_a_missing_package_report_is_a_grader_failure_not_a_score(self):
        w = tempfile.mkdtemp()
        rep, out = os.path.join(w, "report.json"), os.path.join(w, "reward.json")
        with open(rep, "w") as f:
            json.dump(dict(status="ok"), f)
        self.assertEqual(pipeline.reward_file(rep, out, None, os.path.join(w, "package_report.json")), 1)
        self.assertEqual(json.load(open(out))["status"], "grader_failed")


class Harbor(unittest.TestCase):
    def test_the_private_grader_runs_the_frozen_tasks_and_the_public_task_only_names_them(self):
        from tests.harbor_fixtures import receipt
        from tests.test_budget import work_with, made

        work = work_with()
        room = os.path.join(work, "rooms", "room")
        res = os.path.join(work, "results", "room")
        _, src = room_with()
        shutil.copytree(os.path.join(src, "package_tasks"), os.path.join(room, "package_tasks"))
        shutil.copy(os.path.join(src, "tasks.json"), room)
        with open(os.path.join(res, "package_case.json"), "w") as f:
            json.dump(
                dict(
                    schema=V.CASE_SCHEMA,
                    room="../../rooms/room",
                    units=[
                        dict(id="P1.push_book", words="Push the book on the table toward its middle.", timeout_s=120)
                    ],
                ),
                f,
            )
        os.makedirs(os.path.join(res, "self_package"))
        with open(os.path.join(res, "self_package", "package_report.json"), "w") as f:
            json.dump({"units": [{"id": "P1.push_book", "success": True}]}, f)
        receipt(work)  # freeze this fixture after adding its package task and evidence
        read = made(work)
        self.assertIn("sbv.package_tasks.verify evaluate", read("private", "tests/test.sh"))
        self.assertIn("'$OUT/package_report.json'", read("private", "tests/test.sh"))
        self.assertIn("COPY case.json package_case.json /tests/", read("private", "tests/Dockerfile"))
        self.assertEqual(json.loads(read("private", "tests/package_case.json"))["room"], "room")
        self.assertIn("timeout_sec = 5580.0", read("private", "task.toml"))  # 90 min plus the task's 120 s and a minute
        for side in ("public", "private"):
            self.assertIn("| 1 | Push the book on the table toward its middle. |", read(side, "instruction.md"))
        self.assertNotIn("package_tasks", read("public", "tests/test.sh"))
        with self.assertRaises(FileNotFoundError):
            read("public", "tests/package_case.json")


class OriginalScene(unittest.TestCase):
    def test_a_task_runs_on_the_scene_exactly_as_given(self):
        code = (
            """def run(ctx):\n    text = open(ctx['scene_xml']).read()\n"""
            """    return dict(stages=[['as_given', text == ctx['params']['text']]], metrics={}, why=ctx['scene_dir'])\n"""
        )
        d, room = room_with(code=code)
        xml = os.path.join(room, "scene.xml")
        text = open(xml).read().replace("<worldbody>", '<worldbody><light pos="0 0 3"/>', 1)
        with open(xml, "w") as f:
            f.write(text)
        ctx = V.context(room, {"book__0": "book__0"}, dict(text=text), 0, [], "package_tasks/push.py")
        r = V.run_one(os.path.join(room, "package_tasks/push.py"), "mujoco", ctx, os.path.join(d, "run"), 60)
        self.assertTrue(r["success"], r)
        self.assertEqual(r["why"], os.path.abspath(room))  # the room itself, lights and all

    def test_a_copy_missing_a_declared_file_fails_the_task_not_the_grader(self):
        d, room = room_with()
        ctx = V.context(room, {"book__0": "book__0"}, PARAMS, 0, ["missing.json"], "package_tasks/push.py")
        r = V.run_one(os.path.join(room, "package_tasks/push.py"), "mujoco", ctx, os.path.join(d, "run"), 60)
        self.assertIn("task inputs", r["error"])


if __name__ == "__main__":
    unittest.main()
