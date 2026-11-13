"""Defects found by the pre-release code review: inputs a copy controls that stopped the grader instead of earning a
verdict, and a reward written without a step that had failed."""

import json, os, tempfile, unittest
from pathlib import Path
from sbv import pipeline
from sbv.functionality.f1_gate import package
from sbv.functionality.f4_robot import plan as RP
from scenebench_harbor import make_tasks as M, fair_play as FP


class RobotStages(unittest.TestCase):
    def test_a_close_step_on_a_part_already_closed_counts_its_stages(self):
        det = dict(
            success=True,
            moved=0.0,
            amount=0.0,
            done=True,
            note="already closed",
            seconds=0.0,
            on_time=True,
            released=True,
            still=True,
            others_moved={},
        )
        self.assertTrue(all(ok for _, ok in RP.stages("close", det)))
        det.pop("amount")
        det["success"] = False  # as it was returned before: no amount, and the count raised KeyError
        self.assertEqual(len(RP.stages("close", det)), len(RP.STAGES["close"]))


class Reward(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="sbv_review_")
        self.rep, self.out = os.path.join(self.tmp, "report.json"), os.path.join(self.tmp, "reward.json")

    def test_a_missing_nonrigid_report_is_a_grader_failure_not_a_room_without_nonrigid_units(self):
        pipeline.write(self.rep, dict(status="ok"))
        self.assertEqual(pipeline.reward_file(self.rep, self.out, os.path.join(self.tmp, "nonrigid_report.json")), 1)
        got = json.load(open(self.out))
        self.assertEqual(got["status"], "grader_failed")
        self.assertNotIn("reward", got)
        self.assertIn("non-rigid grader stopped", got["reason"])

    def test_appearance_the_grader_could_not_compute_is_a_grader_failure(self):
        pipeline.write(self.rep, dict(status="ok", appearance=dict(status="no PyTorch environment at /x")))
        self.assertEqual(pipeline.reward_file(self.rep, self.out), 1)
        got = json.load(open(self.out))
        self.assertEqual(got["status"], "grader_failed")
        self.assertNotIn("reward", got)
        self.assertIn("appearance could not be computed", got["reason"])

    def test_a_report_cut_short_is_no_report(self):
        with open(self.rep, "w") as f:
            f.write('{"status": "ok", "units": {')
        self.assertEqual(pipeline.reward_file(self.rep, self.out), 1)
        self.assertNotIn("reward", json.load(open(self.out)))

    def test_the_grader_script_writes_the_reward_once_after_the_last_step(self):
        self.assertNotIn("reward_file", M.GRADER_TEST + M.NONRIGID_TEST + M.PACKAGE_TEST)
        self.assertEqual(M.REWARD_TEST.count("reward_file"), 1)

    def test_a_grader_budget_under_one_is_refused(self):
        with open(os.path.join(self.tmp, "budget.json"), "w") as f:
            json.dump(dict(grader=dict(cpus=0.5)), f)
        with self.assertRaisesRegex(SystemExit, "at least 1"):
            M.budget(self.tmp, 8)


class Package(unittest.TestCase):
    def scene(self, body):
        tmp = Path(tempfile.mkdtemp(prefix="sbv_review_"))
        (tmp / "sky.png").write_bytes(b"x")
        xml = tmp / "scene.xml"
        xml.write_text(f"<mujoco>{body}<worldbody/></mujoco>")
        return str(xml)

    def test_every_path_attribute_stays_inside_the_package(self):
        self.assertTrue(package(self.scene('<asset><texture type="cube" fileright="sky.png"/></asset>'))[0])
        for bad in (
            '<asset><texture type="cube" fileright="/etc/hostname"/></asset>',
            '<asset><texture type="cube" fileup="../outside.png"/></asset>',
            '<compiler meshdir="/tests/room"/>',
            '<compiler texturedir="../.."/>',
            '<worldbody><flexcomp file="/etc/hostname"/></worldbody>',
        ):
            ok, errors = package(self.scene(bad))
            self.assertFalse(ok, bad)
            self.assertIn("leaves the package", " ".join(errors))

    def test_the_checker_has_the_same_rule(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "check_scene",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checker", "check_scene.py"),
        )
        C = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(C)
        self.assertFalse(C.package(self.scene('<compiler meshdir="/tests/room"/>'))[0])
        self.assertTrue(C.package(self.scene('<asset><texture type="cube" fileright="sky.png"/></asset>'))[0])


class NonRigidInputs(unittest.TestCase):
    def test_a_sheet_too_large_to_build_is_refused_before_it_is_built(self):
        from sbv.nonrigid.drape import draped_sheet

        with self.assertRaisesRegex(ValueError, "vertices"):
            draped_sheet([0, 0, 1], 0.01, 100000, 1000, 1)
        self.assertGreater(len(draped_sheet([0, 0, 1], 0.01, 1.0, 0.8, 0.2).faces), 0)

    def test_a_side_file_nested_too_deep_does_not_stop_the_package_tasks(self):
        from sbv.package_tasks.input_bundle import files

        tmp = Path(tempfile.mkdtemp(prefix="sbv_review_"))
        (tmp / "nonrigid.json").write_text('{"objects": ' + "[" * 100000 + "]" * 100000 + "}")
        files(tmp)  # raised RecursionError


class CheckerSideFile(unittest.TestCase):
    def test_any_side_file_gets_a_report_not_a_traceback(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "check_scene",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checker", "check_scene.py"),
        )
        C = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(C)
        tmp = Path(tempfile.mkdtemp(prefix="sbv_review_"))
        side = tmp / "nonrigid.json"
        for text in (
            "[1, 2]",
            '{"objects": [5]}',
            '{"objects": [{"name": 5}]}',
            '{"objects": [{"name": "towel__0", "kind": "thin_shell", "on": ["rack__0"], "material": "cotton", "shape": "box"}]}',
            '{"objects": [{"name": "tea__0", "kind": ["fluid"], "on": "cup__0"}]}',
            '{"objects": ' + "[" * 100000 + "]" * 100000 + "}",
        ):
            side.write_text(text)
            objs, problems = C.nonrigid_problems(str(side), {"cup__0", "rack__0"})
            self.assertTrue(problems, text[:40])
        side.write_text(
            json.dumps(
                dict(
                    objects=[
                        dict(
                            name="tea__0",
                            kind="fluid",
                            on="cup__0",
                            material=dict(viscosity=next(iter(C.VISCOSITY))),
                            shape=dict(type="cylinder"),
                        )
                    ]
                )
            )
        )
        self.assertEqual(C.nonrigid_problems(str(side), {"cup__0"})[1], [])


class RenderLimit(unittest.TestCase):
    def test_a_render_that_never_ends_is_a_failed_render(self):
        import subprocess
        from unittest import mock
        from sbv.appearance import views as App

        out = tempfile.mkdtemp(prefix="sbv_review_")
        with mock.patch.object(App.subprocess, "run", side_effect=subprocess.TimeoutExpired("blender", 1)):
            with self.assertRaisesRegex(RuntimeError, "render failed: stopped after"):
                App.render("scene.xml", dict(cameras=[]), out)
        self.assertFalse(
            os.path.exists(os.path.join(out, "render_spec.json"))
        )  # the spec never stays beside the pictures


class ScoreRules(unittest.TestCase):
    def test_a_part_that_blows_up_when_driven_passes_nothing(self):
        from sbv.functionality import f3_functions as Fn

        obs = dict.fromkeys(Fn.COMPONENTS, True)
        obs.update(rest=None, actuation=False, full_travel=False, diagnosis="unstable_rollout")
        self.assertFalse(any(Fn.joint_components(dict(expect=dict(rest=None)), obs).values()))

    def test_a_group_is_keyed_by_the_word_its_objects_are_scored_as(self):
        from sbv import groups as GR
        from sbv.common.scene import ALIAS

        alias, main = next(iter(ALIAS.items()))
        self.assertEqual(GR.word(f"{alias}__3"), main)
        self.assertEqual(GR.word("cup__0"), "cup")

    def test_a_split_scene_file_uses_the_main_file_s_mesh_folder(self):
        tmp = Path(tempfile.mkdtemp(prefix="sbv_review_"))
        (tmp / "meshes").mkdir()
        (tmp / "meshes" / "cup.stl").write_bytes(b"x")
        (tmp / "objects.xml").write_text('<mujoco><asset><mesh name="cup" file="cup.stl"/></asset></mujoco>')
        (tmp / "scene.xml").write_text(
            '<mujoco><compiler meshdir="meshes"/><include file="objects.xml"/><worldbody/></mujoco>'
        )
        self.assertEqual(package(str(tmp / "scene.xml")), (True, []))  # was: missing mesh file cup.stl

    def test_a_copy_body_named_as_a_probe_fails_the_package_check(self):
        tmp = Path(tempfile.mkdtemp(prefix="sbv_review_"))
        (tmp / "scene.xml").write_text(
            '<mujoco><worldbody><body name="probe_sofa"/><body name="sofa__0"/></worldbody></mujoco>'
        )
        ok, errors = package(str(tmp / "scene.xml"))
        self.assertFalse(ok)
        self.assertIn("probe_sofa", " ".join(errors))

    def test_rays_see_a_shape_whose_material_is_fully_transparent(self):
        import mujoco, numpy as np
        from sbv.common.scene import Scene

        tmp = Path(tempfile.mkdtemp(prefix="sbv_review_"))
        xml = tmp / "scene.xml"
        xml.write_text(
            '<mujoco><asset><material name="inv" rgba="0 0 0 0"/></asset><worldbody><geom type="plane" size="5 5 .1"/>'
            '<body name="cup__0" pos="0 0 .1"><freejoint/><geom type="box" size=".05 .05 .05" material="inv"/></body></worldbody></mujoco>'
        )
        s = Scene(str(xml), None, ["cup"])
        s.reset(s.q0)
        g = np.zeros(1, np.int32)
        self.assertAlmostEqual(
            mujoco.mj_ray(s.m, s.d, np.array([0, 0, 1.0]), np.array([0, 0, -1.0]), None, 1, -1, g), 0.85, places=3
        )  # was 1.0: through the box to the floor

    def test_new_calibrations_record_standard_probes(self):
        from sbv.common import scene as SC

        rows = SC.standard_probes()
        self.assertEqual((rows["geom"]["probe_ball#0"]["contype"], rows["geom"]["probe_ball#0"]["condim"]), (1, 3))


class JudgeVerdicts(unittest.TestCase):
    def test_a_malformed_verdict_is_no_verdict_and_a_summary_is_one_line(self):
        tmp = tempfile.mkdtemp(prefix="sbv_review_")

        def judge(folder, prediction):
            d = os.path.join(tmp, folder, "artifacts", "app")
            os.makedirs(d)
            for name, body in (("result.json", {}), ("config.json", dict(task=dict(path=f"/x/{folder}")))):
                with open(os.path.join(tmp, folder, name), "w") as f:
                    json.dump(body, f)
            with open(os.path.join(d, "prediction.json"), "w") as f:
                json.dump(prediction, f)

        good = {k: dict(verdict=False, summary="fine") for k in FP.KEYS}
        judge("j1", dict(good, refusals=dict(verdict=True, summary="x\n## Refusals")))
        judge(
            "j2", dict(good, task_level_reward_hacking=dict(verdict="false", summary=""))
        )  # a string: was read as True
        judge("j3", dict(harness_level_cheating=dict(verdict=None)))
        got = FP.verdicts([tmp], dict(j1="a/t1", j2="a/t2", j3="a/t3"))
        self.assertEqual(set(got), {"a/t1"})
        self.assertEqual(got["a/t1"]["refusals"], (True, "x ## Refusals"))


if __name__ == "__main__":
    unittest.main()
