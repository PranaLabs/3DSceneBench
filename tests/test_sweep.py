import tempfile, unittest
from pathlib import Path
import numpy as np
import mujoco
from unittest.mock import patch
from sbv.functionality import f3_functions as Fn
from sbv.functionality.units import judge
from sbv.common.scene import Scene
from sbv.functionality.f3_functions import sweep_report


class JointSweep(unittest.TestCase):
    def scene(self, x=0.15, parent=False, excluded=False, hinge=False):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        p = Path(t.name) / "s.xml"
        obstacle = f'<geom name="obstacle" type="box" pos="{x} 0 1" size=".005 .1 .1"/>'
        if hinge:
            obstacle = '<geom name="obstacle" type="box" pos=".35 .35 1" size=".02 .02 .1"/>'
        joint = (
            '<joint type="hinge" axis="0 0 1" range="0 1.570796"/>'
            if hinge
            else '<joint type="slide" axis="1 0 0" range="0 1"/>'
        )
        moving = (
            '<geom type="box" size=".6 .01 .01"/>'
            if hinge
            else '<geom name="moving_geom" type="box" size=".005 .1 .1"/>'
        )
        contact = '<contact><exclude body1="cabinet__0" body2="moving"/></contact>' if excluded else ""
        outer = "" if parent else '<body name="barrier__0">' + obstacle + "</body>"
        p.write_text(
            f'<mujoco><compiler angle="radian"/>{contact}<worldbody><body name="cabinet__0"><geom type="box" pos="-1 0 1" size=".1 .1 .1"/>{obstacle if parent else ""}<body name="moving" pos="0 0 1">{joint}{moving}</body></body>{outer}</worldbody></mujoco>'
        )
        return Scene(str(p))

    def run_sweep(self, s, **kw):
        return sweep_report(s, 0, 1, float(s.m.jnt_range[0, 1]), s.q0, **kw)

    def test_thin_obstacle_between_old_samples(self):
        r = self.run_sweep(self.scene())
        self.assertEqual(r["status"], "collision")
        self.assertGreater(r["penetration"], 0.002)

    def test_initial_ten_percent_covered(self):
        self.assertEqual(self.run_sweep(self.scene(0.035))["status"], "collision")

    def test_parent_obstruction_is_checked(self):
        self.assertEqual(self.run_sweep(self.scene(parent=True))["status"], "collision")

    def test_explicit_exclusion(self):
        self.assertEqual(self.run_sweep(self.scene(parent=True, excluded=True))["status"], "clear")

    def test_hinge_tip_collision(self):
        self.assertEqual(self.run_sweep(self.scene(hinge=True))["status"], "collision")

    def test_clear_path_and_state_preservation(self):
        s = self.scene(2)
        q = s.d.qpos.copy()
        v = s.d.qvel.copy()
        self.assertEqual(self.run_sweep(s)["status"], "clear")
        np.testing.assert_array_equal(q, s.d.qpos)
        np.testing.assert_array_equal(v, s.d.qvel)

    def test_things_beside_the_swept_path_are_not_paired(self):
        # 200 small pieces resting 5 cm above the slide's path (like coins on a chest above its drawer): never paired,
        # so they cannot use up the evaluation budget. The thin obstacle on the path is still found.
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        p = Path(t.name) / "s.xml"
        pieces = "".join(f'<geom type="box" pos="{-0.5 + 0.01 * i} 0 1.16" size=".004 .004 .004"/>' for i in range(200))
        p.write_text(
            f'<mujoco><worldbody><body name="cabinet__0"><geom type="box" pos="-1 0 1" size=".1 .1 .1"/>'
            f'<body name="moving" pos="0 0 1"><joint type="slide" axis="1 0 0" range="0 1"/><geom type="box" size=".005 .1 .1"/></body></body>'
            f'<body name="coins__0">{pieces}</body><body name="barrier__0"><geom type="box" pos=".15 0 1" size=".005 .1 .1"/></body></worldbody></mujoco>'
        )
        r = self.run_sweep(Scene(str(p)))
        self.assertEqual(r["status"], "collision")
        self.assertLessEqual(r["pair_count"], 2)

    def test_drawer_front_exactly_at_the_tolerance_is_clear(self):
        # Copy B1's chests (turned -90 degrees): side-panel inner faces at 0.376 m, drawer front 0.378 m half-wide, so
        # the front sits 2 mm into each side for its first 12 mm of travel. Rounding read 2.0000000000000573 mm, and
        # the between-sample bound could never prove "at most 2 mm"; 0.1 mm more must still fail.
        for half, status in ((0.378, "clear"), (0.3781, "collision")):
            t = tempfile.TemporaryDirectory()
            self.addCleanup(t.cleanup)
            p = Path(t.name) / "s.xml"
            p.write_text(
                '<mujoco><compiler angle="degree"/><worldbody><body name="cabinet__0" pos="0.61 2.2 0" euler="0 0 -90">'
                '<geom type="box" pos="0 -.393 .425" size=".25 .017 .335"/>'
                '<geom type="box" pos="0 .393 .425" size=".25 .017 .335"/><body name="drawer" pos="0 0 .53">'
                f'<joint type="slide" axis="1 0 0" range="0 .38"/><geom type="box" pos=".221 0 .083" size=".017 {half} .092"/>'
                "</body></body></worldbody></mujoco>"
            )
            with self.subTest(half=half):
                self.assertEqual(self.run_sweep(Scene(str(p)))["status"], status)

    def test_budget_exhaustion_is_not_a_pass(self):
        self.assertEqual(self.run_sweep(self.scene(), max_evaluations=1)["status"], "unresolved")

    def test_negative_direction_and_full_path_endpoints(self):
        for x, sgn in ((-0.15, -1), (0.0, 1), (1.0, 1)):
            with self.subTest(x=x):
                self.assertEqual(sweep_report(self.scene(x), 0, sgn, 1.0, None)["status"], "collision")

    def test_mask_disabled_and_explicit_pair_override(self):
        s = self.scene()
        g = mujoco.mj_name2id(s.m, mujoco.mjtObj.mjOBJ_GEOM, "obstacle")
        s.m.geom_contype[g] = 2
        s.m.geom_conaffinity[g] = 2
        self.assertEqual(self.run_sweep(s)["status"], "clear")
        s = self.scene(parent=True, excluded=True)
        # Compile an explicit pair as well as an exclusion: MuJoCo pairs take precedence.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "pair.xml"
        path.write_text(
            '<mujoco><contact><exclude body1="cabinet__0" body2="moving"/>'
            '<pair geom1="obstacle" geom2="moving_geom"/></contact><worldbody>'
            '<body name="cabinet__0"><geom name="obstacle" type="box" pos=".15 0 1" size=".005 .1 .1"/>'
            '<body name="moving" pos="0 0 1"><joint type="slide" axis="1 0 0" range="0 1"/>'
            '<geom name="moving_geom" type="box" size=".005 .1 .1"/></body></body></worldbody></mujoco>'
        )
        self.assertEqual(self.run_sweep(Scene(str(path)))["status"], "collision")

    def test_explicit_pair_remains_active_with_zero_masks(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = Path(tmp.name) / "s.xml"
        p.write_text(
            '<mujoco><default><geom contype="0" conaffinity="0"/></default>'
            '<contact><pair geom1="a" geom2="b"/></contact><worldbody>'
            '<geom name="a" type="box" pos=".15 0 1" size=".005 .1 .1"/>'
            '<body name="drawer__0" pos="0 0 1"><joint type="slide" axis="1 0 0" range="0 1"/>'
            '<geom name="b" type="box" size=".005 .1 .1"/></body></worldbody></mujoco>'
        )
        self.assertEqual(self.run_sweep(Scene(str(p)))["status"], "collision")

    def test_invalid_motion_is_not_clear(self):
        s = self.scene(2)
        for travel, sign in ((float("inf"), 1), (-1.0, 1), (1.0, 0), (1.0, float("nan"))):
            self.assertEqual(sweep_report(s, 0, sign, travel, None)["status"], "unresolved")

    def test_deep_nested_boxes_are_not_zero_distance(self):
        s = self.scene(0)
        s.m.geom_size[0] = [0.5, 0.5, 0.5]
        self.assertEqual(self.run_sweep(s)["status"], "collision")

    def test_force_trace_preserves_dynamics(self):
        a, b = self.scene(2), self.scene(2)
        F = 0.4
        seconds = 0.08
        travel, mon = Fn.drive(a, 0, 1, F, seconds, None, a.objects["cabinet__0"])
        b.reset()
        for _ in range(round(seconds / b.m.opt.timestep)):
            b.d.qfrc_applied[0] = F
            b.step(b.m.opt.timestep)
        b.d.qfrc_applied[0] = 0
        for field in ("qpos", "qvel", "qacc", "qfrc_applied"):
            np.testing.assert_array_equal(getattr(a.d, field), getattr(b.d, field))
        r = mon.drive_report
        self.assertEqual(r["command_force"], F)
        self.assertEqual(r["force_unit"], "N")
        self.assertAlmostEqual(r["opening_travel"], travel)
        self.assertLessEqual(len(r["samples"]), 151)
        self.assertAlmostEqual(r["samples"][-1]["time"], seconds)

    def unit(self, s):
        return dict(
            kind="joint",
            slots=["cabinet__0"],
            params=dict(ref=Fn.joint_record(s, "cabinet__0", 0), F=0.01, F_full=0.01, bars=[2.0, 1.25]),
            expect=dict(rest=True),
            hint_components=Fn.HINT_COMPONENTS,
            tol=0.01,
        )

    def test_clear_but_high_damping_is_not_geometric_obstruction(self):
        s = self.scene(2)
        s.m.dof_damping[0] = 10000.0
        u = self.unit(s)
        with patch.object(Fn, "ACT_T", 0.1), patch.object(Fn, "FULL_T", 0.1), patch.object(Fn, "REST_T", 0.1):
            obs, mons = Fn.joint_trial(s, u)
        self.assertTrue(obs["clearance"])
        self.assertFalse(obs["full_travel"])
        self.assertEqual(obs["diagnosis"], "actuation_shortfall")
        r = judge(u, obs, mons)
        self.assertTrue(r["passed"]["input_only"])
        self.assertFalse(r["passed"]["physics_hint"])
        self.assertEqual(r["details"]["drive_reports"][0]["damping"], 10000.0)
        self.assertEqual(r["details"]["clearance_report"]["status"], "clear")

    def test_blocked_opening_can_have_physically_valid_rollout(self):
        s = self.scene()
        s.m.dof_damping[0] = 1.0
        owner = s.objects["cabinet__0"]
        travel, mon = Fn.drive(s, 0, 1, 0.3, 2.0, None, owner)
        self.assertLess(travel, 0.2)
        self.assertTrue(mon.ok([owner], 0.01), mon.report([owner]))
        self.assertEqual(self.run_sweep(s)["status"], "collision")
        self.assertTrue(any(x["contacts"] for x in mon.drive_report["samples"]))

    def test_reference_rejects_collision_before_force_search(self):
        from sbv.functionality.f3_calibrate import cal_joint

        s = self.scene(parent=True)
        with patch.object(Fn, "drive", side_effect=AssertionError("force search should not run")):
            unit, reason = cal_joint(s, "cabinet__0", 0)
        self.assertIsNone(unit)
        self.assertIn("clearance", reason)
        self.assertIn("collision", reason)

    def test_unresolved_geometry_is_reported_and_fails_both_settings(self):
        s = self.scene(2)
        u = self.unit(s)
        with (
            patch.object(Fn, "sweep_report", return_value=dict(status="unresolved", reason="evaluation_budget")),
            patch.object(Fn, "ACT_T", 0.02),
            patch.object(Fn, "FULL_T", 0.02),
            patch.object(Fn, "REST_T", 0.02),
        ):
            obs, mons = Fn.joint_trial(s, u)
        r = judge(u, obs, mons)
        self.assertFalse(any(r["passed"].values()))
        self.assertEqual(r["details"]["diagnosis"], "clearance_unresolved")
