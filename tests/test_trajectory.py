"""Actual integrator steps, not endpoint teleportation, against static obstacles."""

import tempfile, unittest
from pathlib import Path
import mujoco
import numpy as np
from sbv.common.scene import Scene
from sbv.functionality.integrity import Monitor
from sbv.functionality.f2_physical import init_passes


class Trajectories(unittest.TestCase):
    def scene(self, inner, contact=""):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = Path(tmp.name) / "scene.xml"
        p.write_text(f'<mujoco><compiler angle="radian"/>{contact}<worldbody>{inner}</worldbody></mujoco>')
        s = Scene(str(p))
        s.m.opt.gravity[:] = 0
        return s

    def translation(self, extra="", wall_x=0):
        return self.scene(
            f'<geom name="wall" type="box" pos="{wall_x} 0 1" size=".002 .5 .5" {extra}/>'
            '<body name="ball__0" pos="-.2 0 1"><freejoint/><geom name="ball" type="sphere" size=".02" mass="1"/></body>'
        )

    def run_step(self, s, velocity, **kw):
        s.d.qvel[:] = velocity
        mon = Monitor(s, [s.objects[next(iter(s.objects))]], **kw)
        s.step(0.002, mon)
        return mon

    def hinge(self, obstacle, offset=1.0):
        return self.scene(
            f'<geom type="sphere" size=".04" pos="{obstacle[0]} {obstacle[1]} 1"/>'
            f'<body name="arm__0" pos="0 0 1"><joint type="hinge" axis="0 0 1"/>'
            f'<geom type="sphere" pos="{offset} 0 0" size=".03" mass="1"/></body>'
        )

    def test_one_step_thin_wall_and_final_partial_interval(self):
        s = self.translation()
        mon = self.run_step(s, [300, 0, 0, 0, 0, 0], every=5)
        self.assertEqual(s.d.ncon, 0)
        self.assertFalse(mon.ok(mon.owners, 0.005))
        r = mon.report(mon.owners)
        self.assertTrue(r["crossed"])
        self.assertGreater(r["max_penetration"], 0.02)
        self.assertEqual(r["sweep_events"][0]["interval"], [0.0, 0.002])

    def test_offset_hinge_arc_is_not_a_chord(self):
        s = self.hinge([2**-0.5, 2**-0.5])
        mon = self.run_step(s, [np.pi / 2 / 0.002])
        self.assertFalse(mon.ok(mon.owners, 0.005))
        self.assertTrue(mon.crossed(mon.owners))
        s = self.hinge([0.5, 0.5])
        mon = self.run_step(s, [np.pi / 2 / 0.002])
        self.assertTrue(mon.ok(mon.owners, 0.005), mon.report(mon.owners))

    def test_full_revolution_not_lost_to_quaternion_shortest_arc(self):
        s = self.hinge([0, 1])
        mon = self.run_step(s, [2 * np.pi / 0.002])
        self.assertFalse(mon.ok(mon.owners, 0.005))
        self.assertTrue(mon.crossed(mon.owners))

    def test_rotating_long_rod_with_stationary_centre(self):
        s = self.scene(
            '<geom type="box" pos=".35 .35 1" size=".03 .03 .03"/>'
            '<body name="rod__0" pos="0 0 1"><freejoint/><geom type="box" size=".6 .01 .01" mass="1"/></body>'
        )
        mon = self.run_step(s, [0, 0, 0, 0, 0, np.pi / 2 / 0.002])
        self.assertFalse(mon.ok(mon.owners, 0.005))
        self.assertTrue(mon.report(mon.owners)["sweep_events"])

    def test_monitor_preserves_integrator_state_exactly(self):
        a, b = self.translation(), self.translation()
        v = [300, 0, 0, 0, 0, 0]
        a.d.qvel[:] = v
        b.d.qvel[:] = v
        mon = Monitor(a, [a.objects["ball__0"]])
        a.step(0.006, mon)
        b.step(0.006)
        for key in ("qpos", "qvel", "qacc", "ctrl", "qfrc_applied", "xfrc_applied"):
            np.testing.assert_array_equal(getattr(a.d, key), getattr(b.d, key))
        self.assertEqual(a.m.opt.timestep, b.m.opt.timestep)

    def test_clear_translation_and_visual_obstacle(self):
        for s in (self.translation(wall_x=2), self.translation('contype="0" conaffinity="0"')):
            mon = self.run_step(s, [300, 0, 0, 0, 0, 0])
            self.assertTrue(mon.ok(mon.owners, 0.005), mon.report(mon.owners))

    def test_collision_mask_and_ignore_are_respected(self):
        s = self.translation('contype="2" conaffinity="2"')
        mon = self.run_step(s, [300, 0, 0, 0, 0, 0])
        s.step(0.01, mon)
        self.assertTrue(mon.ok(mon.owners, 0.005))
        s = self.translation()
        mon = self.run_step(s, [300, 0, 0, 0, 0, 0], ignore=[0])
        self.assertTrue(mon.ok(mon.owners, 0.005))

    def test_normal_collision_stops_motion_without_tunnelling(self):
        s = self.translation()
        s.d.qvel[0] = 0.3
        mon = Monitor(s, [s.objects["ball__0"]])
        s.step(1.0, mon)
        self.assertLess(s.d.qpos[0], 0)
        self.assertTrue(mon.ok(mon.owners, 0.005), mon.report(mon.owners))

    def test_normal_floor_support_is_not_crossing(self):
        s = self.scene(
            '<geom type="plane" size="2 2 .1"/><body name="box__0" pos="0 0 .1"><freejoint/>'
            '<geom type="box" size=".1 .1 .1" mass="1"/></body>'
        )
        s.m.opt.gravity[2] = -9.81
        mon = Monitor(s, [s.objects["box__0"]])
        s.step(1.0, mon)
        self.assertTrue(mon.ok(mon.owners, 0.005), mon.report(mon.owners))

    def test_budget_exhaustion_does_not_pass(self):
        s = self.translation()
        s.d.qvel[0] = 300
        mon = Monitor(s, [s.objects["ball__0"]])
        mon.trajectory.budget = 1
        s.step(0.002, mon)
        self.assertTrue(mon.report(mon.owners)["sweep_unresolved"])
        self.assertFalse(mon.ok(mon.owners, 1.0))

    def test_report_never_reads_a_later_rollout(self):
        s = self.translation()
        mon = self.run_step(s, [300, 0, 0, 0, 0, 0])
        before = mon.report(mon.owners)
        s.reset()
        s.d.qpos[0] = 100
        s.step(0.01)
        self.assertEqual(before, mon.report(mon.owners))

    def test_crossing_is_not_forgotten_when_trace_is_full(self):
        s = self.translation()
        s.d.qvel[0] = 300
        mon = Monitor(s, [s.objects["ball__0"]])
        mon.trajectory.events = [dict(geoms=[999, 999], crossed=False, penetration=1, owner=999)] * 24
        s.step(0.002, mon)
        self.assertTrue(mon.crossed(mon.owners))
        self.assertFalse(mon.ok(mon.owners, 1.0))

    def test_free_body_full_turn_and_reverse_hinge(self):
        s = self.scene(
            '<geom type="sphere" pos="0 1 1" size=".04"/>'
            '<body name="rotor__0" pos="0 0 1"><freejoint/><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>'
            '<geom type="sphere" pos="1 0 0" size=".03"/></body>'
        )
        mon = self.run_step(s, [0, 0, 0, 0, 0, 2 * np.pi / 0.002])
        self.assertTrue(mon.crossed(mon.owners))
        s = self.hinge([2**-0.5, -(2**-0.5)])
        mon = self.run_step(s, [-np.pi / 2 / 0.002])
        self.assertTrue(mon.crossed(mon.owners))

    def test_compound_joint_bound_covers_surface_motion(self):
        from sbv.common.motion import MotionBounds, pose_path

        s = self.scene(
            '<body name="arm__0" pos="0 0 1"><joint type="hinge" pos=".2 .1 0"/>'
            '<geom type="sphere" size=".02" mass="1"/>'
            '<body pos=".3 0 0"><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/><joint type="slide" axis="1 0 0" range="-.5 .5"/>'
            '<body pos=".2 .1 0"><joint type="hinge" pos="-.1 0 0"/>'
            '<geom name="tip" type="box" pos=".3 0 0" size=".1 .02 .02" mass="1"/></body></body></body>'
        )
        g = mujoco.mj_name2id(s.m, mujoco.mjtObj.mjOBJ_GEOM, "tip")
        q = s.d.qpos.copy()
        delta = np.zeros(s.m.nv)
        delta[:3] = [1.2, 0.4, -0.8]
        bound = MotionBounds(s.m, q, [g]).travel(delta)[0]
        scratch = mujoco.MjData(s.m)
        pose = pose_path(s.m, scratch, q, delta)
        corners = np.array([[x, y, z] for x in (-0.1, 0.1) for y in (-0.02, 0.02) for z in (-0.02, 0.02)])
        pose(0)
        prev = corners @ scratch.geom_xmat[g].reshape(3, 3).T + scratch.geom_xpos[g]
        total = np.zeros(8)
        for t in np.linspace(0, 1, 201)[1:]:
            pose(t)
            cur = corners @ scratch.geom_xmat[g].reshape(3, 3).T + scratch.geom_xpos[g]
            total += np.linalg.norm(cur - prev, axis=1)
            prev = cur
        self.assertLessEqual(total.max(), bound)

    def test_all_pairs_keep_crossing_witnesses(self):
        from sbv.common.swept_depth import depth_bounds

        s = self.scene(
            '<geom name="a" type="sphere" pos=".07 0 1" size=".03"/>'
            '<geom name="b" type="sphere" pos=".72 0 1" size=".04"/>'
            '<body name="ball__0" pos="-.2 0 1"><freejoint/><geom name="moving" type="sphere" size=".02" mass="1"/></body>'
        )
        ids = [mujoco.mj_name2id(s.m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("moving", "a", "b")]
        scratch = mujoco.MjData(s.m)

        def pose(t):
            scratch.qpos[:] = s.start
            scratch.qpos[0] = -0.2 + 1.2 * t
            mujoco.mj_kinematics(s.m, scratch)

        r = depth_bounds(s.m, scratch, [(ids[0], ids[1]), (ids[0], ids[2])], [1.2, 1.2], pose)
        self.assertEqual(r["crossed"], [0, 1])
        self.assertFalse(r["unresolved"])

    def test_native_transient_contact_is_not_skipped_by_every(self):
        s = self.translation(wall_x=-0.2)
        mon = Monitor(s, [s.objects["ball__0"]], every=100)
        s.step(0.002, mon)
        self.assertGreater(mon.report(mon.owners)["max_penetration"], 0.01)
        self.assertTrue(mon.report(mon.owners)["contact_events"])

    def test_unsupported_integrator_is_explicit_uncertainty(self):
        s = self.translation()
        s.m.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
        mon = self.run_step(s, [0] * 6)
        self.assertFalse(mon.ok(mon.owners, 1.0))
        self.assertIn("unsupported_integrator_RK4", mon.report(mon.owners)["sweep_reasons"])

    def test_explicit_pair_with_zero_masks_is_still_physical(self):
        s = self.scene(
            '<geom name="wall" type="box" pos="0 0 1" size=".002 .5 .5" contype="0" conaffinity="0"/>'
            '<body name="ball__0" pos="-.2 0 1"><freejoint/>'
            '<geom name="ball" type="sphere" size=".02" mass="1" contype="0" conaffinity="0"/></body>',
            '<contact><pair geom1="wall" geom2="ball"/></contact>',
        )
        mon = self.run_step(s, [300, 0, 0, 0, 0, 0])
        self.assertFalse(mon.ok(mon.owners, 1.0))
        self.assertTrue(mon.crossed(mon.owners))

    def test_nonfinite_motion_cannot_pass_as_an_empty_sweep(self):
        s = self.translation()
        mon = Monitor(s, [s.objects["ball__0"]])
        mon.before_step()
        s.d.time += 0.002
        s.d.qvel[0] = float("nan")
        mon.tick()
        self.assertFalse(mon.ok(mon.owners, 1.0))
        self.assertIn("integration_path_not_bounded", mon.report(mon.owners)["sweep_reasons"])

    def test_uncertainty_propagates_to_initialization(self):
        r = dict(
            role="free",
            unstable=False,
            mass=True,
            inertia=True,
            com=True,
            contact=True,
            still=True,
            joints_still=True,
            mass_kg=1,
            integrity=dict(crossed=[], max_penetration=0, sweep_unresolved=True),
        )
        self.assertFalse(init_passes(dict(tol=0.01, expect=dict(role="free")), r))

    def test_stacked_boxes_landing_together_do_not_cross(self):
        book = (
            '<body name="book__{0}" pos="0 0 {1}"><freejoint/><geom type="box" size=".11 .14 .0143" mass=".72"/></body>'
        )
        s = self.scene(
            '<geom name="floor" type="plane" size="2 2 .1"/>' + book.format(0, 0.4443) + book.format(1, 0.4729)
        )
        s.m.opt.gravity[:] = [0, 0, -9.81]
        mon = Monitor(s, [s.objects["book__0"], s.objects["book__1"]])
        s.step(1.0, mon)
        self.assertLess(s.d.xpos[s.objects["book__0"]][2], 0.02)
        self.assertGreater(s.d.xpos[s.objects["book__1"]][2], 0.03)  # landed, stacked
        self.assertEqual(mon.crossed(mon.owners), [])

    def test_a_resting_box_dipping_mid_step_does_not_cross_and_a_plate_shot_through_does(
        self,
    ):
        from sbv.common.swept_depth import depth_bounds

        m = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><geom name="top" type="box" pos="0 0 .8" size=".7 .2 .0228"/>'
            '<body name="lamp__0" pos="0 0 2"><freejoint/><geom name="base" type="box" size=".06 .06 .0363"/></body>'
            "</worldbody></mujoco>"
        )
        d = mujoco.MjData(m)
        top, base = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("top", "base"))
        rest = 0.8 + 0.0228 + 0.0363

        def path(z0, z1, dip=0.0):  # the free joint's height along the step, dipping by `dip` in the middle
            def pose(t):
                d.qpos[:] = [0, 0, z0 + (z1 - z0) * t - dip * np.sin(np.pi * t), 1, 0, 0, 0]
                mujoco.mj_kinematics(m, d)

            return pose

        r = depth_bounds(m, d, [(base, top)], [0.01], path(rest + 1e-4, rest + 1e-4, dip=1e-3))
        self.assertGreater(r["observed"][0], 5e-4)
        self.assertEqual(r["crossed"], [])
        r = depth_bounds(
            m, d, [(base, top)], [1.0], path(rest + 0.001, 0.8 - 0.0228 - 0.0363 - 0.001)
        )  # straight through the top
        self.assertEqual(r["crossed"], [0])


if __name__ == "__main__":
    unittest.main()
