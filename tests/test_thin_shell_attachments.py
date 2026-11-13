"""Real MJCF support geometry with a minimal particle entity: no Genesis required for the track contract."""

import os
import tempfile
import unittest
import numpy as np
from sbv.nonrigid.attachments import RingTrack, problems


class Entity:
    def fix_particles(self, ids):
        self.fixed = ids

    def set_particles_vel(self, vel, particles_idx_local):
        self.velocity = vel
        self.ids = particles_idx_local


class Clips(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.xml = os.path.join(self.tmp.name, "scene.xml")
        self.obj = dict(
            kind="thin_shell", on="curtain_rod__0", attachments=dict(type="sliding_ring_clips", count=5, axis=[1, 0, 0])
        )
        self.q = np.array(
            [[x, y, z] for z in [1.95, 1.6, 1.2] for x in np.linspace(-0.4, 0.4, 21) for y in [-0.005, 0.005]]
        )
        self.write_rod()

    def write_rod(self, z=2.0, shape="capsule"):
        rod = '<geom type="%s" size=".01 .7%s" %s/>' % (
            shape,
            " .01" if shape == "box" else "",
            "" if shape == "box" else 'quat=".70710678 0 .70710678 0"',
        )
        with open(self.xml, "w") as f:
            f.write(
                '<mujoco><worldbody><body name="curtain_rod__0" pos="0 0 %s">%s</body></worldbody></mujoco>' % (z, rod)
            )

    def test_only_header_is_constrained_and_external_nudge_cannot_unhook_it(self):
        e = Entity()
        t = RingTrack(e, self.q, self.xml, self.obj)
        self.assertLess(len(t.ids), len(self.q) // 2)
        self.assertTrue(np.all(self.q[t.ids, 2] == 1.95))
        t.apply(self.q, np.ones_like(self.q))
        self.assertTrue(np.all(e.velocity == 0))
        self.assertEqual(t.report(self.q)["clips"], 5)

    def test_draw_is_bounded_velocity_not_position_reset_and_stays_on_actual_rod(self):
        e = Entity()
        t = RingTrack(e, self.q, self.xml, self.obj)
        initial = self.q.copy()
        t.apply(self.q, np.zeros_like(self.q), np.linspace(0, 0.2, 5))
        np.testing.assert_array_equal(self.q, initial)
        self.assertLessEqual(np.linalg.norm(e.velocity, axis=1).max(), 0.01000001)
        self.assertTrue(np.allclose(e.velocity[:, 1:], 0))
        with self.assertRaisesRegex(ValueError, "beyond"):
            t.apply(self.q, np.zeros_like(self.q), np.ones(5) * 2)

    def test_a_floating_or_missing_rod_does_not_create_magic_support(self):
        self.write_rod(z=2.5)
        with self.assertRaisesRegex(ValueError, "header"):
            RingTrack(Entity(), self.q, self.xml, self.obj)
        self.write_rod(shape="box")
        with self.assertRaisesRegex(ValueError, "collidable"):
            RingTrack(Entity(), self.q, self.xml, self.obj)

    def test_declaration_validation(self):
        for change in [dict(count=0), dict(axis=[0, 0, 0]), dict(axis=["bad"]), dict(type="glue")]:
            self.assertTrue(problems(dict(self.obj, attachments={**self.obj["attachments"], **change})))

    def test_solver_domain_contains_a_substantial_draw_in_either_direction(self):
        from sbv.nonrigid.world import cloth_domain

        for sign in [-1, 1]:
            lo, hi = cloth_domain([0, 0, 2], [-0.4, -0.1, 0], [0.4, 0.1, 2], dict(offsets=[0, sign * 1.6]), [1, 0, 0])
            self.assertLess(lo[0], min(-0.4, -0.4 + sign * 1.6))
            self.assertGreater(hi[0], max(0.4, 0.4 + sign * 1.6))

    def test_track_follows_a_translated_rotated_rod_instead_of_room_coordinates(self):
        with open(self.xml, "w") as f:
            f.write(
                '<mujoco><worldbody><body name="curtain_rod__0" pos="1 .4 2">'
                '<geom type="capsule" size=".01 .7" quat=".70710678 -.70710678 0 0"/>'
                "</body></worldbody></mujoco>"
            )
        q = self.q.copy()
        q[:, 0] = 1 - self.q[:, 1]
        q[:, 1] = 0.4 + self.q[:, 0]
        obj = dict(self.obj, attachments={**self.obj["attachments"], "axis": [0, 1, 0]})
        t = RingTrack(Entity(), q, self.xml, obj)
        self.assertAlmostEqual(t.report(q)["max_support_gap_m"], 0.05, places=4)
        np.testing.assert_allclose(t.axis, [0, 1, 0], atol=1e-7)

    def test_a_clip_code_change_runs_attached_and_ordinary_cloth_again(self):
        """Kept cloth runs are keyed by the whole non-rigid package's code (code_dependencies.native('nonrigid'),
        attachments.py and world.py included): the same code reuses both curtains, and a clip code change runs both
        again (the cache used to key only clipped cloth on the clip code)."""
        from unittest.mock import patch
        import subprocess
        from sbv.nonrigid import verify
        from sbv.common import code_dependencies as CD, dependencies as D

        done = subprocess.CompletedProcess([], 0, stdout='RESULT {"readout":0.1,"finite":true}', stderr="")
        real = CD.native
        attached = dict(self.obj, name="curtain__0", material=dict(fabric="cotton"))
        ordinary = {k: v for k, v in attached.items() if k != "attachments"}
        with (
            patch.object(verify.subprocess, "run", return_value=done) as calls,
            patch.object(D, "runtime", lambda *a, **k: {}),
        ):
            for version in ["v1", "v1", "v2"]:
                with patch.object(
                    CD,
                    "native",
                    lambda stage, *a, v=version, **k: dict(real(stage, *a, **k), **{"nonrigid/attachments.py": v}),
                ):
                    verify._one(self.xml, attached, {}, None, os.path.join(self.tmp.name, "attached", "run"))
                    verify._one(self.xml, ordinary, {}, None, os.path.join(self.tmp.name, "ordinary", "run"))
            self.assertEqual(calls.call_count, 4)


if __name__ == "__main__":
    unittest.main()
