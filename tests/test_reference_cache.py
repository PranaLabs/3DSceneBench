import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sbv.common.scene import Scene
from sbv.geometry import g2_shape as G, reference_cache as C
from tests import rooms


class ReferenceCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name) / "cache"
        env = patch.dict(os.environ, SBV_GEOMETRY_CACHE=str(self.folder))
        env.start()
        self.addCleanup(env.stop)
        self.path = rooms.write(str(Path(self.tmp.name) / "room"))
        self.scene = Scene(self.path)
        self.scene.reset(self.scene.q0)
        self.slots = [dict(slot=n) for n in self.scene.objects]

    def test_exact_scores_and_no_sampling_on_hit(self):
        old = {n: G.surface_samples(self.scene, n, 3000) for n in self.scene.objects}
        cold = C.samples(self.scene, self.slots, 3000)
        with patch.object(G, "surface_samples", side_effect=AssertionError("cache miss")):
            warm = C.samples(Scene(self.path), self.slots, 3000)
        for n, a in old.items():
            np.testing.assert_array_equal(a, cold[n])
            np.testing.assert_array_equal(a, warm[n])
            self.assertEqual(G.shape_scores(a, a * 0.9, 0.03), G.shape_scores(warm[n], a * 0.9, 0.03))

    def test_model_pose_count_and_corrupt_cache_invalidation(self):
        C.samples(self.scene, self.slots, 20)
        C.samples(self.scene, self.slots, 30)
        self.scene.d.geom_xpos[:, 0] += 0.1
        C.samples(self.scene, self.slots, 20)
        self.scene.m.geom_size[self.scene.ogeoms[self.slots[0]["slot"]][0], 0] += 0.01
        C.samples(self.scene, self.slots, 20)
        self.assertEqual(len(list(self.folder.glob("*.npz"))), 4)
        for p in self.folder.glob("*.npz"):
            p.write_bytes(b"broken")
        with patch.object(G, "surface_samples", wraps=G.surface_samples) as sample:
            C.samples(self.scene, self.slots, 20)
            self.assertEqual(sample.call_count, len(self.slots))
        with patch.object(G, "surface_samples", side_effect=AssertionError("cache miss")):
            C.samples(self.scene, self.slots, 20)

    def test_cache_optional_and_unwritable(self):
        # A file cannot be a cache directory, even when tests run as root.
        blocked = Path(self.tmp.name) / "blocked"
        blocked.touch()
        for folder in ("off", str(blocked)):
            with (
                patch.dict(os.environ, SBV_GEOMETRY_CACHE=folder),
                patch.object(G, "surface_samples", wraps=G.surface_samples) as sample,
            ):
                for _ in range(2):
                    C.samples(self.scene, self.slots, 20)
                self.assertEqual(sample.call_count, 2 * len(self.slots))

    def test_key_ignores_textures_and_never_serializes_model(self):
        xml = """<mujoco><asset><texture name="t" type="2d" builtin="checker" width="512" height="512"/>
        <material name="m" texture="t"/></asset><worldbody><body name="box__0">
        <geom type="box" size=".1 .1 .1" material="m"/></body></worldbody></mujoco>"""
        scene = Scene(rooms.write(str(Path(self.tmp.name) / "textured"), xml), textures=True)
        names = list(scene.objects)
        with patch.object(C.mujoco, "mj_saveModel", side_effect=AssertionError("serialized model")):
            first = C.cache_key(scene, names, 20)
            scene.m.tex_data[:] = 42
            self.assertEqual(first, C.cache_key(scene, names, 20))
            scene.m.geom_size[0, 0] *= 2
            self.assertNotEqual(first, C.cache_key(scene, names, 20))

    def test_default_has_no_cache_overhead(self):
        with patch.dict(os.environ), patch.object(C, "cache_key", side_effect=AssertionError("hashed")):
            os.environ.pop("SBV_GEOMETRY_CACHE", None)
            C.samples(self.scene, self.slots, 20)

    def test_mesh_content_and_visibility_invalidate(self):
        xml = """<mujoco><asset><mesh name="mesh" vertex="0 0 0 .1 0 0 0 .1 0 0 0 .1"
        face="0 2 1 0 1 3 0 3 2 1 2 3"/></asset><worldbody><body name="mesh__0">
        <geom type="mesh" mesh="mesh"/><geom type="box" size=".1 .1 .1" group="3"/>
        </body></worldbody></mujoco>"""
        scene = Scene(rooms.write(str(Path(self.tmp.name) / "mesh"), xml))
        names = list(scene.objects)
        key = lambda: C.cache_key(scene, names, 20)
        a = key()
        scene.m.mesh_vert[0, 0] += 0.001
        b = key()
        self.assertNotEqual(a, b)
        scene.m.mesh_face[0] = scene.m.mesh_face[0][::-1]
        c = key()
        self.assertNotEqual(b, c)
        scene.m.geom_group[1] = 0
        self.assertNotEqual(c, key())
