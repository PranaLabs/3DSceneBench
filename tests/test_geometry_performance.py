"""Compare optimized G3 with the original traversal, including rejected scenes."""

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sbv.common.scene import Scene
from sbv.geometry import g3_layout as L
from tests import rooms


def legacy_gap(scene, ga, gb, distmax=0.1, **unused):
    best = distmax
    for a in ga:
        for b in L.near(scene, [a], gb, distmax):
            best = min(best, L.pair_gap(scene, a, b, distmax))
    return best


SPEC = dict(support_tol=0.01, penetration_tol=0.002, bounds_tol=0.02, bounds=[[-3, -3, -0.1], [5, 5, 4]])


class GeometryPerformance(unittest.TestCase):
    def scene(self, contents=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(rooms.write(tmp.name))
        if contents is not None:
            path.write_text(contents)
        return Scene(str(path))

    def compare_placement(self, scene):
        corr = {n: n for n in scene.objects}
        with patch.object(L, "gap", legacy_gap):
            old = L.placement(SPEC, [], scene, corr)
        self.assertEqual(old, L.placement(SPEC, [], scene, corr))
        return old

    def test_valid_room_and_fixed_overlap(self):
        scene = self.scene()
        self.compare_placement(scene)
        scene.m.body_pos[scene.objects["cabinet__0"], 0] += 0.2
        rows = self.compare_placement(scene)["objects"]
        self.assertFalse(next(r for r in rows if r["object"] == "cabinet__0")["non_penetration"])

    def test_basket_meshes(self):
        self.compare_placement(Scene(str(Path(__file__).parent / "fixtures/uq_basket_stack.xml")))

    def test_candidates_and_distances_at_different_poses(self):
        scene = self.scene()
        rng = np.random.default_rng(4)
        gs = list(range(scene.m.ngeom))
        for _ in range(3):
            scene.d.geom_xpos[:] += rng.uniform(-0.005, 0.005, scene.d.geom_xpos.shape)
            snapshot = L.GeometrySnapshot(scene, gs)
            for margin in (0.0, 0.01, 0.2):
                for a in gs:
                    others = [g for g in gs if g != a]
                    self.assertEqual(L.near(scene, [a], others, margin), snapshot.near([a], others, margin))
                    self.assertEqual(
                        legacy_gap(scene, [a], others, margin), L.gap(scene, [a], others, margin, snapshot)
                    )
            self.assertEqual(snapshot.near([], gs, 0.1), [])

    def test_sparse_dense_planes_rotations_and_transform_count(self):
        bodies = "".join(
            f'<body name="item__{i}" pos="{(i % 8) * 0.23} {(i // 8) * 0.23} .11" euler="0 0 {i * 7}">'
            '<geom type="box" size=".1 .08 .11"/><geom type="sphere" pos=".1 0 0" size=".03"/></body>'
            for i in range(64)
        )
        scene = self.scene(
            '<mujoco><worldbody><geom type="plane" size="4 4 .1"/>'
            '<geom type="plane" zaxis="1 0 0" size="4 4 .1"/>' + bodies + "</worldbody></mujoco>"
        )
        corr = {n: n for n in scene.objects}
        times = []
        calls = []
        for fn in (legacy_gap, L.gap):
            with patch.object(L, "gap", fn), patch.object(scene, "geom_points", wraps=scene.geom_points) as count:
                start = time.perf_counter()
                result = L.placement(SPEC, [], scene, corr)
                times.append(time.perf_counter() - start)
                calls.append(count.call_count)
            if len(times) == 1:
                baseline = result
            else:
                self.assertEqual(baseline, result)
        self.assertLess(calls[1], calls[0] / 2)  # deterministic work reduction, not a timing assertion
        print(f"G3 64 objects: legacy={times[0]:.3f}s optimized={times[1]:.3f}s geom_points={calls}")
