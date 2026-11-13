"""Cloth physics in the non-rigid verifiers: a flat cloth lying on a bench, and a curved one standing on it, start at
rest (friction with rigid bodies, bending about the starting shape, settling into contact); three sheets crossing each
other are refused at their start (SX.7). Slow: Genesis, with
SBV_SLOW=1 and SBV_NONRIGID_PYTHON pointing at a Python with requirements/nonrigid.txt.
    python -m unittest tests.test_thin_shell_physics -v
"""

import os, shutil, sys, tempfile, unittest
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from sbv.nonrigid import verify, world  # noqa: E402
from tests.test_nonrigid import genesis_ok  # noqa: E402

BENCH = (
    '<mujoco><option timestep="0.002"/><worldbody><geom name="floor" type="plane" size="0 0 0.1"/>'
    '<body name="bench__0" pos="1 1 0"><geom name="bench_top" type="box" pos="0 0 0.43" size="0.4 0.3 0.02"/>'
    '<geom name="bench_leg" type="box" pos="0 0 0.2" size="0.05 0.05 0.2"/></body></worldbody></mujoco>'
)


def sheet(path, layers=1, slope=0.0):
    """A 36 x 24 cm grid of triangles; a second layer, if any, rises `slope` per metre along y through the first."""
    verts, faces = [], []
    for k in range(layers):
        base = len(verts)
        verts += [(x, y, k * slope * y) for y in np.linspace(-0.12, 0.12, 9) for x in np.linspace(-0.18, 0.18, 13)]
        faces += [
            (base + r * 13 + c + 1, base + r * 13 + c + 2, base + (r + 1) * 13 + c + 2)
            for r in range(8)
            for c in range(12)
        ]
        faces += [
            (base + r * 13 + c + 1, base + (r + 1) * 13 + c + 2, base + (r + 1) * 13 + c + 1)
            for r in range(8)
            for c in range(12)
        ]
    with open(path, "w") as f:
        f.write("".join(f"v {x} {y} {z}\n" for x, y, z in verts) + "".join(f"f {a} {b} {c}\n" for a, b, c in faces))


def arch(path, r=0.05):
    """A half tube of cloth, 30 cm long, standing on its two long edges."""
    th, xs = np.linspace(0, np.pi, 17), np.linspace(-0.15, 0.15, 21)
    verts = [(x, r * np.cos(t), r * np.sin(t)) for x in xs for t in th]
    faces = [
        f
        for i in range(20)
        for j in range(16)
        for f in (
            (i * 17 + j + 1, (i + 1) * 17 + j + 1, i * 17 + j + 2),
            (i * 17 + j + 2, (i + 1) * 17 + j + 1, (i + 1) * 17 + j + 2),
        )
    ]
    with open(path, "w") as f:
        f.write("".join(f"v {x} {y} {z}\n" for x, y, z in verts) + "".join(f"f {a} {b} {c}\n" for a, b, c in faces))


class Domain(unittest.TestCase):
    def test_the_solver_box_holds_the_whole_garment_even_with_a_wrong_rail(self):
        lo, hi = world.cloth_domain([0, 0, 1.0], [2.0, 2.0, 0.2], [2.5, 2.5, 1.4])
        self.assertTrue(np.all(lo <= [1.9, 1.9, 0.1]) and np.all(hi >= [2.6, 2.6, 1.5]))
        self.assertTrue(world.touches_domain([[0, 0, 0]], [0, -1, -1], [1, 1, 1]))
        self.assertFalse(world.touches_domain([[0.5, 0, 0]], [0, -1, -1], [1, 1, 1]))

    def test_a_cached_cloth_run_is_not_reused_once_the_fabrics_change(self):
        """A finished cloth run is kept while its inputs are the same. The fabric table is in nonrigid/schema.py, part
        of the non-rigid code the key hashes (code_dependencies.native('nonrigid')), so editing it runs the cloth
        again."""
        from unittest.mock import patch
        import subprocess
        from sbv.common import code_dependencies as CD, dependencies as D

        d = tempfile.mkdtemp(prefix="sbv_cloth_cache_")
        xml = os.path.join(d, "scene.xml")
        real = CD.native
        with open(xml, "w") as f:
            f.write('<mujoco><worldbody><body name="rod__0"><geom size=".1"/></body></worldbody></mujoco>')
        towel = dict(name="towel__0", kind="thin_shell", material=dict(fabric="cotton"))
        done = subprocess.CompletedProcess([], 0, stdout='RESULT {"readout": 0.1, "finite": true}', stderr="")
        with (
            patch.object(verify.subprocess, "run", return_value=done) as runs,
            patch.object(D, "runtime", lambda *a, **k: {}),
        ):
            for table in (
                "cotton (0.15, 560.0)",
                "cotton (0.15, 560.0)",
                "cotton (0.15, 1e-4)",
            ):  # the same twice, then edited
                with patch.object(
                    CD,
                    "native",
                    lambda stage, *a, t=table, **k: dict(real(stage, *a, **k), **{"nonrigid/schema.py": t}),
                ):
                    verify._one(xml, towel, {}, None, os.path.join(d, "run"))
            self.assertEqual(runs.call_count, 2)
        shutil.rmtree(d, ignore_errors=True)


@unittest.skipUnless(
    os.environ.get("SBV_SLOW") == "1" and genesis_ok(),
    "slow: set SBV_SLOW=1 and SBV_NONRIGID_PYTHON to a Python with requirements/nonrigid.txt",
)
class GenesisRun(unittest.TestCase):
    def test_flat_and_curved_cloth_start_at_rest_and_a_tangled_mesh_is_refused(self):
        d = tempfile.mkdtemp(prefix="sbv_cloth_")
        xml = os.path.join(d, "scene.xml")
        with open(xml, "w") as f:
            f.write(BENCH)
        sheet(os.path.join(d, "flat.obj"))
        sheet(os.path.join(d, "tangled.obj"), layers=3, slope=0.15)
        arch(os.path.join(d, "arch.obj"))
        towel = dict(
            name="towel__0",
            kind="thin_shell",
            on="bench__0",
            shape=dict(type="mesh", file="flat.obj", pos=[1.0, 1.0, 0.46], rail=[1.0, 1.0, 0.46]),
            material=dict(fabric="cotton"),
        )
        a = verify.default_action("thin_shell", towel["name"], 0)
        r = verify._one(xml, towel, a, None, os.path.join(d, "flat"))
        self.assertNotIn("error", r, r)
        self.assertTrue(r["finite"])
        lying = dict(
            towel, shape=dict(towel["shape"], pos=[1.0, 1.0, 0.45], rail=[1.0, 1.0, 0.45])
        )  # on the bench top, no gap
        r = verify._one(xml, lying, a, None, os.path.join(d, "lying"))
        self.assertNotIn("error", r, r)
        arched = dict(
            towel,
            shape=dict(towel["shape"], file="arch.obj", pos=[1.0, 1.0, 0.4575]),
            material=dict(fabric="wool_coat"),
        )
        r = verify._one(
            xml, arched, a, None, os.path.join(d, "arch")
        )  # bent about its own shape, not toward flat (1.1 m/s)
        self.assertNotIn("error", r, r)
        r = verify._one(
            xml, dict(towel, shape=dict(towel["shape"], file="tangled.obj")), a, None, os.path.join(d, "tangled")
        )
        self.assertIn("not at rest", r.get("error", ""), r)
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
