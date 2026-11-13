"""SBV_GRAVITY_CACHE: the same case and report with the cache off, cold and warm; every input of the compiled model, the
physics settings and the code are in the key; a run over the F1 budget is not kept."""

import json, os, shutil, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from sbv import calibrate, evaluate
from sbv.common import case as K, gravity_cache as G
from sbv.common.scene import Scene, PHYSICS
from sbv.functionality import f1_gate as F1, f2_physical as P
from tests import rooms

TET = "v 0 0 0\nv .1 0 0\nv 0 .1 0\nv 0 0 .1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"


def untimed(x):
    if isinstance(x, dict):
        return {k: untimed(v) for k, v in x.items() if k not in ("seconds", "seconds_by_part")}
    return [untimed(v) for v in x] if isinstance(x, list) else x


class GravityCache(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.cache = self.tmp / "cache"

    def check(self, room, folder):
        path = self.tmp / "case.json"
        with patch.dict(os.environ, SBV_GRAVITY_CACHE=str(folder)):
            case = calibrate.build(
                room, 2, 2, 2, 1, log=lambda *a: None, robot_tasks=0, push_tasks=0, require_tasks=False, camera_seed=7
            )
            case["reference"] = K.scene_xml(room)
            case["calibration"]["seconds"] = 0  # timing, and with it case_hash
            K.save(case, path)
            report = evaluate.evaluate(K.load(path), room, 2, log=lambda *a: None)
        return path.read_text() + json.dumps(
            untimed(report), default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)
        )

    def test_off_cold_and_warm_give_the_same_case_and_report(self):
        room = os.path.dirname(rooms.write(str(self.tmp / "room")))
        off, cold = self.check(room, "off"), self.check(room, self.cache)
        kept = sorted(p.name.split("-")[0] for p in self.cache.iterdir())
        self.assertEqual(
            kept, ["f1", "init", "init", "settle"]
        )  # nominal and 1 jitter; scoring's seed 0 is the nominal
        with (
            patch.object(F1, "load_and_step", side_effect=AssertionError("F1 stepped")),
            patch.object(P, "_init_record", side_effect=AssertionError("init rolled out")),
        ):
            warm = self.check(room, self.cache)
        self.assertEqual(len(list(self.cache.iterdir())), 4)
        self.assertEqual(off, cold)
        self.assertEqual(off, warm)

    def test_mesh_include_physics_and_code_are_in_the_key(self):
        (self.tmp / "shared").mkdir()
        mesh = self.tmp / "shared" / "tet.obj"
        mesh.write_text(TET)  # outside the room's folder
        part = self.tmp / "room" / "part.xml"
        part.parent.mkdir()
        part.write_text(
            '<mujoco><body name="box__0" pos="0 0 .2"><freejoint/><geom type="box" size=".1 .1 .1"/></body></mujoco>'
        )
        xml = rooms.write(
            str(self.tmp / "room"),
            '<mujoco><compiler meshdir="../shared"/><asset><mesh file="tet.obj"/><texture name="sky" '
            'type="skybox" builtin="gradient" width="64" height="64"/></asset><worldbody><geom type="plane" size="2 2 .1"/>'
            '<include file="part.xml"/><body name="tet__0" pos="1 0 .1"><freejoint/><geom type="mesh" mesh="tet"/></body></worldbody></mujoco>',
        )
        key = lambda **kw: G.key(Scene(xml, **kw), "settle")
        keys = [key(), key(textures=False)]
        self.assertEqual(keys[0], keys[1])  # the pictures are not physics

        def edit(p, a, b):  # a new mtime: MuJoCo's own asset cache goes by it
            t = p.stat().st_mtime_ns + 2 * 10**9
            p.write_text(p.read_text().replace(a, b))
            os.utime(p, ns=(t, t))

        edit(mesh, "v .1 0 0", "v .2 0 0")
        keys.append(key())
        edit(part, 'size=".1 .1 .1"', 'size=".1 .1 .2"')
        keys.append(key())
        keys.append(key(phys=dict(PHYSICS, settle_s=4.0)))
        keys.append(key(phys=dict(PHYSICS, timestep=0.001)))
        code = self.tmp / "code"
        shutil.copytree(G.ROOT, code, ignore=shutil.ignore_patterns("__pycache__"))
        with open(code / "functionality" / "integrity.py", "a") as f:
            f.write("\n# changed\n")
        with patch.object(G, "ROOT", code):
            keys.append(key())
        self.assertEqual(len(set(keys[1:])), len(keys) - 1)

    def test_a_run_over_the_budget_is_not_kept(self):
        xml = rooms.write(str(self.tmp / "room"))
        with patch.dict(os.environ, SBV_GRAVITY_CACHE=str(self.cache)):
            ok, rep, _ = F1.gate(xml, dict(PHYSICS), budget_s=1e-6)
            self.assertFalse(ok)
            self.assertIn("budget", rep["load_and_step"][0])
            self.assertFalse(self.cache.exists())
            first = F1.gate(xml, dict(PHYSICS))[1]
            with patch.object(F1, "load_and_step", side_effect=AssertionError("stepped again")):
                self.assertEqual(F1.gate(xml, dict(PHYSICS))[1], first)
        with patch.dict(os.environ, SBV_GRAVITY_CACHE=str(xml)):
            Scene(xml).settle()  # a file: unreadable and unwritable
        with (
            patch.dict(os.environ, SBV_GRAVITY_CACHE="off"),
            patch.object(G, "key", side_effect=AssertionError("hashed")),
        ):
            Scene(xml).settle()


if __name__ == "__main__":
    unittest.main()
