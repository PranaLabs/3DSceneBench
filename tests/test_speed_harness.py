"""The speed harness, measure part (sbv/speed): the gate's comparison rules, the layer-1 decision (a mismatch turns a
mode off, and so does any nudge draw decided differently), the profile's measurements on a small room,
the copy check (a kept settle replays bit for bit) and the cache key that ignores sbv/speed."""

import json, os, tempfile, unittest
from pathlib import Path
from unittest import mock
import numpy as np
from sbv.common.scene import Scene
from sbv.speed import gate as G, layer1 as L, run as R, profile as PF
from tests import rooms

FRIES = "".join(
    f'<body name="fries__{i}" pos="{1.3 + 0.03 * (i % 4)} {1.9 + 0.03 * (i // 4)} 0.76"><freejoint/>'
    f'<geom type="box" size=".01 .01 .02" density="400"/></body>'
    for i in range(8)
)
START = (2.5, 1.0, 0.0)


def room():
    d = tempfile.mkdtemp()
    rooms.write(d, rooms.robot_room_xml().replace("</worldbody>", FRIES + "</worldbody>"))
    return d


def trial(ok=True, why=None, **det):
    return dict(
        trial="seed:1",
        ok=ok,
        success=ok,
        why=why,
        stage_points=[6, 6],
        pen=0.0,
        crossed=False,
        unresolved=False,
        ok_rules=True,
        details=det,
    )


class Same(unittest.TestCase):
    def test_a_verdict_flip_is_a_difference_and_a_small_drift_is_not(self):
        a = dict(
            ok=True,
            success=True,
            stage_points=[6, 6],
            pen=0.0021,
            details=dict(tilt_deg=3.0, miss_cm=1.0, upright=True),
        )
        self.assertEqual(G.same(a, json.loads(json.dumps(a))), [])
        b = json.loads(json.dumps(a))
        b["details"]["tilt_deg"] = 3.05
        b["pen"] = 0.0025
        self.assertEqual(G.same(a, b), [])  # within 5 % / 5e-3 and penetration within 1 mm
        b["details"]["upright"] = False
        b["stage_points"] = [4, 6]
        d = G.same(a, b)
        self.assertTrue(any("upright" in x for x in d) and any("stage_points" in x for x in d))
        b = json.loads(json.dumps(a))
        b["pen"] = 0.004
        self.assertTrue(G.same(a, b))
        b = json.loads(json.dumps(a))
        b["details"]["miss_cm"] = 1.5
        self.assertTrue(G.same(a, b))

    def test_wall_times_are_never_compared(self):
        self.assertEqual(
            G.same(
                dict(seconds=1.0, ok=True, stages=dict(build_s=3)), dict(seconds=9.0, ok=True, stages=dict(build_s=1))
            ),
            [],
        )

    def test_robot_diffs_by_task_and_trial_and_a_reason_is_a_set_of_clauses(self):
        self.assertEqual(G.robot_diffs({"a": [trial()]}, {"a": [trial()]}), {})
        self.assertIn("a seed:1", G.robot_diffs({"a": [trial()]}, {"a": [trial(False)]}))
        self.assertIn("a seed:1", G.robot_diffs({"a": [trial()]}, {"a": []}))
        x, y = (
            trial(False, "moved hose__1 2.6 cm; moved hook__1 2.6 cm"),
            trial(False, "moved hook__1 2.6 cm; moved hose__1 2.6 cm"),
        )
        self.assertEqual(G.robot_diffs({"a": [x]}, {"a": [y]}), {})  # set order, not a difference


class Decision(unittest.TestCase):
    def rep(self, **kw):
        r = dict(
            robot=dict(tasks=["a"], trials=["seed:0", "seed:1"], diffs={}, errors=[], mismatch_units=[]),
            copies=[dict(copy="c", bit_for_bit=True)],
        )
        r.update(kw)
        return r

    def test_a_layer1_mismatch_turns_frozen_trials_off_with_the_reason(self):
        d = L.decision(self.rep())
        self.assertTrue(d["robot_freeze"]["on"] and d["settle_cache"]["on"])
        r = self.rep()
        r["robot"].update(diffs={"a seed:1": ["ok: True vs False"]}, mismatch_units=["a"])
        d = L.decision(r)
        self.assertFalse(d["robot_freeze"]["on"])
        self.assertIn("ok: True vs False", d["robot_freeze"]["why"])
        d = L.decision(self.rep(copies=[dict(copy="c", bit_for_bit=False)]))
        self.assertFalse(d["settle_cache"]["on"])
        self.assertNotIn("grid", json.dumps(d))

    def test_a_nudge_draw_decided_differently_either_way_turns_frozen_trials_off(self):
        a = [dict(id="a", sigma=0.05, seed=1, passed=False), dict(id="a", sigma=0.05, seed=2, passed=True)]
        b = [dict(id="a", sigma=0.05, seed=1, passed=True), dict(id="a", sigma=0.05, seed=2, passed=True)]
        self.assertTrue(
            L.decision(self.rep(nudge=dict(plain=b, l1=b, diffs=L.nudge_diffs(b, b))))["robot_freeze"]["on"]
        )
        for plain, frozen in ((a, b), (b, a)):  # both move the objects in the reference room: no "lenient" side
            diffs = L.nudge_diffs(plain, frozen)
            self.assertEqual(list(diffs), ["a 0.05:1"])
            self.assertFalse(
                L.decision(self.rep(nudge=dict(plain=plain, l1=frozen, diffs=diffs)))["robot_freeze"]["on"]
            )
        self.assertTrue(L.nudge_diffs([dict(id="a", passed=None)], [dict(id="a", passed=True)]))  # not runnable plain

    def test_modes(self):
        u = dict(id="t", kind="robot", slots=["cup__0"], params=dict(near=[], freeze="once"))
        self.assertEqual(R.as_mode(u, "plain")["params"], dict(near=[]))
        self.assertEqual(R.as_mode(u, "l1")["params"]["freeze"], "once")
        self.assertIsNone(R.mode_env("plain", "/c")["SBV_ROBOT_SETTLE_CACHE"])
        self.assertEqual(R.mode_env("l1", "/c")["SBV_ROBOT_FREEZE"], "1")


class CacheKey(unittest.TestCase):
    def test_editing_the_harness_keeps_every_kept_settle(self):
        from sbv.common import gravity_cache as GC

        s = Scene(os.path.join(room(), "scene.xml"))
        root = Path(tempfile.mkdtemp())
        (root / "speed").mkdir()
        (root / "common").mkdir()
        (root / "speed" / "a.py").write_text("x = 1")
        (root / "common" / "scene.py").write_text("y = 1")

        def key():
            GC._CODE.clear()
            return GC.key(s, "settle")

        with mock.patch.object(GC, "ROOT", root):
            k0 = key()
            (root / "speed" / "a.py").write_text("x = 2")
            self.assertEqual(key(), k0)
            (root / "common" / "scene.py").write_text("y = 2")
            self.assertNotEqual(key(), k0)  # in the gravity stage's code (code_dependencies.STAGES)
        GC._CODE.clear()


class Profile(unittest.TestCase):
    def test_collision_complexity_and_a_sampled_settle(self):
        from sbv.functionality.f4_robot import plan as RP

        if not RP.installed():
            self.skipTest("needs the RB-Y1 model")
        w = RP.World(os.path.join(room(), "scene.xml"), START, textures=False)
        cc = PF.collision_complexity(w)
        self.assertEqual(cc["fries__0"]["geoms"], 1)
        self.assertNotIn("rby1_base", cc)
        ss = PF.sampled_settle(w, 0.1)
        self.assertEqual(ss["steps"], 50)
        self.assertGreater(ss["contacts"]["table__0"], 0)
        self.assertIn("box-box", ss["pairs"])
        self.assertTrue(set(ss["contacts"]) >= {f"fries__{i}" for i in range(8)})

    def test_a_zone_settle_wakes_the_task_zone_only(self):
        from sbv.functionality.f4_robot import plan as RP

        if not RP.installed():
            self.skipTest("needs the RB-Y1 model")
        w = RP.World(os.path.join(room(), "scene.xml"), START, textures=False)
        u = dict(id="robot|cup__0|cabinet__0", slots=["cup__0", "cabinet__0"], params=dict(near=["table__0"]))
        with mock.patch.dict(w.phys, settle_s=0.2):
            z = PF.zone_settle(w, u, set())
        self.assertIn("fries__0", z["zone"])
        self.assertEqual(z["restless"], [])
        self.assertLessEqual(z["awake_trees_max"], w.m.ntree)

    def test_a_copy_replays_its_kept_settle_bit_for_bit(self):
        from sbv.functionality.f4_robot import plan as RP

        if not RP.installed():
            self.skipTest("needs the RB-Y1 model")
        case = dict(geometry=dict(navigation=dict(start=[2.5, 1.0])), physics=dict(settle_s=0.2))
        r = L.copy_replay(room(), case, log=lambda *a: None)
        self.assertTrue(r["bit_for_bit"], r)


if __name__ == "__main__":
    unittest.main()
