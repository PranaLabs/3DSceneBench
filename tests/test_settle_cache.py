"""The two robot-trial speed switches (f4_robot/plan.py). SBV_ROBOT_SETTLE_CACHE keeps a
whole-room settle and replays it bit for bit; frozen trials (SBV_ROBOT_FREEZE=1 at calibration, params.freeze in the
case) restore that settle for a trial that changes nothing for its task (the nominal run and each jitter seed, each
from its own kept settle), so it is the same, bit for bit, as before, and settle only the task's zone, what the
whole-room settle moves and what touches them for an edited or nudged trial, from the file's start as before. Off by
default: nothing is kept and every trial settles as before."""

import os, tempfile, unittest
from unittest import mock
import numpy as np, mujoco
from sbv import calibrate as CB
from sbv.common.scene import Scene, PHYSICS
from sbv.functionality.f4_robot import plan as RP, pick_place
from tests import rooms

FAR = '<body name="crate__1" pos="0.5 3.5 0.15"><freejoint/><geom type="box" size=".15 .15 .15"/></body>'
# 1 cm above the crate
BALL = '<body name="ball__0" pos="0.5 3.5 0.36"><freejoint/><geom type="sphere" size=".05"/></body>'
# falls onto the cup
HIGH = '<body name="crate__2" pos="1.85 2.0 2.6"><freejoint/><geom type="box" size=".1 .1 .1" density="800"/></body>'
START = (2.5, 1.0, 0.0)


def state(w):
    d = w.d
    return [d.qpos.copy(), d.qvel.copy(), d.qacc.copy(), d.xipos.copy(), d.qacc_warmstart.copy(), d.time, d.ncon]


@unittest.skipUnless(RP.installed(), "needs the RB-Y1 model (scripts/get_robot.sh)")
class SettleCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.xml = rooms.write(
            tempfile.mkdtemp(), rooms.robot_room_xml().replace("</worldbody>", FAR + BALL + "</worldbody>")
        )

    def setUp(self):
        RP._KEPT.clear()
        env = {k: v for k, v in os.environ.items() if k not in (RP.CACHE_ENV, RP.FREEZE_ENV)}
        p = mock.patch.dict(os.environ, env, clear=True)
        p.start()
        self.addCleanup(p.stop)

    def settled(self, seed=1):
        w = RP.World(self.xml, START, textures=False)
        w.jitter(seed)
        x = w.settle_steps()
        return w, x

    def same(self, a, b):
        for x, y in zip(a, b):
            self.assertTrue(np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y)

    def test_off_keeps_nothing(self):
        self.settled()
        self.assertEqual(RP._KEPT, {})

    def test_a_kept_settle_is_the_settle_bit_for_bit(self):
        plain, (x0, x1) = self.settled()
        for where in ("mem", tempfile.mkdtemp()):
            RP._KEPT.clear()
            os.environ[RP.CACHE_ENV] = where
            self.settled()
            if where != "mem":
                RP._KEPT.clear()
                self.assertEqual(len([f for f in os.listdir(where) if f.endswith(".json")]), 1)
            w, (y0, y1) = self.settled()  # replayed: from this process or the disk
            self.same(state(plain), state(w))
            self.same([x0, x1], [y0, y1])
            self.settled(seed=2)
            self.assertEqual(len(RP._KEPT), 2)  # another jitter is another key
        for _ in range(200):
            mujoco.mj_step(plain.m, plain.d)
            mujoco.mj_step(w.m, w.d)
        self.same(state(plain), state(w))  # and it goes on the same

    def test_frozen_settle_moves_only_the_zone_and_wakes_on_contact(self):
        w = RP.World(self.xml, START, textures=False)
        zone = RP.around(w, ["cup__0"])
        tree = lambda n: w.m.body_treeid[w.objects[n]]
        self.assertEqual(zone, {"cup__0", "table__0"})  # the crate 3 m off is not near the cup
        c0 = w.d.xipos[w.objects["crate__1"]].copy()
        w.freeze(zone)
        for _ in range(20):
            mujoco.mj_step(w.m, w.d)
        self.assertGreaterEqual(w.d.tree_asleep[tree("crate__1")], 0)
        self.assertTrue(np.array_equal(w.d.xipos[w.objects["crate__1"]], c0))
        self.assertTrue(
            all(w.d.tree_asleep[tree(n)] < 0 for n in zone)
            and all(w.d.tree_asleep[w.m.body_treeid[b]] < 0 for b in w.robot if w.m.body_treeid[b] >= 0)
        )
        w = RP.World(self.xml, START, textures=False)
        w.freeze({"ball__0"})  # the ball falls 1 cm onto the sleeping crate
        for _ in range(200):
            mujoco.mj_step(w.m, w.d)
        self.assertLess(w.d.tree_asleep[tree("crate__1")], 0)  # touched: awake

    def test_the_case_decides_and_the_environment_only_turns_it_off(self):
        u = dict(params=dict(freeze="once"))
        self.assertEqual(RP.mode(u), "once")
        self.assertIsNone(RP.mode(dict(params={})))
        self.assertFalse(RP.requested())
        self.assertEqual(RP.mode(dict(params=dict(freeze="file"))), "once")  # an earlier draft's label
        os.environ[RP.FREEZE_ENV] = "1"
        self.assertIsNone(RP.mode(dict(params={})))  # a case calibrated awake is scored awake
        self.assertTrue(RP.requested())  # but calibrate and add_tasks record it
        os.environ[RP.FREEZE_ENV] = "0"
        self.assertIsNone(RP.mode(u))

    def test_settled_once_the_nominal_trial_is_the_same_and_the_others_still_pass(self):
        s = Scene(self.xml, PHYSICS, probes=True, textures=False)
        u = dict(
            id="robot|cup__0|cabinet__0",
            kind="robot",
            slots=["cup__0", "cabinet__0"],
            params=dict(rel=[0.0, 0.0], start=list(START), bounds=CB.room_bounds(s), isolate=True, near=["table__0"]),
        )
        plain = pick_place.trial(s, u, None, 0)[1][0].details
        u = dict(u, params=dict(u["params"], freeze="once"))  # as calibrate records it
        once = pick_place.trial(s, u, None, 0)[1][0].details
        self.assertEqual(plain, once)  # bit for bit: restored, no step
        self.assertEqual(len(RP._KEPT), 1)
        jit = pick_place.trial(s, dict(u, params={k: v for k, v in u["params"].items() if k != "freeze"}), None, 1)[1][
            0
        ].details
        self.assertEqual(
            pick_place.trial(s, u, None, 1)[1][0].details, jit
        )  # a jitter seed: its own kept settle, bit for bit
        self.assertTrue(
            pick_place.trial(s, u, None, 0, dict(mass=1.3, friction=0.15))[0] == [True]
        )  # an edit: settled again, frozen
        self.assertEqual(len(RP._KEPT), 2)  # one whole-room settle per start and seed
        drawn = {}
        ok = CB._nudged(s, u, lambda *a: drawn.setdefault("edit", a[4]) and ([True], []), ["table__0"], 0.05, 3)
        rng = np.random.default_rng(3)
        self.assertTrue(ok)
        self.assertTrue(
            np.array_equal(drawn["edit"]["nudge"]["table__0"], rng.uniform(-0.05, 0.05, 2))
        )  # the draws calibrate makes
        self.assertTrue(
            np.array_equal(s.q0, Scene(self.xml, PHYSICS, probes=True, textures=False).q0)
        )  # the floor maps' start put back

    def test_a_changed_trial_settles_from_the_file_and_keeps_what_falls_awake(self):
        xml = rooms.write(tempfile.mkdtemp(), rooms.robot_room_xml().replace("</worldbody>", HIGH + "</worldbody>"))
        s = Scene(xml, PHYSICS, probes=True, textures=False)
        keep = ["cup__0", "cabinet__0"]
        u = dict(
            id="robot|cup__0|cabinet__0",
            kind="robot",
            slots=keep,
            params=dict(
                rel=[0.0, 0.0],
                start=list(START),
                bounds=CB.room_bounds(s),
                isolate=True,
                near=["table__0"],
                freeze="once",
            ),
        )
        w = RP.World(xml, START, textures=False)
        self.assertNotIn("crate__2", RP.around(w, keep))  # outside the zone
        self.assertIn("crate__2", w.restless())
        self.assertEqual(float(w.d.time), 0.0)  # put back at the file's start
        RP._KEPT.clear()  # (setup starts on the free floor)
        crate = lambda w: float(w.d.xipos[w.objects["crate__2"]][2])
        for seed, edit in ((0, None), (1, None), (0, dict(mass=1.3, friction=0.15))):
            w = pick_place.setup(xml, u, seed, s, edit, keep)[0]
            self.assertLess(crate(w), 2.0)  # it fell, as in the whole-room settle
            self.assertEqual(w.base is None, bool(edit))  # edited: from the file, not restored
        self.assertEqual(len(RP._KEPT), 2)  # seeds 0 and 1: their whole-room settles


if __name__ == "__main__":
    unittest.main()
