"""Isolation for doors and drawers (sbv/functionality/sleep.py): the same physics, the same case, the same score."""

import os, tempfile, unittest
import mujoco, numpy as np
from sbv import calibrate, evaluate
from sbv.common.scene import Scene
from sbv.functionality import sleep as SL, f3_functions as Fn
from tests import rooms


def scene(xml):
    p = os.path.join(tempfile.mkdtemp(), "scene.xml")
    with open(p, "w") as f:
        f.write(xml)
    S = Scene(p, probes=True)
    S.settle()
    return S


BOX = '<body name="{}" pos="{} 0 .05"><freejoint/><geom type="box" size=".05 .05 .05" mass="1" friction=".3 .005 .0001"/></body>'
ROW = (
    '<mujoco><worldbody><geom name="floor" type="plane" size="5 5 .1"/>'
    + BOX.format("striker__0", 0)
    + BOX.format("box__1", 0.3)
    + BOX.format("box__2", 3.0)
    + "</worldbody></mujoco>"
)
LEAF = (
    '<mujoco><compiler angle="radian" autolimits="true"/><default><geom friction="0.6 0.005 0.0001" density="600"/></default>'
    '<worldbody><geom name="floor" type="plane" size="0 0 .1"/><body name="desk__0"><geom type="box" pos="0 0 .4" size=".03 .4 .4"/>'
    '<body name="desk_leaf" pos="-.06 0 .8"><joint name="leaf_hinge" type="hinge" axis="0 -1 0" range="0 1.4" damping=".5" frictionloss="5"'
    ' solimpfriction="0.99 0.999 0.001" solreffriction="0.004 1"/><geom type="box" pos="-.27 0 -.01" size=".2 .35 .01" mass="1"/></body></body>'
    '<body name="box__0" pos="-.33 0 .83"><freejoint/><geom type="box" size=".04 .04 .03" mass=".5"/></body>'
    '<body name="block__1" pos="2 2 0"><geom type="box" pos="0 0 .2" size=".2 .2 .2"/></body></worldbody></mujoco>'
)


class Isolation(unittest.TestCase):
    def test_a_chain_wakes_what_it_hits_and_the_rest_sleeps(self):  # a striker slides into box 1; box 2 rests 3 m away
        S = scene(ROW)
        m, d = S.m, S.d
        tree = lambda n: m.body_treeid[S.objects[n]]

        def push():
            S.reset()
            d.qvel[m.jnt_dofadr[m.body_jntadr[S.objects["striker__0"]]]] = 3.5
            mujoco.mj_forward(m, d)
            S.step(1.0)
            return {n: d.xpos[b].copy() for n, b in S.objects.items()}, d.tree_asleep[tree("box__2")]

        full, _ = push()
        with SL.isolated(S, ["striker__0"], []):
            iso, asleep = push()
        self.assertGreater(full["box__1"][0] - 0.3, 0.05)  # box 1 was asleep at the start and is hit
        for n in full:
            np.testing.assert_allclose(iso[n], full[n], atol=1e-3, err_msg=n)
        self.assertGreaterEqual(asleep, 0)
        self.assertIsNone(S.on_step)  # box 2 slept; the hook is gone afterwards

    def test_scratch_data_is_built_and_posed_with_sleep_off(
        self,
    ):  # MuJoCo 3.14: garbage sleep state or wrong poses otherwise
        from unittest import mock
        from sbv.functionality import trajectory as TR

        S = scene(LEAF)
        m = S.m
        bit = int(mujoco.mjtEnableBit.mjENBL_SLEEP)
        j = S.joints["desk__0"][0]
        rec = Fn.joint_record(S, "desk__0", j)
        full = Fn.sweep_report(S, j, rec["sgn"], rec["travel"], None)
        built = []

        def spy(mm):
            built.append(bool(mm.opt.enableflags & bit))
            return real(mm)

        real = mujoco.MjData
        with SL.isolated(S, ["desk__0"], []):
            S.reset()
            S.step(0.01)
            with mock.patch.object(mujoco, "MjData", spy):
                iso = Fn.sweep_report(S, j, rec["sgn"], rec["travel"], None)
                T = TR.Trajectory(S, {S.objects["box__0"]}, set())
            self.assertTrue(m.opt.enableflags & bit)  # the rollouts keep sleep on
        self.assertEqual(built, [False, False])
        self.assertEqual(iso["status"], full["status"])
        np.testing.assert_allclose(T.scratch.geom_xpos[T.static], T.static_pos, atol=0)
        ref = mujoco.MjData(m)
        ref.qpos[:] = T.scratch.qpos
        mujoco.mj_kinematics(m, ref)
        np.testing.assert_allclose(T.scratch.geom_xpos, ref.geom_xpos, atol=0)  # the poses sleep off gives

    def test_what_rests_on_a_door_set_half_open_falls_as_in_the_whole_room(self):  # rest_stays teleports the leaf
        S = scene(LEAF)
        j = S.joints["desk__0"][0]
        rec = Fn.joint_record(S, "desk__0", j)
        box = S.objects["box__0"]

        def rest():
            return Fn.rest_stays(S, j, rec["sgn"], 0.5 * rec["travel"], None, S.objects["desk__0"])[0], S.d.xpos[
                box
            ].copy()

        full = rest()
        with SL.isolated(S, ["desk__0"], []):
            iso = rest()
        self.assertEqual(iso[0], full[0])
        np.testing.assert_allclose(iso[1], full[1], atol=1e-3)

    def test_the_test_room_gives_the_same_case_and_scores_100_both_ways(self):
        xml = rooms.write(tempfile.mkdtemp())
        cases, scores = {}, {}
        for iso in (True, False):
            c = calibrate.build(
                os.path.dirname(xml),
                procs=4,
                draws=2,
                need=2,
                jitter=1,
                log=lambda *a: None,
                require_tasks=False,
                robot_tasks=0,
                push_tasks=0,
                camera_seed=7,
                isolate=iso,
            )
            r = evaluate.evaluate(dict(c, reference=xml), xml, procs=4, log=lambda *a: None)
            scores[iso] = r["functionality"]
            self.assertEqual(c["functionality"]["isolate"], iso)
            self.assertEqual(r["functionality"]["physics_hint"]["functionality"], 100.0)
            cases[iso] = {k: c["functionality"][k] for k in ("units", "excluded")}
        self.assertTrue(any(u["kind"] == "joint" for u in cases[True]["units"]))
        self.assertEqual(cases[True], cases[False])
        self.assertEqual(scores[True], scores[False])
