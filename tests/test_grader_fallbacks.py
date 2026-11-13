"""A copy MuJoCo cannot put to sleep, or whose whole room diverges when it settles, is scored, not a grader failure.
Sleep: a unit that stops with FatalError runs again with the whole room awake (functionality/sleep.py awake), which is
what isolation stands in for. Settle: a diverged settle is the copy's own verdict, as Unstable is for one unit: no unit
runs, nothing rests, nothing touches."""

import os, tempfile, unittest
from unittest import mock
import mujoco, numpy as np
from sbv import calibrate, evaluate
from sbv.common.scene import Scene, Unstable
from sbv.functionality import sleep as SL
from tests import rooms

BOX = '<body name="{}" pos="{} 0 .05"><freejoint/><geom type="box" size=".05 .05 .05" mass="1"/></body>'
# two levers tied by a tendon equality, as the gym's rowing machine: MuJoCo 3.14 cannot sleep it
PAIR = (
    '<body name="rower__3" pos="{} {} .3"><geom type="box" size=".3 .1 .3" mass="20"/>'
    '<body name="lever_l" pos="-.2 0 .35"><joint name="hl" type="hinge" axis="1 0 0" range="-1 1" damping=".5"/>'
    '<geom type="capsule" fromto="0 0 0 0 .2 0" size=".02" mass=".3"/></body>'
    '<body name="lever_r" pos=".2 0 .35"><joint name="hr" type="hinge" axis="1 0 0" range="-1 1" damping=".5"/>'
    '<geom type="capsule" fromto="0 0 0 0 .2 0" size=".02" mass=".3"/></body></body>'
)
TIE = (
    '<tendon><fixed name="tie"><joint joint="hl" coef="1"/><joint joint="hr" coef="-1"/></fixed></tendon>'
    '<equality><tendon tendon1="tie"/></equality>'
)
ROW = (
    '<mujoco><compiler autolimits="true"/><worldbody><geom name="floor" type="plane" size="5 5 .1"/>'
    + BOX.format("striker__0", 0)
    + BOX.format("box__1", 0.3)
    + PAIR.format(0, 2)
    + "</worldbody>"
    + TIE
    + "</mujoco>"
)


def write(xml, like=None):
    """xml in a fresh folder; like: a scene file whose folder's other files (meshes, textures) are linked beside it."""
    d = tempfile.mkdtemp()
    p = os.path.join(d, os.path.basename(like) if like else "scene.xml")
    if like:
        for f in os.listdir(os.path.dirname(like)):
            if f != os.path.basename(like):
                os.symlink(os.path.join(os.path.dirname(like), f), os.path.join(d, f))
    with open(p, "w") as f:
        f.write(xml)
    return p


class Sleep(unittest.TestCase):
    def test_a_tendon_equality_stops_isolation_and_awake_runs_it_as_the_whole_room(self):
        S = Scene(write(ROW), probes=True)
        S.settle()
        m, d = S.m, S.d

        def push():
            S.reset()
            d.qvel[m.jnt_dofadr[m.body_jntadr[S.objects["striker__0"]]]] = 3.5
            mujoco.mj_forward(m, d)
            S.step(1.0)
            return {n: d.xpos[b].copy() for n, b in S.objects.items()}

        full = push()
        with self.assertRaisesRegex(mujoco.FatalError, "tendon equality"), SL.isolated(S, ["striker__0"], []):
            push()
        with SL.awake(), SL.isolated(S, ["striker__0"], []):
            woke = push()
        self.assertFalse(SL.AWAKE)
        self.assertIsNone(S.on_step)
        for n in full:
            np.testing.assert_allclose(woke[n], full[n], atol=0, err_msg=n)

    def test_a_copy_mujoco_cannot_sleep_is_scored_as_with_isolation_off(self):
        xml = rooms.write(tempfile.mkdtemp())
        case = calibrate.build(
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
        )
        self.assertTrue(case["functionality"]["isolate"])
        self.assertTrue(any(u["kind"] == "joint" for u in case["functionality"]["units"]))
        text = open(xml).read()
        copy = write(
            text.replace("</worldbody>", PAIR.format(3, 3) + "</worldbody>", 1).replace("</mujoco>", TIE + "</mujoco>"),
            xml,
        )
        r = evaluate.evaluate(dict(case, reference=xml), copy, procs=4, log=lambda *a: None)
        self.assertEqual(r["status"], "ok", r.get("failure"))
        self.assertTrue(r.get("awake_reruns"))  # before: grader_failed
        off = dict(case, reference=xml, functionality=dict(case["functionality"], isolate=False))
        whole = evaluate.evaluate(off, copy, procs=4, log=lambda *a: None)
        self.assertEqual(r["functionality"], whole["functionality"])


class Settle(unittest.TestCase):
    def test_a_room_whose_settle_diverges_is_scored_not_a_grader_failure(self):
        xml = rooms.write(tempfile.mkdtemp())
        case = calibrate.build(
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
        )
        real = evaluate.P.init_record

        def no_q1(scene, qpos=None, keep=None):
            return real(scene, qpos)  # as when seed 0's own rollout diverged

        with (
            mock.patch.object(evaluate.P, "init_record", no_q1),
            mock.patch.object(Scene, "settle", side_effect=Unstable("unstable simulation")),
        ):
            r = evaluate.evaluate(
                dict(case, reference=xml), xml, procs=4, log=lambda *a: None
            )  # before: Unstable escaped evaluate
        self.assertEqual(r["status"], "ok")
        self.assertIn("diverged", r["settle_diverged"])
        units = {u["id"]: u for u in case["functionality"]["units"]}
        ran = [
            k
            for k, v in r["units"].items()
            if units.get(k, {}).get("kind") not in (None, "init") and not v[0].get("settle_diverged")
        ]
        self.assertEqual(ran, [])
        self.assertLess(r["functionality"]["physics_hint"]["functionality"], 100.0)
        for row in (r.get("groups") or {}).get("groups", []):
            self.assertFalse(row["at_rest"])


if __name__ == "__main__":
    unittest.main()
