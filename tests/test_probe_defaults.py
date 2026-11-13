"""The probes (the ball, mug, load box and gripper of the functionality tests) are added to a room's own file, so they
used to take that file's root <default>: a copy's defaults decided how the probes touched it. Scoring now gives the
copy's probes the rows the reference room's file gives them (scene.reference_probes), which is what calibration froze
the tests with, so no case changes and a reference room's own model is the same bit for bit."""

import os, tempfile, unittest
import numpy as np, mujoco
from sbv.common import scene as SC
from sbv.common.scene import Scene, PHYSICS

BODY = """<worldbody><geom name="floor" type="plane" size="5 5 .1" contype="1" conaffinity="1"/>
<body name="cup__0" pos="0 0 .1"><freejoint/><geom type="box" size=".05 .05 .05" contype="1" conaffinity="1"/></body></worldbody>"""
PLAIN = f"<mujoco>{BODY}</mujoco>"
OWN = f'<mujoco><compiler autolimits="true"/><default><geom condim="4" margin="0.002" solref=".008 1"/><joint armature=".01" damping=".5"/></default>{BODY}</mujoco>'
COPY = (
    f'<mujoco><default><geom contype="0" conaffinity="0" condim="6" margin="0.01" priority="2"/><joint armature="1" frictionloss="5" limited="false"/>'
    f'<equality solref=".5 1"/></default>{BODY}</mujoco>'
)


def write(text):
    p = os.path.join(tempfile.mkdtemp(prefix="sbv_probe_"), "scene.xml")
    with open(p, "w") as f:
        f.write(text)
    return p


def arrays(m):
    return {
        k: np.array(getattr(m, k)) for k in dir(m) if not k.startswith("_") and isinstance(getattr(m, k), np.ndarray)
    }


class ProbeDefaults(unittest.TestCase):
    def test_a_copy_s_defaults_no_longer_reach_the_probes(self):
        true, copy = write(PLAIN), write(COPY)
        want = SC.probe_rows(Scene(true, None, ["cup"], probes=True).m)
        self.assertNotEqual(SC.probe_rows(Scene(copy, None, ["cup"], probes=True).m), want)  # as it was: the copy's own
        got = Scene(copy, dict(PHYSICS, probes=SC.reference_probes(true)), ["cup"], probes=True)
        self.assertEqual(SC.probe_rows(got.m), want)
        ball = got.m.geom_bodyid == got.probe["ball"]
        self.assertTrue((got.m.geom_contype[ball] == 1).all() and (got.m.geom_conaffinity[ball] == 1).all())
        m, d = got.m, got.d
        got.reset(got.q0)
        b = got.probe["mug"]
        adr = m.jnt_qposadr[m.body_jntadr[b]]
        g = int(np.flatnonzero(m.geom_bodyid == b)[0])
        d.qpos[adr : adr + 3] = [1.0, 1.0, m.geom_size[g][1] - 0.002]
        d.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(m, d)  # the mug 2 mm into the floor
        self.assertTrue(
            any(g in (c.geom1, c.geom2) for c in d.contact[: d.ncon])
        )  # it touches: the per-body masks the broad phase reads follow the geoms
        self.assertEqual(
            {k: np.array(getattr(m, k))[b].tolist() for k in ("body_contype", "body_conaffinity")},
            {
                k: np.array(getattr(Scene(true, None, ["cup"], probes=True).m, k))[b].tolist()
                for k in ("body_contype", "body_conaffinity")
            },
        )

    def test_the_true_room_s_own_defaults_stay_and_its_model_is_untouched(self):
        for text in (PLAIN, OWN):
            true = write(text)
            a = Scene(true, None, ["cup"], probes=True)
            rows = SC.reference_probes(true)
            self.assertEqual(
                rows, SC.probe_rows(a.m)
            )  # the small model of the file's defaults gives the room's own rows
            self.assertFalse(SC.set_probe_rows(a.m, rows))  # so nothing is written
            b = Scene(true, dict(PHYSICS, probes=rows), ["cup"], probes=True)
            x, y = arrays(a.m), arrays(b.m)
            self.assertEqual(sorted(x), sorted(y))
            self.assertEqual([k for k in x if not np.array_equal(x[k], y[k], equal_nan=True)], [])
        own = SC.reference_probes(write(OWN))
        self.assertEqual(own["geom"]["probe_mug#0"]["condim"], 4)  # a room that sets a default keeps it for its probes

    def test_a_copy_with_autolimits_off_compiles_with_the_probes(self):
        s = Scene(
            write(PLAIN.replace("<mujoco>", '<mujoco><compiler autolimits="false"/>')), None, ["cup"], probes=True
        )
        j = mujoco.mj_name2id(s.m, mujoco.mjtObj.mjOBJ_JOINT, "probe_slide_l")
        self.assertTrue(s.m.jnt_limited[j])


if __name__ == "__main__":
    unittest.main()
