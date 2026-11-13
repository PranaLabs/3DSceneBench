"""The slow push: the breaking force of a ramp follows weight x friction for a sliding box (the 0.1 s blow it replaces
followed the mass), a tall object gives way earlier by tipping, and a copy passes within a factor of 2 of the true
force."""

import os, tempfile, unittest
from sbv.common.scene import Scene
from sbv.functionality import f2_calibrate as C, f2_physical as P


def room(mass, mu, h=0.05):
    x = f"""<mujoco><worldbody><geom name="floor" type="plane" size="5 5 0.1" friction="0.05 0.005 0.0001"/>
<body name="box__0" pos="0 0 {h}"><freejoint/><geom type="box" size="0.05 0.05 {h}" mass="{mass}" friction="{mu} 0.005 0.0001"/></body>
</worldbody></mujoco>"""
    d = tempfile.mkdtemp(prefix="sbv_ramp_")
    path = os.path.join(d, "scene.xml")
    with open(path, "w") as f:
        f.write(x)
    return path


def force(mass, mu, h=0.05):
    s = Scene(room(mass, mu, h), probes=True)
    s.settle()
    u, _ = C.cal_push(s, "box__0", C.context(s, [3.0, 0.0]))
    return s, u


class SlowPush(unittest.TestCase):
    def test_weight_times_friction(self):
        _, a = force(1.0, 0.5)
        _, b = force(2.0, 0.5)
        _, c = force(1.0, 0.25)
        self.assertAlmostEqual(b["params"]["F"] / a["params"]["F"], 2.0, delta=0.25)  # the blow gave 2.00 too
        self.assertAlmostEqual(c["params"]["F"] / a["params"]["F"], 0.5, delta=0.15)  # the blow gave 0.64

    def test_a_tall_box_tips_first(self):
        _, low = force(1.0, 0.5)
        _, tall = force(1.0, 0.5, h=0.4)  # 10 cm wide, 80 cm tall
        self.assertLess(tall["params"]["F"], 0.5 * low["params"]["F"])

    def test_bars(self):
        s, u = force(1.0, 0.5)
        F = u["params"]["F"]
        self.assertEqual(P.trial(s, u, qpos=s.start)[0], [False, True, False, True])  # the true box against itself
        self.assertEqual(
            P.ramp_outcome(1.6 * F, F, [2.0, 1.25]), [False, True, False, False]
        )  # passes at 2, not at 1.25
        self.assertEqual(P.ramp_outcome(None, F, [2.0, 1.25]), [False, False, False, False])  # holds past 2.5 F: fails
        self.assertEqual(P.ramp_outcome(0.4 * F, F, [2.0, 1.25]), [True, True, True, True])  # too easy: fails


if __name__ == "__main__":
    unittest.main()
