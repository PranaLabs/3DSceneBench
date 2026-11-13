"""Tilt (F2.3) is calibrated up to 60 degrees: a box with friction 1.0 on its floor slips near 45 degrees and gets a
tilt unit; before, the search stopped at 35 degrees (friction 0.7)."""

import os, tempfile, unittest
from sbv.common.scene import Scene
from sbv.functionality import f2_calibrate as C


def room(mu):
    x = f"""<mujoco><worldbody><geom name="floor" type="plane" size="5 5 0.1" friction="{mu} 0.005 0.0001"/>
<body name="box__0" pos="0 0 0.05"><freejoint/><geom type="box" size="0.1 0.1 0.05" mass="1" friction="{mu} 0.005 0.0001"/></body>
</worldbody></mujoco>"""
    d = tempfile.mkdtemp(prefix="sbv_tilt_")
    path = os.path.join(d, "scene.xml")
    with open(path, "w") as f:
        f.write(x)
    return path


class TiltRange(unittest.TestCase):
    def test_a_grippy_box_gets_a_unit(self):
        s = Scene(room(1.0), probes=True)
        s.settle()
        r = C.cal_tilt(s, "box__0", C.context(s, [3.0, 0.0]))
        self.assertIsNotNone(r)
        self.assertGreater(r[0]["params"]["angle"], 35.0)


if __name__ == "__main__":
    unittest.main()
