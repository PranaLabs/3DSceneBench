"""Press steps (F4.6): the reader finds a key that slides and refuses a part that does not; the robot presses the middle
key of a three-key pad on the table, which goes down and springs back without its neighbours, then two keys in a row."""

import tempfile, unittest
from sbv.common.scene import Scene, PHYSICS
from sbv.calibrate import room_bounds
from sbv.functionality import f2_calibrate as C
from sbv.functionality.f4_robot import plan as RP, tasks as T
from sbv.functionality.units import robot_trial
from tests import rooms

KEY = (
    '<body name="pad_key_{0}" pos="0 {1} 0.0145"><joint name="key_{0}_slide" type="slide" axis="0 0 1" range="-0.004 0" '
    'stiffness="200" damping="1"/><geom name="key_{0}_cap" type="box" size="0.012 0.009 0.004" density="800"/></body>'
)
PAD = (
    '<body name="keypad__0" pos="1.82 1.7 0.74"><freejoint/><geom name="pad_base" type="box" pos="0 0 0.005" size="0.03 0.06 0.005" density="1500"/>'
    + "".join(KEY.format(i, y) for i, y in enumerate((-0.03, 0.0, 0.03)))
    + '<body name="pad_lid" pos="-0.032 0 0.012"><joint name="lid_hinge" type="hinge" axis="0 1 0" range="0 1.5" damping="0.1"/>'
    '<geom name="lid_bar" type="box" size="0.002 0.05 0.002" density="800"/></body></body>'
)


def room():
    xml = rooms.write(tempfile.mkdtemp(), rooms.robot_room_xml().replace("</worldbody>", PAD + "</worldbody>"))
    s = Scene(xml, dict(PHYSICS), probes=True, textures=False)
    s.settle()
    b = room_bounds(s)
    c = [(b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2]
    return s, C.context(s, c), c, b


class Press(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s, cls.ctx, cls.c, cls.b = room()

    def activity(self, *parts):
        return T.named(
            self.s,
            self.ctx,
            dict(activity=[dict(words="x", steps=[dict(kind="press", object="keypad__0", part=p) for p in parts])]),
            self.c,
            self.b,
        )

    def run_(self, u):
        u["params"]["near"] = RP.nearby(self.s, u["slots"])
        self.s.reset()
        out, mons = robot_trial("robot_activity")(self.s, u, None, 0)
        return out, mons[0].details

    def test_the_key_is_found_and_a_part_that_does_not_slide_is_refused(self):
        units, bad = self.activity("key_1_slide")
        self.assertEqual(bad, [])
        self.assertEqual(units[0]["params"]["steps"][0]["ref"]["name"], "key_1_slide")
        for part, reason in [
            ("lid", "lid_hinge does not slide"),
            ("slide", "names 3 parts"),
            ("key_9_slide", "no part named"),
        ]:
            units, bad = self.activity(part)
            self.assertEqual(units, [])
            self.assertIn(reason, bad[0][1])

    @unittest.skipUnless(RP.installed(), "needs the RB-Y1 model (scripts/get_robot.sh)")
    def test_the_key_goes_down_and_springs_back_alone(self):
        out, det = self.run_(self.activity("key_1_slide")[0][0])
        st = det["steps"][0]
        self.assertEqual(out, [True], det)
        self.assertGreaterEqual(st["down"], 0.8 * 0.004)
        self.assertLessEqual(st["back"], 0.2 * 0.004)
        self.assertEqual(st["neighbours_pressed"], [])

    @unittest.skipUnless(RP.installed(), "needs the RB-Y1 model (scripts/get_robot.sh)")
    def test_two_keys_in_a_row_from_one_place(self):
        out, det = self.run_(self.activity("key_0_slide", "key_2_slide")[0][0])
        self.assertEqual((out, [x["success"] for x in det["steps"]]), ([True], [True, True]), det)
        self.assertLess(det["steps"][1]["seconds"], 6.0)  # the second key: no drive, just the hand


if __name__ == "__main__":
    unittest.main()
