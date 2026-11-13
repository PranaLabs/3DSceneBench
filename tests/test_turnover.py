"""F3.6 turn over: what a container or a small holder holds falls out when it is turned upside down, as in the reference
room."""

import tempfile, unittest
from pathlib import Path
from sbv.common.scene import Scene
from sbv.functionality import f3_functions as Fn, f3_calibrate as CF, units as U
from tests.test_declared import walls

COASTER = (
    '<body name="coaster__0" pos="0 0 .003"><freejoint/><geom type="cylinder" size=".05 .003" mass=".007"/></body>'
)
CUP = '<body name="cup__0" pos="0 0 .056"><freejoint/><geom type="cylinder" size=".035 .05" mass=".21"/></body>'
WELD = '<equality><weld body1="coaster__0" body2="cup__0"/></equality>'
OPEN_CUP = f'<body name="cup__1" pos=".5 0 0"><freejoint/>{walls(0, 0.04, 0.1)}</body>'
SOLID_CUP = '<body name="cup__1" pos=".5 0 .05"><freejoint/><geom type="cylinder" size=".045 .05"/></body>'
GHOST_CUP = '<body name="cup__1" pos=".5 0 .05"><freejoint/><geom type="cylinder" size=".045 .05" contype="0" conaffinity="0"/></body>'
TOP = dict(pos=[0, 0, 0.11], dir=[0, 0, -1], radius=0.04)


def scene_of(bodies, extra=""):
    p = Path(tempfile.mkdtemp()) / "scene.xml"
    p.write_text(
        f'<mujoco><worldbody><geom name="floor" type="plane" size="2 2 .1"/>{bodies}</worldbody>{extra}</mujoco>'
    )
    s = Scene(str(p), probes=True)
    s.settle()
    return s


class TurnOver(unittest.TestCase):
    def test_a_cup_falls_off_its_coaster_and_a_glued_one_does_not(self):
        unit, why = CF.cal_turnover(scene_of(COASTER + CUP), "coaster__0", ["cup__0"])
        self.assertIsNotNone(unit, why)
        self.assertEqual(unit["expect"], [True])
        self.assertEqual((U.number(unit), U.level(unit), U.layer(unit)), ("F3.6", "affordance", "F3"))
        glued = scene_of(COASTER + CUP, WELD)  # a light coaster under a heavy cup: the hand must not yield
        self.assertEqual(Fn.trial(glued, unit, {n: n for n in glued.objects})[0], [False])

    def test_the_ball_put_in_through_the_opening_falls_out(self):
        unit, why = CF.cal_turnover(scene_of(OPEN_CUP), "cup__1", k=0, op=TOP)
        self.assertIsNotNone(unit, why)
        self.assertEqual(unit["expect"], [True])
        self.assertEqual(unit["id"], "turnover|cup__1|opening0")
        solid = scene_of(SOLID_CUP)  # a solid copy: the ball never gets in, a failed run
        self.assertEqual(Fn.trial(solid, unit, {"cup__1": "cup__1"})[0], "the ball does not get in")

    def test_a_copy_cup_with_no_collision_surface_fails_and_the_grade_goes_on(self):
        unit, why = CF.cal_opening(scene_of(OPEN_CUP), "cup__1", 0, TOP)
        self.assertIsNotNone(unit, why)
        turn, why = CF.cal_turnover(scene_of(OPEN_CUP), "cup__1", k=0, op=TOP)
        self.assertIsNotNone(turn, why)
        ghost = scene_of(GHOST_CUP)
        self.assertEqual(Fn.trial(ghost, unit, {"cup__1": "cup__1"})[0], [False])
        self.assertEqual(Fn.trial(ghost, turn, {"cup__1": "cup__1"})[0], Fn.NO_SURFACE)

    def test_what_cannot_be_turned_over_is_said(self):
        fixed = scene_of(COASTER.replace("<freejoint/>", "") + CUP)
        unit, why = CF.cal_turnover(fixed, "coaster__0", ["cup__0"])
        self.assertIsNone(unit)
        self.assertIn("fixed", why)
        unit, _ = CF.cal_turnover(scene_of(COASTER + CUP), "coaster__0", ["cup__0"])
        self.assertEqual(
            Fn.trial(fixed, unit, {n: n for n in fixed.objects})[0],
            "the holder has no free joint, so it cannot be turned over",
        )
        self.assertIsNone(
            Fn.trial(fixed, unit, {"coaster__0": "coaster__0"})[0]
        )  # the copy has no cup: missing, scored as such

    def test_the_turned_object_keeps_its_mass(self):
        s = scene_of(COASTER + CUP)
        m0 = s.m.body_mass.copy()
        CF.cal_turnover(s, "coaster__0", ["cup__0"])
        self.assertTrue((s.m.body_mass == m0).all())


if __name__ == "__main__":
    unittest.main()
