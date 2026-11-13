"""Declared container functions (functions.json): F3.5, the container carries what it holds."""

import json, tempfile, unittest
from pathlib import Path
from sbv.common.scene import Scene
from sbv.functionality import declared as D, f3_functions as Fn, f3_calibrate as CF

CUP = '<geom type="box" pos="0 0 .004" size=".04 .04 .004"/>' + "".join(
    f'<geom type="box" pos="{x} {y} .058" size="{sx} {sy} .05"/>'
    for x, y, sx, sy in (
        (0.037, 0, 0.003, 0.04),
        (-0.037, 0, 0.003, 0.04),
        (0, 0.037, 0.04, 0.003),
        (0, -0.037, 0.04, 0.003),
    )
)


def room(pencil_inside=True):
    """A cup on the floor; pencil__0 stands in it leaning, about half of it above the rim (or lies beside the cup);
    pencil__1 lies on the floor next to the cup."""
    p0 = (
        '<body name="pencil__0" pos="0 0 .115" euler="0 .3 0">'
        if pencil_inside
        else '<body name="pencil__0" pos=".15 .1 .005" euler="0 1.5708 0">'
    )
    xml = (
        f'<mujoco><compiler angle="radian"/><worldbody><geom name="floor" type="plane" size="2 2 .1"/>'
        f'<body name="cup__0"><freejoint/>{CUP}</body>{p0}<freejoint/><geom type="capsule" size=".004 .11" density="600"/></body>'
        '<body name="pencil__1" pos=".15 0 .005" euler="0 1.5708 0"><freejoint/><geom type="capsule" size=".004 .075" density="600"/></body>'
        "</worldbody></mujoco>"
    )
    p = Path(tempfile.mkdtemp()) / "scene.xml"
    p.write_text(xml)
    s = Scene(str(p), probes=True)
    s.settle()
    return s


class Holds(unittest.TestCase):
    def test_a_pencil_half_out_of_its_cup_is_held_and_one_beside_it_is_not(self):
        s = room()
        unit, why = CF.cal_holds(s, "cup__0", ["pencil__0"])
        self.assertIsNotNone(unit, why)
        self.assertEqual(unit["expect"], [True])
        unit, why = CF.cal_holds(s, "cup__0", ["pencil__1"])
        self.assertIsNone(unit)
        self.assertIn("the reference room fails it", why)

    def test_a_copy_with_the_pencil_out_of_the_cup_fails(self):
        unit, _ = CF.cal_holds(room(), "cup__0", ["pencil__0"])
        copy = room(pencil_inside=False)
        self.assertEqual(Fn.trial(copy, unit, {n: n for n in copy.objects})[0], [False])
        self.assertIsNone(Fn.trial(copy, unit, {"cup__0": "cup__0"})[0])  # the copy has no pencil__0 at all
        glued = room()
        glued.m.body_jntadr[
            glued.objects["cup__0"]
        ] = -1  # a copy whose cup has no free joint: a reason, not a TypeError
        self.assertEqual(
            Fn.trial(glued, unit, {n: n for n in glued.objects})[0],
            "the container has no free joint, so it cannot be lifted",
        )

    def test_declarations_the_room_cannot_carry_out(self):
        s = room()
        spec = dict(containers={"vase__0": {}, "cup__0": {"holds": ["pencil__0", "pen__3"]}})
        self.assertEqual(
            D.check(spec, s), ["no object named vase__0", "cup__0 holds pen__3, but there is no object named pen__3"]
        )
        d = tempfile.mkdtemp()
        xml = Path(d) / "scene.xml"
        xml.write_text("<mujoco/>")
        self.assertEqual(D.load(str(xml)), {})
        (Path(d) / "functions.json").write_text(json.dumps(spec))
        self.assertEqual(D.load(str(xml)), spec)


def walls(cx, half, h, side_hole=None):
    """A box cup's base and four walls round (cx, 0), inner half-width `half`, wall height h. side_hole: the +x wall
    gets a square hole of that half-size at mid height."""
    t = 0.005
    g = [f'<geom type="box" pos="{cx} 0 {t}" size="{half + t} {half + t} {t}"/>']
    for x, y, sx, sy in (
        (half + t, 0, t, half + t),
        (-half - t, 0, t, half + t),
        (0, half + t, half + t, t),
        (0, -half - t, half + t, t),
    ):
        if side_hole and x > 0:
            z0, s = 2 * t + h / 2, side_hole  # the +x wall in four pieces round the hole
            g += [
                f'<geom type="box" pos="{cx + x} 0 {(2 * t + z0 - s) / 2}" size="{t} {half + t} {(z0 - s - 2 * t) / 2}"/>',
                f'<geom type="box" pos="{cx + x} 0 {(z0 + s + 2 * t + h) / 2}" size="{t} {half + t} {(2 * t + h - z0 - s) / 2}"/>',
                f'<geom type="box" pos="{cx + x} {(s + half + t) / 2} {z0}" size="{t} {(half + t - s) / 2} {s}"/>',
                f'<geom type="box" pos="{cx + x} {-(s + half + t) / 2} {z0}" size="{t} {(half + t - s) / 2} {s}"/>',
            ]
        else:
            g.append(f'<geom type="box" pos="{cx + x} {y} {t + h / 2}" size="{sx} {sy} {h / 2}"/>')
    return "".join(g)


def scene_of(body):
    p = Path(tempfile.mkdtemp()) / "scene.xml"
    p.write_text(
        f'<mujoco><compiler angle="radian"/><worldbody><geom name="floor" type="plane" size="2 2 .1"/>{body}</worldbody></mujoco>'
    )
    s = Scene(str(p), probes=True)
    s.settle()
    return s


# A cup whose long handle puts the centre of its bounding box outside the cup (the case main's F3.2 gets wrong)
LONG = walls(0.25, 0.06, 0.2) + '<geom type="box" pos="-.15 0 .1" size=".35 .01 .01"/>'
TOP = dict(pos=[0.25, 0, 0.21], dir=[0, 0, -1], radius=0.06)


class Openings(unittest.TestCase):
    def test_a_long_handled_cup_holds_the_ball_at_its_declared_opening(self):
        s = scene_of(f'<body name="cup__0"><freejoint/>{LONG}</body>')
        self.assertFalse(Fn.container_once(s, "cup__0", None)[0])  # dropped over the box centre, it misses the cup
        unit, why = CF.cal_opening(s, "cup__0", 0, TOP)
        self.assertIsNotNone(unit, why)

    def test_a_copy_with_its_own_body_frame_is_lined_up_first(self):
        unit, _ = CF.cal_opening(scene_of(f'<body name="cup__0"><freejoint/>{LONG}</body>'), "cup__0", 0, TOP)
        # The same cup in the same place, its body origin 30 cm along x and its axes turned 90 degrees
        import re

        turned = re.sub(
            r'pos="(\S+) (\S+) (\S+)"',
            lambda m: f'pos="{-float(m[2])} {float(m[1]) - 0.3} {m[3]}" euler="0 0 -1.5708"',
            LONG,
        )
        copy = scene_of(f'<body name="cup__0" pos=".3 0 0" euler="0 0 1.5708"><freejoint/>{turned}</body>')
        self.assertEqual(Fn.trial(copy, unit, {"cup__0": "cup__0"})[0], [True])

    def test_each_declared_opening_gets_its_own_run(self):
        # a box with a top opening and a 3.5 cm hole in its +x wall: a ball thrown sideways through the hole stays in
        s = scene_of(f'<body name="can__0"><freejoint/>{walls(0, 0.06, 0.2, side_hole=0.0175)}</body>')
        side = dict(pos=[0.065, 0, 0.11], dir=[-1, 0, 0], radius=0.0175)
        for k, op in enumerate((dict(TOP, pos=[0, 0, 0.21]), side)):
            unit, why = CF.cal_opening(s, "can__0", k, op)
            self.assertIsNotNone(unit, f"opening {k}: {why}")

    def test_a_lidded_cup_fails_and_a_narrow_opening_is_refused(self):
        s = scene_of(
            f'<body name="cup__0"><freejoint/>{LONG}<geom type="box" pos=".25 0 .215" size=".07 .07 .005"/></body>'
        )
        unit, why = CF.cal_opening(s, "cup__0", 0, TOP)
        self.assertIsNone(unit)
        self.assertIn("the reference room fails it", why)
        self.assertEqual(
            D.check(dict(containers={"cup__0": {"openings": [dict(TOP, radius=0.01)]}}), s),
            ["cup__0 opening 0 is 2.0 cm across, too narrow for the 2 cm probe ball"],
        )

    def test_a_ball_resting_on_a_solid_top_is_not_in(self):
        # lifted, a solid block carries a ball lying on its top too; the ball must have gone in past the opening
        s = scene_of('<body name="cup__0"><freejoint/><geom type="box" pos="0 0 .1" size=".07 .07 .1"/></body>')
        unit, why = CF.cal_opening(s, "cup__0", 0, dict(TOP, pos=[0, 0, 0.2]))
        self.assertIsNone(unit)
        self.assertIn("the reference room fails it", why)

    def test_a_fixed_container_walls_the_ball_in(self):
        s = scene_of(f'<body name="bin__0">{walls(0, 0.06, 0.2)}</body>')
        unit, why = CF.cal_opening(s, "bin__0", 0, dict(TOP, pos=[0, 0, 0.21]))
        self.assertIsNotNone(unit, why)


class Drafts(unittest.TestCase):
    """python -m sbv.declare drafts functions.json from the reference room; calibrate stops when a room needs one and
    has none."""

    def test_the_hollow_cup_is_declared_and_the_solid_crate_and_vase_are_not(self):
        from sbv import declare
        from tests import rooms

        d = tempfile.mkdtemp()
        rooms.write(d, rooms.room_xml(), functions=False)  # the default room, without its functions.json
        _, spec = declare.draft(d, log=lambda *a: None)
        self.assertEqual(list(spec["containers"]), ["cup__0"])
        op = spec["containers"]["cup__0"]["openings"][0]
        self.assertEqual(op["dir"], [0.0, 0.0, -1.0])
        self.assertAlmostEqual(op["pos"][2], 0.1, 3)
        self.assertGreater(op["radius"], 0.03)
        self.assertEqual(sorted(spec["not_containers"]), ["crate__0", "vase__0"])
        self.assertIn("no hollow", spec["not_containers"]["crate__0"])

    def test_a_pencil_standing_in_its_cup_is_drafted_as_held(self):
        from sbv import declare

        _, spec = declare.draft(room().xml, log=lambda *a: None)
        self.assertEqual(spec["containers"]["cup__0"].get("holds"), ["pencil__0"])  # pencil__1 lies beside it

    def test_a_long_handled_cup_gets_its_opening_over_its_centre_of_mass(self):
        from sbv import declare

        s = scene_of(f'<body name="cup__0"><freejoint/>{LONG}</body>')
        entry, why = declare.propose(s, "cup__0")
        self.assertIsNotNone(entry, why)
        self.assertGreater(entry["openings"][0]["pos"][0], 0.15)  # over the cup at x .25, not the box centre

    def test_calibration_stops_for_a_room_with_a_cup_and_no_functions_json(self):
        from sbv import calibrate
        from tests import rooms

        d = tempfile.mkdtemp()
        xml = rooms.write(d, rooms.room_xml(), functions=False)
        self.assertEqual(D.container_like(D.object_names(xml)), ["crate__0", "cup__0", "vase__0"])
        self.assertEqual(
            D.container_like(["plant__0", "flowerpot__1", "mug__2", "pot__3"]), ["mug__2", "pot__3"]
        )  # mug is a cup
        self.assertIn("sbv.declare", D.missing_file(xml, D.object_names(xml)))
        with self.assertRaises(SystemExit) as stop:
            calibrate.build(
                d,
                procs=1,
                draws=1,
                need=1,
                jitter=1,
                robot_tasks=0,
                push_tasks=0,
                require_tasks=False,
                log=lambda *a: None,
                checkpoint=tempfile.mkdtemp(),
            )
        self.assertIn("no functions.json", str(stop.exception))
        Path(d, "functions.json").write_text('{"containers": {}}')
        self.assertIsNone(
            D.missing_file(xml, D.object_names(xml))
        )  # an empty file is a decision; the stop is for a forgotten one
        plain = Path(tempfile.mkdtemp()) / "scene.xml"
        plain.write_text('<mujoco><worldbody><body name="chair__0"><geom size=".1"/></body></worldbody></mujoco>')
        self.assertIsNone(
            D.missing_file(str(plain), D.object_names(str(plain)))
        )  # no container words: nothing to declare
