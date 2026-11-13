"""Physical support regressions: geometry, grounded chains and gravity controls."""

import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from sbv.common.scene import Scene
from sbv.geometry import g3_layout as L
from sbv.geometry.support import SupportIndex, rooted_paths
from sbv.functionality import f2_calibrate as C, f2_physical as P

FLOOR = '<geom name="floor" type="plane" size="2 2 .1"/>'


def box(pos, size, extra=""):
    return f'<geom type="box" pos="{pos}" size="{size}" {extra}/>'


def body(name, geoms, free=False, extra=""):
    return f'<body name="{name}" {extra}>{"<freejoint/>" if free else ""}{geoms}</body>'


class ContactSupport(unittest.TestCase):
    def scene(self, xml, floor=FLOOR, asset=""):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = Path(tmp.name) / "scene.xml"
        p.write_text(
            f'<mujoco><compiler angle="radian"/><default><geom solref=".004 1"/></default><asset>{asset}</asset><worldbody>{floor}{xml}</worldbody></mujoco>'
        )
        return Scene(str(p))

    def rows(self, scene, mounts=(), corr=None):
        spec = dict(
            support_tol=0.01, penetration_tol=0.002, bounds_tol=0.02, bounds=[[-3, -3, -0.1], [3, 3, 4]], mounts=mounts
        )
        return {r["object"]: r for r in L.placement(spec, [], scene, corr or {n: n for n in scene.objects})["objects"]}

    def test_real_basket_rims_and_calibration_pairs(self):
        s = Scene(str(Path(__file__).parent / "fixtures/uq_basket_stack.xml"))
        s.settle()
        for i in range(1, 7):
            a, b = f"utility_basket__{i}", f"utility_basket__{i - 1}"
            self.assertTrue(L.supported_by(s, a, s.ogeoms[b], 0.01), (a, b))
            self.assertIn((b, a), C.support_pairs(s))
        self.assertTrue(all(r["support"] for r in self.rows(s).values()))
        p0 = {n: s.center(n).copy() for n in s.objects}
        s.step(2)
        self.assertLess(max(np.linalg.norm(s.center(n) - p0[n]) for n in s.objects), 0.001)

    def test_shoulders_scale_rotation_and_gap(self):
        for scale in (0.4, 1.0, 2.0):
            for yaw in (0, 0.6, 1.4):
                for gap, expected in ((0, True), (0.005, True), (0.02, False)):
                    with self.subTest(scale=scale, yaw=yaw, gap=gap):
                        pedestal = box(f"0 0 {0.25 * scale}", f"{0.25 * scale} {0.2 * scale} {0.25 * scale}")
                        top = box("0 0 0", f"{0.5 * scale} {0.25 * scale} {0.05 * scale}")
                        # The ends extend below the actual shoulder contact.
                        top += "".join(
                            box(f"{x * scale} 0 {-0.1 * scale}", f"{0.015 * scale} {0.2 * scale} {0.15 * scale}")
                            for x in (-0.45, 0.45)
                        )
                        s = self.scene(
                            body("stand__0", pedestal, extra=f'euler="0 0 {yaw}"')
                            + body("odd__0", top, True, f'pos="0 0 {0.55 * scale + gap}" euler="0 0 {yaw}"')
                        )
                        self.assertEqual(self.rows(s)["odd__0"]["support"], expected)

    def test_float_noise_at_the_tolerance_is_support(self):
        # A bench at z 0.5 with half height 0.49 has its lowest point 0.010000000000000009 m up: float noise over the
        # 0.01 m tolerance failed the bench and everything on it. Every compare allows 1e-6; a real gap still fails.
        for floor in (
            FLOOR,
            box("0 0 -.05", "2 2 .05"),
        ):  # plane (geom distance) and box floor (nearby() filter and box SAT)
            for z, expected in ((0.5, True), (0.5001, False), (0.501, False)):
                with self.subTest(floor=floor, z=z):
                    s = self.scene(
                        body("bench__0", box("0 0 0", ".0125 1.07 .49"), extra=f'pos="0 0 {z}"')
                        + body("cup__0", box("0 0 0", ".01 .04 .05"), True, f'pos="0 0 {z + 0.54:.4f}"'),
                        floor=floor,
                    )
                    r = self.rows(s)
                    self.assertEqual((r["bench__0"]["support"], r["cup__0"]["support"]), (expected, expected))
        wall, mount = (
            box("0 0 1", ".1 .5 1"),
            [{"a": "shelf__0", "b": "world"}],
        )  # a frozen mount, 0.26 - 0.1 - 0.15 m off the wall
        for x, expected in ((0.26, True), (0.261, False)):
            with self.subTest(mount=x):
                s = self.scene(body("shelf__0", box(f"{x} 0 1", ".15 .2 .05")), floor=FLOOR + wall)
                self.assertEqual(self.rows(s, mount)["shelf__0"]["support"], expected)

    def test_cup_hovering_past_table_edge_is_not_supported(self):
        table = body("table__0", box("0 0 .375", ".3 .3 .375"))
        for shape in ('<geom type="box" size=".04 .04 .05"/>', '<geom type="cylinder" size=".04 .05"/>'):
            for past, up in ((0.002, 0.001), (0.002, 0.009), (0.005, 0.005), (0.008, 0.009)):
                with self.subTest(shape=shape, past=past, up=up):
                    s = self.scene(table + body("cup__0", shape, True, f'pos="{0.342 + past} 0 {0.8 + up}"'))
                    self.assertFalse(self.rows(s)["cup__0"]["support"])
            s = self.scene(table + body("cup__0", shape, True, 'pos="0.2 0 0.805"'))
            self.assertTrue(self.rows(s)["cup__0"]["support"])

    def test_visual_trim_does_not_change_support(self):
        stand = body("stand__0", box("0 0 .25", ".2 .2 .25"))
        trim = box("0 0 -.12", ".3 .3 .02", 'contype="0" conaffinity="0" mass="0"')
        s = self.scene(stand + body("item__0", box("0 0 0", ".1 .1 .05") + trim, True, 'pos="0 0 .55"'))
        self.assertTrue(self.rows(s)["item__0"]["support"])

    def test_desk_picture_level_with_its_collision_top(self):
        # Built rooms often draw a desk top with a picture-only mesh whose top is level with the collision top. A 0.05
        # mm lift makes the picture the first hit; at exactly level, a floating-point tie decides.
        slab = " ".join(f"{x * 0.4} {y * 0.3} {z * 0.03}" for x in (-1, 1) for y in (-1, 1) for z in (-1, 1))
        picture = (
            lambda lift: f'<geom type="mesh" mesh="slab" pos="0 0 {0.77 + lift}" contype="0" conaffinity="0" mass="0"/>'
        )
        for solid in (
            lambda z: box(f"0 0 {z}", ".4 .3 .03"),
            lambda z: f'<geom type="mesh" mesh="slab" pos="0 0 {z}"/>',
        ):
            for lift, drop, expected in (
                (0, 0, True),
                (5e-5, 0, True),
                (0, 0.02, False),
            ):  # level, or collision 2 cm under
                with self.subTest(solid=solid(0.77), lift=lift, drop=drop):
                    desk = body("desk__0", picture(lift) + box("0 0 .37", ".05 .05 .37") + solid(0.77 - drop))
                    item = body("book__0", box("0 0 0", ".1 .08 .015"), True, 'pos=".05 .03 .815"')
                    s = self.scene(desk + item, asset=f'<mesh name="slab" vertex="{slab}"/>')
                    self.assertEqual(self.rows(s)["book__0"]["support"], expected)

    def test_flush_mesh_leg_with_no_witness_is_supported(self):
        # MuJoCo reads a convex mesh leg within nanometres of a box floor as 0 apart with no witness (all six 0). Such a
        # reading is a touch, read with a lifted 1 mm.
        leg = " ".join(
            f"{0.0175 * np.cos(np.pi * i / 4):.6f} {0.0175 * np.sin(np.pi * i / 4):.6f} {z}"
            for i in range(8)
            for z in (-0.196, 0.196)
        )
        floor = '<geom name="floor" type="box" pos="0 0 -.05" size="2 2 .05"/>'
        for free in (False, True):
            stool = lambda z: self.scene(
                body(
                    "stool__0",
                    '<geom name="leg" type="mesh" mesh="leg"/>',
                    free,
                    f'pos="1.2 .7 {z!r}" euler="2.6e-7 2.6e-7 0"',
                ),
                floor,
                f'<mesh name="leg" vertex="{leg}"/>',
            )
            s = stool(0.3)
            s.reset(s.q0)
            low = s.geom_points(s.m.geom("leg").id)[:, 2].min()
            raw = []
            for dz, expected in ((1e-10, True), (1e-9, True), (1e-8, True), (0.005, True), (0.02, False)):
                with self.subTest(free=free, dz=dz):
                    s = stool(0.3 + dz - low)
                    s.reset(s.q0)
                    w = np.zeros(6)
                    raw.append(
                        mujoco.mj_geomDistance(s.m, s.d, s.m.geom("leg").id, s.m.geom("floor").id, 0.011, w) == 0
                        and not w.any()
                    )
                    self.assertEqual(self.rows(s)["stool__0"]["support"], expected)
            self.assertTrue(
                any(raw)
            )  # MuJoCo's own reading does go wrong on this grid, so the check above means something

    def test_bridge_has_two_supporters_and_stays_at_rest(self):
        stands = "".join(body(f"stand__{i}", box(f"{x} 0 .25", ".1 .2 .25")) for i, x in enumerate((-0.35, 0.35)))
        s = self.scene(stands + body("bridge__0", box("0 0 .55", ".5 .2 .05"), True))
        parents, paths = SupportIndex(s, 0.01).graph()
        self.assertEqual(parents["bridge__0"], {"stand__0", "stand__1"})
        self.assertIn("bridge__0", paths)
        self.assertTrue(P.init_record(s)["bridge__0"]["still"])

    def test_floating_base_costs_itself_on_a_slope(self):
        # A counter fixed in the air with two cups on its top, and a missing lamp: the counter's row falls from 1 at
        # support_tol (1 cm) to 0 at 3 cm, each cup's from 1 at 2 cm to 0 at 4.5 cm, and the missing row stays 0.
        spec = dict(support_tol=0.01, penetration_tol=0.002, bounds_tol=0.02, bounds=[[-3, -3, -0.1], [3, 3, 4]])

        def placed(lift):
            cups = "".join(
                body(f"cup__{i}", box("0 0 0", ".04 .04 .05"), True, f'pos="{x} 0 {0.95 + lift}"')
                for i, x in enumerate((-0.2, 0.2))
            )
            s = self.scene(body("counter__0", box(f"0 0 {0.45 + lift}", ".4 .3 .45")) + cups)
            out = L.placement(spec, [{"slot": "lamp__0"}], s, {n: n for n in s.objects})
            return out, {r["object"]: r for r in out["objects"]}

        for lift, root, cup in ((0, 1, 1), (0.0101, 0.995, 1), (0.015, 0.75, 1), (0.03, 0, 0.6), (0.05, 0, 0)):
            with self.subTest(lift=lift):
                out, rows = placed(lift)
                self.assertAlmostEqual(rows["counter__0"]["score"], root, places=6)
                for n in ("cup__0", "cup__1"):
                    self.assertAlmostEqual(rows[n]["score"], cup, places=6)
                self.assertEqual([r["passed"] for r in rows.values()], [lift == 0] * 3 + [False])
                self.assertEqual(rows["lamp__0"]["score"], 0)
                self.assertAlmostEqual(out["score"], (root + 2 * cup) / 4, places=6)
                if lift:
                    self.assertEqual(rows["cup__1"]["support_root"], "counter__0")
        spec["bounds"][1][0] = 0.35  # the counter now pokes out of the room: its row is 0
        rows = placed(0.015)[1]
        self.assertEqual(rows["counter__0"]["score"], 0)
        self.assertAlmostEqual(rows["cup__1"]["score"], 1)
        spec["bounds"][1][0] = 3  # a root that sinks 1 cm sideways into a fixed block fails its own row
        cups = "".join(
            body(f"cup__{i}", box("0 0 0", ".04 .04 .05"), True, f'pos="{x} 0 {0.965}"')
            for i, x in enumerate((-0.2, 0.2))
        )
        s = self.scene(
            body("counter__0", box("0 0 .465", ".4 .3 .45")) + body("block__0", box(".45 0 .45", ".06 .3 .45")) + cups
        )
        rows = {
            r["object"]: r for r in L.placement(spec, [], s, {n: n for n in s.objects})["objects"]
        }  # but still passes its gap
        self.assertEqual(
            (rows["counter__0"]["non_penetration"], rows["counter__0"]["score"]), (False, 0)
        )  # (1.5 cm) to what
        for n in ("cup__0", "cup__1"):
            self.assertAlmostEqual(rows[n]["score"], 1)  # rests on it, by design

    def test_scope_averages_row_scores(self):
        from sbv import scope as SC

        case = dict(
            objects=[dict(slot=s) for s in "ab"], geometry=dict(inventory=dict(permitted_extras=[]), pose_size=[])
        )
        place = [
            dict(object="x", support=False, non_penetration=True, in_bounds=True, passed=False, score=0.75),
            dict(object="y", support=True, non_penetration=True, in_bounds=True, passed=True),
        ]  # no score: a report from before
        rep = dict(
            geometry=dict(
                G2=dict(per_object={}, pose_size=dict(objects=[])),
                G3=dict(relations=dict(relations=[]), placement=dict(objects=place), navigation=dict(routes=[])),
            )
        )
        out = SC.readouts(
            case, rep, [], [], dict(x="counter", y="cup"), dict(a="x", b="y"), {}, [], {}, 1.0, {"a", "b"}
        )
        self.assertAlmostEqual(out["geometry"]["G3"]["placement"]["score"], 0.875)

    def cycle_scene(self, anchored=False):
        a = box("-.3 0 1.05", ".1 .1 .05") + box(".3 0 1.2", ".1 .1 .05")
        b = box("-.3 0 .95", ".1 .1 .05") + box(".3 0 1.3", ".1 .1 .05")
        if anchored:
            b += box("-.3 0 .45", ".1 .1 .45")
        return self.scene(body("a__0", a) + body("b__0", b))

    def test_floating_interlocked_fixed_cycle_is_not_a_root(self):
        s = self.cycle_scene()
        parents, paths = SupportIndex(s, 0.01).graph()
        self.assertEqual(parents, {"a__0": {"b__0"}, "b__0": {"a__0"}})
        self.assertEqual(paths, {"world": ["world"]})
        rows = self.rows(s)
        self.assertTrue(all(r["support_local"] for r in rows.values()))
        self.assertFalse(any(r["support"] for r in rows.values()))
        self.assertFalse(
            any(r["support"] for r in self.rows(s, [{"a": "a__0", "b": "b__0"}, {"a": "b__0", "b": "a__0"}]).values())
        )

    def test_supported_cycle_is_allowed_and_losing_anchor_breaks_it(self):
        s = self.cycle_scene(True)
        self.assertTrue(all(r["support"] for r in self.rows(s).values()))
        self.assertEqual(rooted_paths({"a": {"b"}, "b": {"a", "world"}})["a"], ["a", "b", "world"])
        self.assertNotIn("a", rooted_paths({"a": {"b"}, "b": {"a"}}))

    def test_hook_is_supported_above_its_lowest_point(self):
        beam = box("0 0 1", ".2 .2 .05")
        hook = box("0 0 1.1", ".3 .2 .05") + "".join(box(f"{x} 0 .7", ".04 .2 .35") for x in (-0.26, 0.26))
        s = self.scene(body("hook__0", hook, True), floor=FLOOR + beam)
        self.assertTrue(self.rows(s)["hook__0"]["support"])
        self.assertTrue(P.init_record(s)["hook__0"]["still"])

    def test_side_touch_needs_frozen_mount_and_real_interface(self):
        wall = box("0 0 1", ".05 .5 1")
        mount = [{"a": "shelf__0", "b": "world"}]
        for free in (False, True):
            s = self.scene(body("shelf__0", box(".15 0 1", ".1 .2 .05"), free), floor=FLOOR + wall)
            self.assertFalse(self.rows(s)["shelf__0"]["support"])
            self.assertEqual(self.rows(s, mount)["shelf__0"]["support"], not free)
        s = self.scene(body("shelf__0", box(".3 0 1", ".1 .2 .05")), floor=FLOOR + wall)
        self.assertFalse(self.rows(s, mount)["shelf__0"]["support"])

    def wall_scene(self):
        fixed = lambda n, pos, size: body(n, box(pos, size))
        return self.scene(
            fixed("clock__0", ".08 0 1.5", ".03 .1 .1")  # flush on the wall
            + fixed("shelf__0", ".15 0 1", ".1 .2 .02")  # flush on the wall
            + body("book__0", box(".15 0 1.07", ".03 .1 .05"), True)  # free, on the shelf
            + fixed("lamp__0", ".2 0 .93", ".03 .03 .05")  # hangs under the shelf only
            + fixed("sign__0", ".09 .35 1.5", ".01 .1 .1")  # 3 cm off the wall
            + fixed("vent__0", ".06 -.35 1.5", ".02 .1 .1"),
            floor=FLOOR + box("0 0 1", ".05 .5 1"),
        )  # 1 cm into it

    def test_calibration_mounts_fixed_objects_that_touch(self):
        s = self.wall_scene()
        spec = dict(support_tol=0.01, penetration_tol=0.002, mounts=[])
        self.assertEqual(
            L.mounts(spec, s),
            [{"a": "clock__0", "b": "world"}, {"a": "shelf__0", "b": "world"}, {"a": "lamp__0", "b": "shelf__0"}],
        )
        given = [{"a": "sign__0", "b": "world"}]  # --mounts come first and stay
        self.assertEqual(L.mounts(dict(spec, mounts=given), s)[0], given[0])
        self.assertFalse(any(r["passed"] for r in self.rows(s).values()))  # before: all six exempt
        rows = self.rows(s, L.mounts(spec, s))
        self.assertEqual(sorted(n for n, r in rows.items() if not r["passed"]), ["sign__0", "vent__0"])

    def test_remount_writes_what_calibration_would(
        self,
    ):  # sbv.remount: a case calibrated before the mounts, updated in place
        from sbv.remount import remount

        s = self.wall_scene()
        spec = dict(
            support_tol=0.01, penetration_tol=0.002, bounds_tol=0.02, bounds=[[-3, -3, -0.1], [3, 3, 4]], mounts=[]
        )
        case = dict(
            objects=[dict(slot=n) for n in sorted(s.objects)],
            geometry=dict(placement=dict(spec, exempt=sorted(s.objects))),
        )
        place = remount(case, s)
        self.assertEqual((place["mounts"], place["exempt"]), (L.mounts(spec, s), ["sign__0", "vent__0"]))
        self.assertEqual(
            remount(case, s, [{"a": "sign__0", "b": "world"}])["mounts"][0], {"a": "sign__0", "b": "world"}
        )  # given first

    def test_mounts_use_reference_correspondence(self):
        s = self.scene(body("shelf_copy__0", box(".15 0 1", ".1 .2 .05")), floor=FLOOR + box("0 0 1", ".05 .5 1"))
        row = self.rows(s, [{"a": "reference_shelf", "b": "world"}], {"reference_shelf": "shelf_copy__0"})[
            "shelf_copy__0"
        ]
        self.assertTrue(row["support"])

    def test_penetration_supports_only_from_above(self):
        # A counter base sunk 4 cm into the floor still holds up what stands on it; a crate pushed 3 cm sideways into a
        # low wall, and a shelf on a coat stand's pole that pokes 1 cm through it, hold up nothing. Without pen, the
        # shelf held up its box. Every penetrating object still fails non_penetration.
        s = self.scene(
            body("base__0", box("0 0 .26", ".4 .3 .3"))
            + body("item__0", box("0 0 0", ".1 .1 .05"), True, 'pos="0 0 .61"')
        )
        rows = self.rows(s)
        x = s.d.geom_xpos.copy()
        self.assertEqual(
            [(rows[n]["support"], rows[n]["non_penetration"]) for n in ("base__0", "item__0")],
            [(True, False), (True, True)],
        )
        self.assertIn("item__0", SupportIndex(s, 0.01, 0.002).graph()[1])
        np.testing.assert_array_equal(x, s.d.geom_xpos)
        s = self.scene(
            body("crate__0", box(".12 0 1", ".1 .2 .1")), floor=FLOOR + box("0 0 .5", ".05 .5 .5", 'name="wall"')
        )
        crate, wall = s.ogeoms["crate__0"][0], mujoco.mj_name2id(s.m, mujoco.mjtObj.mjOBJ_GEOM, "wall")
        self.assertEqual(
            (self.rows(s)["crate__0"]["support"], SupportIndex(s, 0.01, 0.002).sunk_from_above(crate, wall)),
            (False, False),
        )
        s = self.scene(
            body("coat_stand__0", box("0 0 .515", ".02 .02 .515"))
            + body("shelf__0", box("0 0 1", ".3 .15 .02"))
            + body("box__0", box("0 0 0", ".05 .05 .05"), True, 'pos=".2 0 1.07"')
        )
        self.assertIn("box__0", SupportIndex(s, 0.01).graph()[1])
        rows = self.rows(s)
        self.assertEqual([rows[n]["support"] for n in ("coat_stand__0", "shelf__0", "box__0")], [True, False, False])
        self.assertFalse(rows["shelf__0"]["non_penetration"])

    def test_penetration_lift_factor_and_every_deep_part(self):
        # A crate sunk 5 mm into a ramp tilted 20 degrees rests from above, but straight up it is 5.3 mm deep: lifting
        # by the depth alone leaves it 0.3 mm in, lifting by depth / cos 30 frees it. A table whose leg is sunk 5 mm
        # into a block from above and whose side panel is pushed 3 cm into the same block from the side does not rest:
        # the panel is still inside after the lift, so the leg alone cannot hold the table up.
        ramp = box("0 0 .3", ".15 .15 .01", 'name="ramp" euler="0 .349066 0"')
        n = np.array([np.sin(0.349066), 0, np.cos(0.349066)])
        c = np.array([0, 0, 0.3]) + n * (0.01 + 0.05 - 0.005)
        s = self.scene(
            body("crate__0", box(" ".join(f"{v:.6f}" for v in c), ".05 .05 .05", 'euler="0 .349066 0"')),
            floor=FLOOR + ramp,
        )
        crate, ramp = s.ogeoms["crate__0"][0], mujoco.mj_name2id(s.m, mujoco.mjtObj.mjOBJ_GEOM, "ramp")
        self.assertTrue(SupportIndex(s, 0.01, 0.002).sunk_from_above(crate, ramp))
        self.assertTrue(self.rows(s)["crate__0"]["support"])
        legs = box("0 0 .395", ".05 .05 .1") + box(".185 0 .25", ".015 .1 .15")
        s = self.scene(body("table__0", legs), floor=FLOOR + box("0 0 .15", ".2 .2 .15", 'name="block"'))
        self.assertIn("table__0", SupportIndex(s, 0.01).graph()[1])  # without pen, the leg holds it up
        self.assertFalse(self.rows(s)["table__0"]["support"])

    def test_visual_support_and_disabled_collision_pair_are_not_support(self):
        for attrs in ('contype="0" conaffinity="0"', 'contype="2" conaffinity="2"'):
            s = self.scene(
                body("stand__0", box("0 0 .25", ".2 .2 .25", attrs))
                + body("item__0", box("0 0 .55", ".1 .1 .05"), True)
            )
            self.assertFalse(self.rows(s)["item__0"]["support"])

    def test_upside_contact_does_not_support(self):
        s = self.scene(body("item__0", box("0 0 .5", ".1 .1 .1"), True), floor=FLOOR + box("0 0 .65", ".5 .5 .05"))
        self.assertFalse(self.rows(s)["item__0"]["support"])

    def test_geometric_contact_is_not_a_stability_certificate(self):
        stand = body("stand__0", box("0 0 .25", ".1 .2 .25"))
        for x, still in ((0, True), (0.25, False)):
            with self.subTest(x=x):
                s = self.scene(stand + body("item__0", box(f"{x} 0 .55", ".3 .15 .05"), True))
                self.assertTrue(self.rows(s)["item__0"]["support"])
                self.assertEqual(P.init_record(s)["item__0"]["still"], still)

    def test_distance_queries_do_not_mutate_the_scene(self):
        s = self.cycle_scene(True)
        before = (s.d.qpos.copy(), s.d.geom_xpos.copy())
        SupportIndex(s, 0.01).graph()
        np.testing.assert_array_equal(before[0], s.d.qpos)
        np.testing.assert_array_equal(before[1], s.d.geom_xpos)

    def test_friction_support_needs_load_and_rest_not_just_side_contact(self):
        for mu, expected in ((0, False), (0.1, True), (0.5, True)):
            with self.subTest(friction=mu):
                walls = "".join(box(f"{x} 0 .5", ".02 .2 .3", f'friction="{mu} .005 .0001"') for x in (-0.1198, 0.1198))
                item = body(
                    "held__0", box("0 0 0", ".1 .1 .1", f'mass=".1" friction="{mu} .005 .0001"'), True, 'pos="0 0 .5"'
                )
                s = self.scene(item, floor=FLOOR + walls)
                s.step(0.1)
                s.start = s.d.qpos.copy()
                s.reset()
                self.assertEqual("held__0" in SupportIndex(s, 0.01).graph()[1], expected)
                self.assertEqual(("world", "held__0") in C.support_pairs(s), expected)

    def test_contact_policy_exclude_and_explicit_pair(self):
        s = self.scene(
            body("stand__0", box("0 0 .25", ".2 .2 .25")) + body("item__0", box("0 0 .55", ".1 .1 .05"), True)
        )
        xml = (
            Path(s.xml)
            .read_text()
            .replace("</mujoco>", '<contact><exclude body1="stand__0" body2="item__0"/></contact></mujoco>')
        )
        Path(s.xml).write_text(xml)
        s = Scene(s.xml)
        self.assertFalse(self.rows(s)["item__0"]["support"])
        xml = xml.replace('pos="0 0 .25"', 'name="stand_geom" pos="0 0 .25"').replace(
            'pos="0 0 .55"', 'name="item_geom" pos="0 0 .55"'
        )
        xml = xml.replace("</contact>", '<pair geom1="stand_geom" geom2="item_geom"/></contact>')
        Path(s.xml).write_text(xml)
        s = Scene(s.xml)
        self.assertTrue(self.rows(s)["item__0"]["support"])

    def test_remove_cycle_does_not_watch_removed_object_itself(self):
        s = self.scene(body("a__0", box("0 0 .1", ".1 .1 .1"), True) + body("b__0", box("0 0 .3", ".1 .1 .1"), True))
        ctx = dict(
            pairs=[("world", "a__0"), ("a__0", "b__0"), ("b__0", "a__0")],
            under={"a__0": ["world", "b__0"], "b__0": ["a__0"]},
        )
        events = C.cal_remove(s, ctx)
        self.assertEqual([u["slots"] for u, _ in events], [["a__0", "b__0"], ["b__0", "a__0"]])

    def test_convex_mesh_and_round_object_support(self):
        s = self.scene(
            body("ball__0", '<geom type="sphere" size=".1"/>', True, 'pos="0 0 .6"')
            + body("stand__0", box("0 0 .25", ".2 .2 .25"))
        )
        self.assertTrue(self.rows(s)["ball__0"]["support"])
        xml = Path(s.xml).read_text().replace('type="sphere" size=".1"', 'type="mesh" mesh="cube"')
        verts = " ".join(f"{x} {y} {z}" for x in (-0.1, 0.1) for y in (-0.1, 0.1) for z in (-0.1, 0.1))
        xml = xml.replace("<worldbody>", f'<asset><mesh name="cube" vertex="{verts}"/></asset><worldbody>')
        Path(s.xml).write_text(xml)
        s = Scene(s.xml)
        self.assertTrue(self.rows(s)["ball__0"]["support"])

    def test_calibration_rejects_nonfixed_mount_declarations(self):
        from sbv.calibrate import build

        s = self.scene(body("item__0", box("0 0 .1", ".1 .1 .1"), True))
        with self.assertRaisesRegex(ValueError, "fixed reference object"):
            build(s.xml, mounts=[{"a": "item__0", "b": "world"}])

    def test_calibration_rejects_malformed_and_ungrounded_mounts(self):
        from sbv.calibrate import build

        s = self.cycle_scene()
        for mounts in (
            {},
            [{"a": "a__0", "b": []}],
            [{"a": "a__0", "b": "world"}],
            [{"a": "a__0", "b": "b__0"}, {"a": "b__0", "b": "a__0"}],
        ):
            with self.subTest(mounts=mounts), self.assertRaises(ValueError):
                build(s.xml, mounts=mounts)


if __name__ == "__main__":
    unittest.main()
