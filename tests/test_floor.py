import tempfile, unittest
from pathlib import Path
import numpy as np
from sbv.common.scene import Scene
from sbv.geometry.g3_layout import floor_plan, ground_mask, occupancy


class FloorSupport(unittest.TestCase):
    def scene(self, geoms):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        p = Path(t.name) / "s.xml"
        p.write_text(f"<mujoco><worldbody>{geoms}</worldbody></mujoco>")
        return Scene(str(p))

    def plan(self, s):
        return floor_plan(s, [[-1, -1, -0.1], [1, 1, 2]], 0.15)

    def cell(self, r, x, y):
        xs, ys, v = r
        return bool(v[np.argmin(abs(xs - x)), np.argmin(abs(ys - y))])

    def test_void_not_free_and_finite_floor_edge_eroded(self):
        s = self.scene('<geom name="floor" type="box" pos="0 0 -.05" size=".5 .5 .05"/>')
        r = self.plan(s)
        self.assertTrue(self.cell(r, 0, 0))
        self.assertFalse(self.cell(r, 0.7, 0))
        self.assertFalse(self.cell(r, 0.43, 0))

    def test_empty_scene_no_ground(self):
        self.assertFalse(self.plan(self.scene(""))[2].any())

    def test_visual_floor_does_not_support(self):
        self.assertFalse(
            self.plan(self.scene('<geom type="box" pos="0 0 -.05" size="1 1 .05" contype="0" conaffinity="0"/>'))[
                2
            ].any()
        )

    def test_visual_decoration_does_not_obstruct(self):
        r = self.plan(
            self.scene(
                '<geom type="plane" size="2 2 .1"/><geom type="box" pos="0 0 .5" size=".3 .3 .1" contype="0" conaffinity="0"/>'
            )
        )
        self.assertTrue(self.cell(r, 0, 0))

    def test_table_is_not_floor(self):
        self.assertFalse(
            self.plan(self.scene('<body name="table__0"><geom type="box" pos="0 0 .7" size="1 1 .05"/></body>'))[
                2
            ].any()
        )

    def test_named_but_movable_floor_is_not_ground(self):
        self.assertFalse(
            self.plan(
                self.scene('<body name="floor__0"><freejoint/><geom type="box" pos="0 0 -.05" size="1 1 .05"/></body>')
            )[2].any()
        )

    def test_wrong_floor_height_is_not_silently_accepted(self):
        self.assertFalse(self.plan(self.scene('<geom type="box" pos="0 0 -.15" size="1 1 .05"/>'))[2].any())

    def test_no_start_has_actionable_error(self):
        from sbv.geometry.g3_layout import navigation_start

        with self.assertRaisesRegex(ValueError, "no supported navigation start"):
            navigation_start(self.scene(""), [[-1, -1, 0], [1, 1, 2]], 0.3, [0, 0])

    def test_hole_and_infinite_wall_plane(self):
        s = self.scene(
            '<geom type="box" pos="-.6 0 -.05" size=".4 1 .05"/><geom type="box" pos=".6 0 -.05" size=".4 1 .05"/>'
        )
        r = self.plan(s)
        self.assertFalse(self.cell(r, 0, 0))
        self.assertTrue(self.cell(r, 0.6, 0))
        s = self.scene('<geom type="plane" size="2 2 .1"/><geom type="plane" size="1 1 .1" zaxis="1 0 0"/>')
        r = self.plan(s)
        self.assertFalse(self.cell(r, -0.5, 0))
        self.assertTrue(self.cell(r, 0.5, 0))

    def test_floor_body_name_does_not_matter(self):
        for name in ("floor", "floor_tiles", "ground", "room_geometry_floor"):
            s = self.scene(f'<body name="{name}"><geom type="box" pos="0 0 -.05" size=".5 .5 .05"/></body>')
            self.assertTrue(self.cell(self.plan(s), 0, 0), name)

    def test_robot_start_and_heading_maps_reject_void(self):
        from sbv.functionality.f4_robot.plan import floor_map, yaw_maps, nearest_free

        s = self.scene('<geom type="box" pos="0 0 -.05" size=".6 .6 .05"/>')
        xs, ys, occ = floor_map(s, [[-2, -2, -0.1], [2, 2, 2]])
        free = yaw_maps(occ)
        self.assertIsNone(nearest_free(xs, ys, free, [1.5, 1.5], within=0.5))
        self.assertIsNotNone(nearest_free(xs, ys, free, [0, 0]))
        for k in range(len(free)):
            self.assertFalse(self.cell((xs, ys, free[k]), 0.5, 0))

    def test_ground_mask_same_as_every_box_against_every_cell(self):
        """The mask the old loop gave (every candidate box against every cell, kept here), on offset, rotated, raised,
        tilted and plane floors."""
        import mujoco
        from sbv.common.rays import first_hit
        from sbv.geometry.g3_layout import CELL, boxes

        def old(s, xs, ys):
            m = s.m
            gs = [g for g in range(m.ngeom) if s.collides(g) and m.body_weldid[m.geom_bodyid[g]] == 0]
            b = boxes(s, gs)
            planes = {g for g in gs if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE}
            out = np.zeros((len(xs), len(ys)), bool)
            cand = [g for g in gs if g in planes or b[g][0][2] <= 0.02 and b[g][1][2] >= -0.02]
            for i, x in enumerate(xs):
                for j, y in enumerate(ys):
                    near = [
                        g
                        for g in cand
                        if g in planes
                        or (b[g][0][:2] <= [x + CELL / 2, y + CELL / 2]).all()
                        and (b[g][1][:2] >= [x - CELL / 2, y - CELL / 2]).all()
                    ]
                    hits = (
                        [
                            first_hit(
                                s, near, np.array([x + dx, y + dy, 0.0 + 0.02 + 0.001]), np.array([0.0, 0.0, -1.0])
                            )
                            for dx in (0, -CELL / 2, CELL / 2)
                            for dy in (0, -CELL / 2, CELL / 2)
                            if (dx == 0) == (dy == 0)
                        ]
                        if near
                        else [None]
                    )
                    out[i, j] = all(
                        h is not None and abs(0.02 + 0.001 - h[0]) <= 0.02 and h[2][2] >= 0.98 for h in hits
                    )
            return out

        rooms = [
            '<geom type="box" pos="-.5 0 -.05" size=".5 .5 .05"/><geom type="box" pos=".52 .1 -.04" size=".4 .3 .05"/>'
            '<geom type="box" pos=".3 -.6 .005" size=".2 .2 .005" euler="0 0 30"/><geom type="box" pos="-.3 .6 .2" size=".3 .3 .2"/>'
            '<geom type="cylinder" pos=".7 .7 -.05" size=".25 .05"/><geom type="box" pos="0 0 -.06" size=".1 .1 .05" euler="0 20 0"/>',
            '<geom type="plane" size="2 2 .1"/><geom type="box" pos=".2 .1 0" size=".3 .3 .01" euler="0 20 0"/><geom type="plane" size="1 1 .1" zaxis="1 0 0" pos="-.6 0 0"/>',
        ]
        for r in rooms:
            for lo, hi in (
                ([-1.13, -0.97, -0.1], [1.21, 1.04, 2]),
                ([-1, -1, -0.1], [1, 1, 2]),
            ):  # cell edges off and on the box edges
                s = self.scene(r)
                s.reset(s.q0)
                xs, ys, _ = occupancy(s, [lo, hi])
                want = old(s, xs, ys)
                self.assertTrue(want.any() and not want.all())
                np.testing.assert_array_equal(ground_mask(s, xs, ys), want)
