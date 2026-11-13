"""Merged wall meshes must not fill empty space; surfaces and closed interiors remain excluded."""

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import mujoco
import trimesh
from sbv.appearance.heldout import clearance


class VisualMeshClearance(unittest.TestCase):
    def room(self, collides=False, body=None, name="wall_visual"):
        # Face-split OBJ exports are common; processing must restore closed components without moving vertices.
        mesh = trimesh.util.concatenate(
            [
                trimesh.creation.box(
                    extents=[0.2, 4, 4], transform=trimesh.transformations.translation_matrix([x, 0, 0])
                )
                for x in (-2, 2)
            ]
        )
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        file = Path(temp.name, "walls.obj")
        mesh.export(file)
        top, end = (
            (f'<body name="{body}">', "</body>") if body else ("", "")
        )  # an object's or an architecture body, else the world body
        xml = f'<mujoco><asset><mesh name="walls" file="{file}"/></asset><worldbody>{top}<geom name="{name}" type="mesh" mesh="walls" contype="{int(collides)}" conaffinity="{int(collides)}"/>{end}</worldbody></mujoco>'
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        return SimpleNamespace(m=m, d=d)

    def test_between_disconnected_walls_is_clear(self):
        self.assertEqual(clearance(self.room(), [0, 0, 0]), [])

    def test_inside_a_closed_wall_is_rejected(self):
        self.assertIn("wall_visual", clearance(self.room(), [2, 0, 0]))

    def test_near_surface_still_uses_full_radius(self):
        self.assertIn("wall_visual", clearance(self.room(), [1.78, 0, 0], 0.15))

    def test_deep_closed_interior_rejected_without_surface_nearness(self):
        r = self.room()
        self.assertIn("wall_visual", clearance(r, [2, 0, 0], 0.01))

    def test_an_objects_visual_shell_keeps_its_box(self):
        # protocol v3: only the room's own meshes (world-body geoms, architecture bodies) are judged by their surfaces;
        # an object's open visual-only shell (a wardrobe drawn without a back) still keeps every camera out of its box
        self.assertIn("wall_visual", clearance(self.room(body="wardrobe__0"), [0, 0, 0]))
        self.assertEqual(clearance(self.room(body="wall__0"), [0, 0, 0]), [])
        self.assertIn(
            "jacket__0_look", clearance(self.room(name="jacket__0_look"), [0, 0, 0])
        )  # an object's look on the world body
        self.assertEqual(clearance(self.room(name="wall__0_visual"), [0, 0, 0]), [])  # a wall's, on the world body

    def test_collision_mesh_keeps_conservative_exclusion(self):
        self.assertIn("wall_visual", clearance(self.room(True), [0, 0, 0]))

    def test_local_mesh_query_follows_geom_pose(self):
        r = self.room()
        r.m.geom_pos[0] = [0, 0, 10]
        mujoco.mj_forward(r.m, r.d)
        self.assertEqual(clearance(r, [0, 0, 10]), [])
        self.assertIn("wall_visual", clearance(r, [2, 0, 10]))
