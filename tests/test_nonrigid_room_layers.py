"""The non-rigid runs' room cut (sbv/nonrigid/room.py) on a scene whose material names its texture in a <layer>, as
MuJoCo 3 materials may: the room is measured without its pictures, and a cut that keeps looks keeps that texture."""

import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from sbv.nonrigid import room

SCENE = """<mujoco><asset>
  <texture name="wood" type="2d" file="wood.png"/>
  <material name="table_mat"><layer texture="wood" role="rgb"/></material>
</asset><worldbody>
  <geom name="floor" type="plane" size="3 3 .1"/>
  <body name="table__0" pos="0 0 .4"><geom type="box" size=".5 .4 .4" material="table_mat"/></body>
</worldbody></mujoco>"""


class MaterialLayers(unittest.TestCase):
    def setUp(self):
        from PIL import Image

        self.dir = tempfile.mkdtemp()
        self.xml = os.path.join(self.dir, "scene.xml")
        Image.fromarray(np.full((4, 4, 3), 128, np.uint8)).save(os.path.join(self.dir, "wood.png"))
        with open(self.xml, "w") as f:
            f.write(SCENE)

    def test_the_room_is_measured_without_its_pictures(self):
        boxes = room._aabbs(self.xml)
        lo, hi = boxes["table__0"]
        np.testing.assert_allclose(hi - lo, [1.0, 0.8, 0.8], atol=1e-6)

    def test_a_cut_with_looks_keeps_a_layer_texture(self):
        r = room._lean(self.xml, keep={"table__0"}, looks=True)
        self.assertEqual([t.get("name") for t in r.iter("texture")], ["wood"])
        r = room._lean(self.xml, keep={"table__0"})
        self.assertEqual(ET.tostring(r).count(b"<layer"), 1)  # the material keeps its layer: its texture is kept too


OBJ = "v 0 0 0\nv .2 0 0\nv 0 .2 0\nv 0 0 .2\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n"
SCENE_DIRS = """<mujoco><compiler {comp}/><asset>
  <mesh name="cup" file="cup.obj"/>
  <texture name="wood" type="2d" file="wood.png"/>
  <material name="table_mat" texture="wood"/>
</asset><worldbody>
  <geom name="floor" type="plane" size="3 3 .1"/>
  <body name="table__0" pos="0 0 .4"><geom type="box" size=".5 .4 .4" material="table_mat"/></body>
  <body name="cup__0" pos="0 0 .81"><freejoint/><geom type="mesh" mesh="cup"/></body>
</worldbody></mujoco>"""


class CompilerDirectories(unittest.TestCase):
    def make(self, comp, mesh_dir, tex_dir):
        from PIL import Image

        d = tempfile.mkdtemp()
        for sub, write in (
            (mesh_dir, lambda p: open(os.path.join(p, "cup.obj"), "w").write(OBJ)),
            (tex_dir, lambda p: Image.fromarray(np.full((4, 4, 3), 128, np.uint8)).save(os.path.join(p, "wood.png"))),
        ):
            os.makedirs(os.path.join(d, sub), exist_ok=True)
            write(os.path.join(d, sub))
        with open(os.path.join(d, "scene.xml"), "w") as f:
            f.write(SCENE_DIRS.format(comp=comp))
        return os.path.join(d, "scene.xml")

    def check(self, xml):
        boxes = room._aabbs(xml)  # the room loads in MuJoCo through the cut
        self.assertIn("cup__0", boxes)
        out = tempfile.mkdtemp()
        paths = room.cut(xml, {"table__0", "cup__0"}, "cup__0", out, looks=True)
        for p in paths.values():  # the written files load from another folder: every path absolute, no compiler dirs
            import mujoco

            mujoco.MjModel.from_xml_path(p)
            self.assertTrue(all(os.path.isabs(e.get("file")) for e in ET.parse(p).getroot().iter() if e.get("file")))

    def test_meshdir_and_texturedir(self):
        self.check(self.make('meshdir="assets" texturedir="textures"', "assets", "textures"))

    def test_assetdir_is_the_fallback(self):
        self.check(self.make('assetdir="a"', "a", "a"))

    def test_no_compiler_directories(self):
        self.check(self.make('angle="radian"', "", ""))


if __name__ == "__main__":
    unittest.main()
