"""F1.1 package check: a symlink cannot carry a file from outside the copy's folder into it, and includes cannot
loop."""

import tempfile, unittest
from pathlib import Path
from sbv.functionality.f1_gate import package


class PackageSymlinks(unittest.TestCase):
    def test_symlink_cannot_escape_or_be_the_main_scene(self):
        root = Path(tempfile.mkdtemp())
        outside = root / "outside.xml"
        outside.write_text("<mujoco/>")
        pkg = root / "pkg"
        pkg.mkdir()
        (pkg / "part.xml").symlink_to(outside)
        xml = pkg / "scene.xml"
        xml.write_text('<mujoco><include file="part.xml"/></mujoco>')
        self.assertFalse(package(str(xml))[0])  # an include that leads outside
        xml.unlink()
        xml.symlink_to(outside)
        self.assertFalse(package(str(xml))[0])  # the scene file itself a symlink

    def test_symlinked_mesh_outside_fails_inside_passes(self):
        root = Path(tempfile.mkdtemp())
        (root / "outer.obj").write_text("v 0 0 0\n")
        pkg = root / "pkg"
        pkg.mkdir()
        (pkg / "m.obj").symlink_to(root / "outer.obj")
        xml = pkg / "scene.xml"
        xml.write_text('<mujoco><asset><mesh file="m.obj"/></asset></mujoco>')
        ok, errors = package(str(xml))
        self.assertFalse(ok)
        self.assertIn("leaves the package", " ".join(errors))
        (pkg / "m.obj").unlink()
        (pkg / "real.obj").write_text("v 0 0 0\n")
        (pkg / "m.obj").symlink_to(pkg / "real.obj")
        self.assertTrue(package(str(xml))[0])  # a symlink that stays inside the folder is fine

    def test_include_loop_ends(self):
        pkg = Path(tempfile.mkdtemp())
        (pkg / "a.xml").write_text('<mujoco><include file="b.xml"/></mujoco>')
        (pkg / "b.xml").write_text('<mujoco><include file="a.xml"/></mujoco>')
        (pkg / "scene.xml").write_text('<mujoco><include file="a.xml"/></mujoco>')
        self.assertTrue(package(str(pkg / "scene.xml"))[0])  # no endless recursion; MuJoCo itself reports the loop


if __name__ == "__main__":
    unittest.main()
