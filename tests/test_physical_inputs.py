"""Physical cache inputs retain authored precision and ignore rendering-only edits."""

from pathlib import Path
import tempfile
import unittest
import numpy as np
import mujoco
from sbv.common.scene_inputs import physical


class PhysicalInputs(unittest.TestCase):
    def test_precise_physics_survives_the_view_and_a_small_change_invalidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            room = Path(tmp) / "room"
            room.mkdir()
            xml = room / "scene.xml"
            xml.write_text(
                '<mujoco><worldbody><body name="box__0" pos=".123456789123 0 .2"><freejoint/>'
                '<geom type="box" size=".123456789123 .1 .1" mass=".987654321987" rgba=".3 .4 .5 .6"/>'
                "</body></worldbody></mujoco>"
            )
            before = physical(room)
            (Path(tmp) / "view").mkdir()
            (Path(tmp) / "view/scene.xml").write_text(before["xml"])
            a = mujoco.MjModel.from_xml_path(str(xml))
            b = mujoco.MjModel.from_xml_path(str(Path(tmp) / "view/scene.xml"))
            for attr in ("body_mass", "body_inertia", "body_pos", "geom_size", "qpos0"):
                self.assertTrue(np.array_equal(getattr(a, attr), getattr(b, attr)), attr)
            xml.write_text(xml.read_text().replace(".987654321987", ".987654321988"))
            self.assertNotEqual(before, physical(room))

    def test_light_color_and_whitespace_do_not_invalidate_but_alpha_does(self):
        with tempfile.TemporaryDirectory() as tmp:
            xml = Path(tmp) / "scene.xml"
            xml.write_text('<mujoco><worldbody>\n<geom size=".1" rgba="1 0 0 1"/>\n</worldbody></mujoco>')
            before = physical(tmp)
            xml.write_text(
                xml.read_text().replace("<worldbody>", '<worldbody><light pos="0 0 1"/>').replace("1 0 0 1", "0 1 0 1")
            )
            self.assertEqual(before, physical(tmp))
            xml.write_text(xml.read_text().replace("0 1 0 1", "0 1 0 .5"))
            self.assertNotEqual(before, physical(tmp))

    def test_statistic_is_physics_and_invalidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            xml = Path(tmp) / "scene.xml"
            xml.write_text('<mujoco><statistic meaninertia="1"/><worldbody><geom size=".1"/></worldbody></mujoco>')
            before = physical(tmp)
            xml.write_text(xml.read_text().replace('meaninertia="1"', 'meaninertia="2"'))
            self.assertNotEqual(before, physical(tmp))
