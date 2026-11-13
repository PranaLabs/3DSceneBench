"""Appearance light: 5 cm under the lower of the true and copy ceilings, for the copy and the reference room alike, and
part of the reference cache key. A visual-only ceiling counts too."""

import copy, importlib.util, os, tempfile, unittest
from pathlib import Path
import numpy as np
from sbv.common.scene import Scene
from sbv.appearance import views as App

BOUNDS = [[-2, -2, -0.05], [2, 2, 3.2]]


def room(ceiling, solid=True, panel=None):
    """A closed 4 x 4 m box room whose ceiling underside is at `ceiling` (None: no ceiling); solid=False: panel: a
    visual-only 1 x 1 m board this high over the middle."""
    walls = "".join(
        f'<geom name="wall{k}" type="box" pos="{x} {y} 1.6" size="{sx} {sy} 1.6" rgba=".8 .8 .8 1"/>'
        for k, (x, y, sx, sy) in enumerate(
            [(2.05, 0, 0.05, 2.1), (-2.05, 0, 0.05, 2.1), (0, 2.05, 2.1, 0.05), (0, -2.05, 2.1, 0.05)]
        )
    )
    flags = "" if solid else ' contype="0" conaffinity="0"'
    top = (
        ""
        if ceiling is None
        else f'<geom name="ceiling" type="box" pos="0 0 {ceiling + 0.05}" size="2.1 2.1 .05" rgba=".9 .9 .9 1"{flags}/>'
    )
    if panel is not None:
        top += f'<geom name="panel" type="box" pos="0 0 {panel + 0.02}" size=".5 .5 .02" rgba=".9 .9 .9 1" contype="0" conaffinity="0"/>'
    return (
        f'<mujoco><worldbody><geom name="floor" type="box" pos="0 0 -.05" size="2.1 2.1 .05" rgba=".6 .5 .4 1"/>{walls}{top}'
        '<body name="table__0" pos="0 0 .4"><geom type="box" size=".4 .4 .4" rgba=".7 .2 .2 1"/></body></worldbody></mujoco>'
    )


class Light(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.case = dict(
            room=dict(bounds=BOUNDS),
            appearance=dict(light_z=2.95, cameras=[]),
            case_hash="h",
            reference=self.xml("true", 3.0),
        )

    def xml(self, name, ceiling, solid=True, panel=None):
        p = Path(self.tmp.name) / f"{name}.xml"
        p.write_text(room(ceiling, solid, panel))
        return str(p)

    def test_light_goes_under_the_lower_ceiling(self):
        for ceiling, z in ((2.80, 2.75), (3.10, 2.95), (None, 2.95), (2.95, 2.90)):
            s = Scene(self.xml("c", ceiling))
            s.reset(s.q0)
            self.assertAlmostEqual(App.light_for(self.case, s)["z"], z, places=6, msg=str(ceiling))

    def test_visual_only_ceiling_counts(self):
        for ceiling, z in ((2.80, 2.75), (3.10, 2.95)):
            s = Scene(self.xml("v", ceiling, solid=False))
            s.reset(s.q0)
            self.assertAlmostEqual(App.light_for(self.case, s)["z"], z, places=6, msg=str(ceiling))

    def test_small_drawn_piece_is_no_ceiling(self):
        """A visual-only board covering 6% of the room (a lamp, a panel) leaves the light under the real ceiling."""
        s = Scene(self.xml("p", 2.80, panel=2.40))
        s.reset(s.q0)
        self.assertAlmostEqual(App.light_for(self.case, s)["z"], 2.75, places=6)

    def test_light_height_keys_the_reference_cache(self):
        a, b = App.ref_key(self.case), App.ref_key(self.case, 2.75)
        self.assertNotEqual(a, b)
        self.assertEqual(a, App.ref_key(self.case, 2.95))  # the case's height is the default
        moved = copy.deepcopy(self.case)
        moved["appearance"]["light_z"] = 2.9
        self.assertNotEqual(a, App.ref_key(moved))

    @unittest.skipUnless(importlib.util.find_spec("bpy"), "needs Blender's Python module")
    def test_low_ceiling_copy_renders_lit(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("needs Pillow (requirements/core.txt) beside Blender's Python")
        cam = dict(name="view_0", position=[-1.5, -1.5, 1.4], look_at=[1, 1, 0.8], hfov_deg=70)

        def mean(z, out):
            spec = dict(bounds=BOUNDS, resolution=48, samples=4, light_z=z, cameras=[cam])
            App.render(xml, spec, os.path.join(self.tmp.name, out))
            return float(np.asarray(Image.open(os.path.join(self.tmp.name, out, "view_0.png"))).mean())

        for solid in (True, False):  # a colliding ceiling, and a visual-only one
            xml = self.xml(f"low{solid}", 2.80, solid)
            s = Scene(xml)
            s.reset(s.q0)
            z = App.light_for(self.case, s)["z"]
            old, new = mean(2.95, f"old{solid}"), mean(z, f"new{solid}")
            self.assertLess(old, 5, msg=str(solid))
            self.assertGreater(new, 40, msg=str(solid))  # light above the ceiling: black; under it: lit


if __name__ == "__main__":
    unittest.main()
