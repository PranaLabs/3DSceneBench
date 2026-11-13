"""Input and held-out cameras stand where they see the room, even when the room bounds reach past a wall."""

import os, tempfile, unittest
from sbv.common.scene import Scene
from sbv.appearance.views import cameras, sees

ROOM = """<mujoco><worldbody>
  <geom name="floor" type="plane" size="6 6 0.1"/>
  <geom name="wall_n" type="box" pos="0 3 1.25" size="3 0.05 1.25"/><geom name="wall_s" type="box" pos="0 -3 1.25" size="3 0.05 1.25"/>
  <geom name="wall_e" type="box" pos="3 0 1.25" size="0.05 3 1.25"/><geom name="wall_w" type="box" pos="-3 0 1.25" size="0.05 3 1.25"/>
  <body name="table__0" pos="0.5 0.5 0.4"><geom type="box" size="0.4 0.3 0.4"/></body>
</worldbody></mujoco>"""
BOUNDS = [[-6.0, -6.0, -0.05], [6.0, 6.0, 3.0]]  # the floor and the bounds reach 3 m past every wall


def scene(text):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "scene.xml")
    with open(p, "w") as f:
        f.write(text)
    return Scene(p)


class InputCameras(unittest.TestCase):
    def test_cameras_stand_inside_the_walls(self):
        s = scene(ROOM)
        for kind in ("input", "held_out"):
            for seed in (None, 1, 2):
                for cam in cameras(s, BOUNDS, kind=kind, seed=seed):
                    x, y, z = cam["position"]
                    self.assertTrue(abs(x) < 3 and abs(y) < 3, (kind, seed, cam["name"], cam["position"]))
                    self.assertTrue(sees(s, (x, y), z, cam["look_at"]))

    def test_eight_input_cameras_keep_the_first_four(self):
        """The 8- and 12-picture conditions add input_4-7 and input_8-11 halfway from each wall's middle to its corners;
        input_0-3 (the 1, 2 and 4 picture conditions) stay where they were, and no input camera stands where a held-out
        one does."""
        from sbv.appearance.views import CONDITIONS

        s = scene(ROOM)
        cams = cameras(s, BOUNDS, height=1.6, kind="input")
        held = cameras(s, BOUNDS, kind="held_out")
        self.assertEqual([c["name"] for c in cams], [f"input_{k}" for k in range(12)])
        self.assertEqual(CONDITIONS["12"], [c["name"] for c in cams])
        self.assertEqual(CONDITIONS["8"], [c["name"] for c in cams[:8]])
        self.assertEqual(CONDITIONS["4"], [c["name"] for c in cams[:4]])
        middles = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # input_0-3 face the room from each wall's middle
        for cam, (u, v) in zip(cams[:4], middles):
            x, y, _ = cam["position"]
            self.assertTrue(x * u + y * v > 1.5 and abs(x * v - y * u) < 1.0, (cam["name"], cam["position"]))
        spots = {tuple(c["position"][:2]) for c in held}
        self.assertEqual(
            len({tuple(c["position"][:2]) for c in cams} | spots), 16
        )  # twelve inputs, four held-out, no shared spot

    def test_the_extra_four_show_what_the_first_eight_miss(self):
        """A cup hidden behind a tall partition near a corner: no camera aimed at the centre shows it, so one of the
        12-picture condition's input_8-11 is aimed at it; input_0-7 stay as they were."""
        import numpy as np
        from sbv.appearance.views import shown

        room = ROOM.replace(
            "</worldbody>",
            '<body name="bookcase__0" pos="2.2 2.0 1.0"><geom type="box" size="0.05 0.9 1.0"/></body>'
            '<body name="cup__0" pos="2.7 2.6 0.05"><freejoint/><geom type="box" size="0.04 0.04 0.05" mass="0.2"/></body></worldbody>',
        )
        s = scene(room)
        s.settle()
        b = [[-3.0, -3.0, -0.05], [3.0, 3.0, 3.0]]
        cams = cameras(s, b, height=1.6, kind="input")
        self.assertEqual([c["name"] for c in cams], [f"input_{k}" for k in range(12)])
        see = lambda c: shown(s, np.array(c["position"]), np.array(c["look_at"]), c["hfov_deg"], ["cup__0"])
        self.assertFalse(any(see(c) for c in cams[:8]))
        self.assertTrue(any(see(c) for c in cams[8:]))
        held = cameras(s, b, kind="held_out")
        self.assertEqual(len({tuple(c["position"][:2]) for c in cams + held}), 16)

    def test_twelve_input_cameras_however_many_cover_finds(self):
        """cover() may find 0 to 4 of input_8-11; the halfway-back anchors fill the rest. With 2 or 3 found, input_9 and
        input_10 used to be added again from their anchors: 13 or 14 cameras, two of them named alike."""
        import numpy as np
        from unittest.mock import patch
        from sbv.appearance import views as V

        s = scene(ROOM)
        plain = cameras(s, BOUNDS, height=1.6, kind="input")
        for found in range(5):

            def cover(*a, **k):
                for j in range(found):
                    yield np.array([-2.0, -2.0 + 0.8 * j, 1.6]), np.array([0.0, 0.0, 0.8])
                if found < 4:
                    yield None

            with patch.object(V, "cover", cover):
                cams = cameras(s, BOUNDS, height=1.6, kind="input")
            self.assertEqual([c["name"] for c in cams], [f"input_{k}" for k in range(12)], found)
            self.assertEqual(cams[:8], plain[:8], found)  # the 1-8 picture conditions keep theirs
            self.assertEqual(
                [c["position"][:2] for c in cams[8 : 8 + found]],
                [[-2.0, round(-2.0 + 0.8 * j, 3)] for j in range(found)],
                found,
            )

    def test_without_the_check_a_camera_would_stand_outside(self):
        """The room this guards against: the nearest free floor to a wall's middle lies behind the wall."""
        s = scene(ROOM)
        self.assertFalse(sees(s, (0.0, 5.7), 1.6, [0.0, 0.0, 0.8]))
        self.assertTrue(sees(s, (0.0, 2.5), 1.6, [0.0, 0.0, 0.8]))


if __name__ == "__main__":
    unittest.main()
