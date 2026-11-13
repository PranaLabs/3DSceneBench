"""The near-camera warning (appearance.near_camera, never scored) lists only what the copy puts nearer to a held-out
camera than the reference room has along the same ray: the reference room's own furniture, copied faithfully, is no
warning."""

import unittest
import numpy as np
from sbv.appearance.views import NEAR_SLACK, camera, near_cameras
from tests.test_input_cameras import scene, ROOM

CAM = camera("view_same_0", np.array([-1.5, -1.5, 1.2]), np.array([1.0, -1.5, 1.2]), 60, 512)  # looking along +x


def room(crate_face=0.35, board=False):
    """The fixture room with a crate whose face stands crate_face metres in front of CAM, and optionally a picture-only
    board 10 cm in front of it."""
    x = -1.5 + crate_face + 0.05
    extra = f'<body name="crate__0" pos="{x} -1.5 1.2"><geom name="crate_g" type="box" size="0.05 0.3 0.3"/></body>'
    if board:
        extra += '<geom name="board" type="box" pos="-1.4 -1.5 1.2" size="0.005 0.3 0.3" contype="0" conaffinity="0"/>'
    s = scene(ROOM.replace("</worldbody>", extra + "</worldbody>"))
    s.reset(s.q0)
    return s


class NearCamera(unittest.TestCase):
    def setUp(self):
        self.true = room()

    def warned(self, copy):
        return [(w["camera"], w["geom"]) for w in near_cameras([CAM], copy, true=self.true)]

    def test_the_true_room_copied_is_no_warning(self):
        self.assertEqual(self.warned(room()), [])
        self.assertEqual(
            [w["geom"] for w in near_cameras([CAM], room())], ["crate_g"]
        )  # without the reference room: warned, as before

    def test_a_board_is_warned_even_in_front_of_true_furniture(self):
        w = near_cameras([CAM], room(board=True), true=self.true)
        self.assertEqual([(x["geom"], x["true_distance"]) for x in w], [("board", 0.35)])
        self.assertAlmostEqual(w[0]["distance"], 0.095, places=3)

    def test_furniture_moved_nearer_than_the_slack_is_warned(self):
        self.assertEqual(self.warned(room(crate_face=0.35 - NEAR_SLACK / 2)), [])  # 5 cm nearer: placement noise
        self.assertEqual(self.warned(room(crate_face=0.15)), [("view_same_0", "crate_g")])  # 20 cm nearer


if __name__ == "__main__":
    unittest.main()
