"""The appearance floor: the room's check writes it into the case; without Blender here, floor() says so and the case
keeps the fixed mapping. The mapping itself is in tests/test_reward.py."""

import copy, json, os, tempfile, unittest
from unittest import mock
from sbv import pipeline
from sbv.appearance import views as App
from sbv.common import case as K
from sbv.common.scene import PHYSICS
from tests import rooms

SCORES = dict(
    parse=1.0,
    sim_ready=1.0,
    reward=100.0,
    reward_geometry=100.0,
    reward_functionality=100.0,
    reward_functionality_hint=100.0,
    reward_balance=100.0,
    reward_response=100.0,
    reward_affordance=100.0,
    reward_interaction=100.0,
    inventory=100.0,
    shape=100.0,
    pose_size=100.0,
    relations=100.0,
    placement=100.0,
    navigation=100.0,
    siglip2=1.0,
    dinov3=1.0,
    lpips=0.0,
)


class AppearanceFloor(unittest.TestCase):
    def test_the_check_writes_the_floor_into_the_case(self):
        tmp = tempfile.mkdtemp(prefix="sbv_floor_")
        room = os.path.dirname(rooms.write(os.path.join(tmp, "room")))
        out = os.path.join(tmp, "out")
        case = dict(
            schema=K.SCHEMA,
            reference="room/scene.xml",
            physics=dict(PHYSICS),
            room=dict(bounds=[[0, 0, 0], [4, 3, 2.5]]),
            appearance=dict(resolution=512, samples=64, cameras=[]),
            objects=[],
            functionality=dict(units=[], excluded=[]),
        )
        floor = dict(siglip2=0.89, dinov3=0.65, lpips_vgg=0.27)
        with (
            mock.patch.object(pipeline.calibrate, "build", return_value=case),
            mock.patch.object(pipeline.Q, "score", return_value=dict(SCORES)),
            mock.patch.object(pipeline.App, "floor", return_value=floor) as fl,
        ):
            res = pipeline.check(room, out, log=lambda *a: None)
        self.assertTrue(res["ready"])
        fl.assert_called_once()
        with open(os.path.join(out, "case.json")) as f:
            self.assertEqual(json.load(f)["appearance"]["floor"], floor)
        with (
            mock.patch.object(
                pipeline.calibrate,
                "build",
                return_value=copy.deepcopy(
                    dict(case, appearance={k: v for k, v in case["appearance"].items() if k != "floor"})
                ),
            ),
            mock.patch.object(pipeline.Q, "score", return_value=dict(SCORES)),
            mock.patch.object(pipeline.App, "floor", return_value=None),
        ):
            pipeline.check(
                room, out2 := os.path.join(tmp, "out2"), log=lambda *a: None
            )  # a first check where no floor can be computed;
        with open(os.path.join(out2, "case.json")) as f:
            self.assertNotIn("floor", json.load(f)["appearance"])  # none: the fixed mapping
        from sbv.appearance.cache_identity import identity

        if identity(App.TORCH_PY) is None:
            self.skipTest("image model weights are not on this machine, so a floor is never reused")
        with (
            mock.patch.object(pipeline.calibrate, "build", return_value=copy.deepcopy(case)),
            mock.patch.object(pipeline.Q, "score", return_value=dict(SCORES)),
            mock.patch.object(pipeline.App, "floor", return_value=None) as again,
        ):
            pipeline.check(
                room, out, log=lambda *a: None
            )  # out's floor is cached (qualification.floor): the same inputs reuse it
        again.assert_not_called()
        with open(os.path.join(out, "case.json")) as f:
            self.assertEqual(json.load(f)["appearance"]["floor"], floor)

    def test_without_blender_the_floor_is_none_with_a_reason(self):
        tmp = tempfile.mkdtemp(prefix="sbv_floor_")
        rooms.write(os.path.join(tmp, "room"))
        said = []
        case = dict(
            _dir=tmp,
            reference="room/scene.xml",
            physics=dict(PHYSICS),
            room=dict(bounds=[[0, 0, 0], [4, 3, 2.5]]),
            appearance=dict(
                resolution=64,
                samples=1,
                cameras=[dict(name="view_0", position=[2, 1.5, 1.4], look_at=[2, 1.5, 0.8], hfov_deg=60)],
            ),
        )
        os.environ["SBV_REF_CACHE"] = os.path.join(tmp, "cache")
        try:
            with mock.patch.object(App, "render", side_effect=RuntimeError("render failed: no bpy")):
                self.assertIsNone(App.floor(case, said.append))
        finally:
            del os.environ["SBV_REF_CACHE"]
        self.assertTrue(said and said[0].startswith("appearance floor:"), said)

    def test_the_floor_does_not_change_the_reference_render_key(self):
        tmp = tempfile.mkdtemp(prefix="sbv_floor_")
        rooms.write(os.path.join(tmp, "room"))
        case = dict(
            _dir=tmp,
            reference="room/scene.xml",
            room=dict(bounds=[[0, 0, 0], [4, 3, 2.5]]),
            appearance=dict(resolution=512, cameras=[]),
        )
        k = App.ref_key(case)
        case["appearance"]["floor"] = dict(siglip2=0.9)
        self.assertEqual(App.ref_key(case), k)
        case["appearance"]["cameras"] = [dict(name="view_0")]
        self.assertNotEqual(App.ref_key(case), k)


if __name__ == "__main__":
    unittest.main()
