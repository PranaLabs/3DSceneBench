"""Hidden appearance camera validity and compatibility with frozen physics cases."""

import copy
import math
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import tempfile
import json
from pathlib import Path
import numpy as np
from tests.test_input_cameras import scene, ROOM
from sbv.appearance.views import camera, evaluate
from sbv.appearance.heldout import build, clearance, separated, update_case


class HeldoutGroups(unittest.TestCase):
    def setUp(self):
        self.room = scene(
            ROOM.replace("</worldbody>", '<geom name="ceiling" type="box" pos="0 0 2.9" size="3 3 .1"/></worldbody>')
        )
        self.bounds = [[-3, -3, -0.05], [3, 3, 3]]
        self.inputs = [
            camera(
                f"input_{i}",
                np.array([2.4 * math.sin(i * math.pi / 4), 2.4 * math.cos(i * math.pi / 4), 1.65]),
                np.array([0.0, 0.0, 0.8]),
                85,
                512,
            )
            for i in range(8)
        ]

    def test_counts_heights_shift_and_serialized_clearance(self):
        made = build(self.room, self.bounds, self.inputs)
        self.assertEqual([len(x) for x in made["camera_groups"].values()], [8, 8])
        self.assertEqual(len({x["name"] for x in made["cameras"]}), 16)
        for i, c in enumerate(made["cameras"]):
            self.assertFalse(clearance(self.room, c["position"]))
            src = self.inputs[i % 8]
            a = np.arctan2(src["position"][0], src["position"][1])
            b = np.arctan2(c["position"][0], c["position"][1])
            self.assertAlmostEqual((b - a) % (2 * np.pi), np.pi / 8, places=3)
            self.assertAlmostEqual(c["position"][2], 1.65 if i < 8 else 2.2 if i % 2 == 0 else 1.1)
            self.assertFalse(separated(c["position"], self.inputs, np.zeros(2)))

    def test_every_camera_stands_apart_from_every_input(self):
        # An extra input camera (as cover() adds input_8-11) right where view_same_0 and view_high_0 would stand.
        plain = build(self.room, self.bounds, self.inputs)
        spot = np.array(plain["cameras"][0]["position"])
        extra = camera("input_8", np.r_[spot[:2], 1.65], np.array([0.0, 0.0, 0.8]), 85, 512)
        inputs = self.inputs + [extra]
        made = build(self.room, self.bounds, inputs)
        moved = {c["name"] for c in made["cameras"] if c not in plain["cameras"]}
        self.assertEqual(moved, {"view_same_0", "view_high_0"})  # the other fourteen keep their spots
        for c in made["cameras"]:
            self.assertFalse(separated(c["position"], inputs, np.zeros(2)), c["name"])
            self.assertFalse(clearance(self.room, c["position"]))
        for e in made["validation"]:
            self.assertEqual(e["separation"], {"m": 0.5, "deg": 15.0})
        self.assertEqual(made["protocol"], "shifted-eight-and-height-eight-v3")

    def test_inside_ceiling_is_not_valid_even_with_long_exit_ray(self):
        self.assertIn("ceiling", clearance(self.room, [0, 0, 2.9]))
        self.assertTrue(clearance(self.room, [0.5, 0.5, 0.4]))

    def test_preserves_inputs_and_physics_and_drops_stale_floor(self):
        original = {
            "room": {"bounds": self.bounds},
            "appearance": {"input_cameras": self.inputs, "floor": {"lpips_vgg": 0.5}},
            "physics": {"timestep": 0.002},
            "functionality": {"units": [{"id": "frozen"}]},
        }
        snapshot = copy.deepcopy(original)
        new = update_case(original, self.room)
        self.assertEqual(original, snapshot)
        self.assertEqual(new["functionality"], original["functionality"])
        self.assertEqual(new["physics"], original["physics"])
        self.assertEqual(new["appearance"]["input_cameras"], self.inputs)
        self.assertNotIn("floor", new["appearance"])

    def test_evaluate_reports_both_groups_without_input_contamination(self):
        from sbv.appearance import views as V

        made = build(self.room, self.bounds, self.inputs)
        case = {"physics": {}, "appearance": {"camera_groups": made["camera_groups"], "heldout_protocol": made}}
        names = [c["name"] for c in made["cameras"]] + ["input_0"]
        raw = dict(views=names, per_view={"lpips_vgg": [0.2] * 8 + [0.6] * 8 + [0.99]}, unavailable={})
        fake = SimpleNamespace(q0=None, reset=lambda q: None)
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "metrics.json").write_text(json.dumps(raw))
            with (
                patch.object(V, "light_for", return_value={"z": 2.7}),
                patch.object(V, "reference_renders", return_value=({}, folder)),
                patch.object(V, "render", return_value={}),
                patch.object(V, "TORCH_PY", __file__),
                patch.object(V.subprocess, "run", return_value=SimpleNamespace(returncode=0)),
            ):
                result = evaluate(case, "unused", folder, condition="8", scene=fake)
        self.assertEqual(result["groups"]["same_height"]["mean"]["lpips_vgg"], 0.2)
        self.assertEqual(result["groups"]["high_low"]["mean"]["lpips_vgg"], 0.6)
        self.assertEqual(result["input"]["mean"]["lpips_vgg"], 0.99)
        self.assertEqual(result["mean"]["lpips_vgg"], 0.4)

    def test_switching_gpu_backend_invalidates_render_caches(self):
        import os
        from sbv.appearance import views as V

        c = {"room": {"bounds": self.bounds}, "appearance": {"cameras": [], "input_cameras": []}}
        with (
            patch("sbv.common.case.reference_path", return_value=self.room.xml),
            patch("sbv.common.case.fingerprint", return_value="roomhash"),
        ):
            with patch.dict(os.environ, SBV_RENDER_DEVICE="GPU", SBV_CYCLES_BACKEND="CUDA"):
                a = (V.input_key(c), V.ref_key(c))
            with patch.dict(os.environ, SBV_RENDER_DEVICE="GPU", SBV_CYCLES_BACKEND="OPTIX"):
                b = (V.input_key(c), V.ref_key(c))
        self.assertNotEqual(a[0], b[0])
        self.assertNotEqual(a[1], b[1])

    def test_no_invalid_fallback_or_fewer_views(self):
        with self.assertRaises(ValueError):
            build(self.room, self.bounds, self.inputs[:7])
        with self.assertRaises(ValueError):
            build(self.room, self.bounds, self.inputs, height_delta=10)

    def test_a_blocked_ray_turns_to_the_nearest_clear_spot(self):
        # A shelf along input_0's shifted ray (67.5 degrees about the centre) from 0.55 m to 3.15 m out: nothing on it
        # is clear.
        shelf = (
            '<body name="shelf__0" pos="0.708 1.709 0.8"><geom type="box" size="1.3 0.1 0.8" euler="0 0 67.5"/></body>'
        )
        room = scene(
            ROOM.replace(
                "</worldbody>", '<geom name="ceiling" type="box" pos="0 0 2.9" size="3 3 .1"/>' + shelf + "</worldbody>"
            )
        )
        plain, made = build(self.room, self.bounds, self.inputs), build(room, self.bounds, self.inputs)
        turned = [c for c in made["cameras"] if c["azimuth_shift_deg"] != 22.5]
        self.assertEqual(sorted(c["name"] for c in turned), ["view_high_0", "view_same_0"])
        self.assertEqual(
            [c for c in made["cameras"] if c not in turned],
            [c for c in plain["cameras"] if c["name"] not in ("view_same_0", "view_high_0")],
        )
        center = np.zeros(2)
        for c in turned:
            ev = next(e for e in made["validation"] if e["camera"] == c["name"])
            self.assertEqual(c["azimuth_shift_deg"], 22.5 - ev["turned_deg"])
            self.assertEqual(ev["turned_deg"] % 2.5, 0)
            self.assertFalse(clearance(room, c["position"]))
            self.assertFalse(separated(c["position"], self.inputs, center))
            mates = [m for m in made["cameras"] if m["name"] != c["name"] and m["name"][5:9] == c["name"][5:9]]
            self.assertFalse(separated(c["position"], mates, center))
        self.assertNotIn("turned_deg", next(e for e in made["validation"] if e["camera"] == "view_same_1"))

    def test_a_room_with_no_clear_spot_says_so(self):
        crate = '<body name="crate__0" pos="0 0 .8"><geom type="box" size="2.9 2.9 .8"/></body>'  # fills the floor
        room = scene(
            ROOM.replace(
                "</worldbody>", '<geom name="ceiling" type="box" pos="0 0 2.9" size="3 3 .1"/>' + crate + "</worldbody>"
            )
        )
        with self.assertRaisesRegex(
            ValueError, "view_same_0.*too small or too full.*input_cameras.json and heldout_cameras.json"
        ):
            build(room, self.bounds, self.inputs)


if __name__ == "__main__":
    unittest.main()
