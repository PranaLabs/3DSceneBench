"""A room's own input and held-out cameras (input_cameras.json, heldout_cameras.json): checked on the reference room,
used by calibration in place of the shifted rule, never published, and refused when they break a rule."""

import copy, json, math, os, tempfile, unittest
import numpy as np
from sbv.appearance import heldout as H
from sbv.appearance.views import camera
from tests.test_input_cameras import scene, ROOM


def docs(room, bounds, inputs):
    """The two files as the harness writes them: inputs in camera record format, held-out in two groups."""
    made = H.build(room, bounds, inputs)
    held = [
        dict(c, group="same_height" if c["height_group"] == "same_height" else "high_low", rule=dict(step="default"))
        for c in made["cameras"]
    ]
    return dict(protocol="test", cameras=copy.deepcopy(inputs)), dict(
        protocol="test", cameras=held, camera_groups=made["camera_groups"]
    )


def searched(room, bounds, inputs):
    """Held-out cameras by a plain grid search under the same rules (placeable, separated, distinct), as a room might
    choose its own."""
    from sbv.geometry.g3_layout import floor_plan

    xs, ys, free = plan = floor_plan(room, bounds, H.FLOOR_RADIUS_M)
    center = np.mean(np.asarray(bounds, float), axis=0)[:2]
    spots = [(x, y) for i, x in enumerate(xs) for j, y in enumerate(ys) if free[i, j]][::3]
    cams, groups = [], {"same_height": [], "high_low": []}
    for g in groups:
        for i, src in enumerate(inputs[:8]):
            z0 = src["position"][2]
            zs = [z0] if g == "same_height" else [z0 + 0.5, z0 - 0.5]
            name = f"view_{'same' if g == 'same_height' else 'hl'}_{i}"
            pos = next(
                np.array([x, y, z])
                for z in zs
                for x, y in spots
                if not H.placement(room, bounds, [x, y, z], src["look_at"], plan)
                and not H.separated([x, y, z], inputs, center)
                and all(np.linalg.norm(np.subtract(c["position"], [x, y, z])) > 0.1 for c in cams)
            )
            cams.append(dict(camera(name, pos, np.array(src["look_at"]), 60, 512), source_input=src["name"], group=g))
            groups[g].append(name)
    return dict(protocol="test", cameras=cams, camera_groups=groups)


class SuppliedCameras(unittest.TestCase):
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
                60,
                512,
            )
            for i in range(8)
        ]
        self.own = docs(self.room, self.bounds, self.inputs)

    def test_valid_cameras_are_used_as_given(self):
        a, b = copy.deepcopy(self.own)
        a["cameras"][3]["selection_note"] = "kept private"
        made = H.validate(self.room, self.bounds, a, b)
        self.assertEqual(made["protocol"], H.SUPPLIED)
        self.assertEqual([c["name"] for c in made["input_cameras"]], [f"input_{k}" for k in range(8)])
        self.assertNotIn("selection_note", made["input_cameras"][3])  # only the camera record reaches cameras.json
        self.assertEqual({g: len(v) for g, v in made["camera_groups"].items()}, {"same_height": 8, "high_low": 8})
        case = H.update_case(
            dict(room=dict(bounds=self.bounds), appearance=dict(resolution=512, floor={"lpips_vgg": 0.5})),
            self.room,
            self.own,
        )
        self.assertEqual(case["appearance"]["heldout_protocol"]["protocol"], H.SUPPLIED)
        self.assertEqual(case["appearance"]["input_cameras"], made["input_cameras"])
        self.assertNotIn("floor", case["appearance"])

    def test_a_heldout_camera_beside_an_input_is_refused(self):
        a, b = copy.deepcopy(self.own)
        src = self.inputs[2]["position"]
        c = b["cameras"][0]
        b["cameras"][0] = dict(
            camera(c["name"], np.array([src[0] + 0.1, src[1], src[2]]), np.array(c["look_at"]), 60, 512),
            source_input=c["source_input"],
            group=c["group"],
        )
        with self.assertRaisesRegex(ValueError, "under 0.5 m and 15.0 degrees from input_2"):
            H.validate(self.room, self.bounds, a, b)

    def test_broken_rules_are_named(self):
        a, b = copy.deepcopy(self.own)
        a["cameras"] = a["cameras"][:7]
        with self.assertRaisesRegex(ValueError, "8 or 12 cameras"):
            H.validate(self.room, self.bounds, a, b)
        a, b = copy.deepcopy(self.own)
        b["cameras"][1]["source_input"] = b["cameras"][0]["source_input"]
        with self.assertRaisesRegex(ValueError, "linking input_0-7 once each"):
            H.validate(self.room, self.bounds, a, b)
        a, b = copy.deepcopy(self.own)
        a["cameras"][0] = camera("input_0", np.array([0.5, 0.5, 0.4]), np.array([0.0, 0.0, 0.8]) + 2, 60, 512)
        with self.assertRaisesRegex(ValueError, "input_0: "):
            H.validate(self.room, self.bounds, a, b)  # inside the table
        a, b = copy.deepcopy(self.own)
        a["cameras"][1]["position"] = [1.0, 1.0, 1.65]  # record no longer matches its pose
        with self.assertRaisesRegex(ValueError, "input_1: cam_to_world disagrees"):
            H.validate(self.room, self.bounds, a, b)

    def test_both_files_or_neither(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(H.supplied(d))
            with open(os.path.join(d, H.INPUTS_FILE), "w") as f:
                json.dump(self.own[0], f)
            with self.assertRaisesRegex(ValueError, "come together"):
                H.supplied(d)
            with open(os.path.join(d, H.HELDOUT_FILE), "w") as f:
                json.dump(self.own[1], f)
            self.assertEqual(H.supplied(os.path.join(d, "scene.xml"))[1]["camera_groups"], self.own[1]["camera_groups"])


class SuppliedPipeline(unittest.TestCase):
    def test_calibration_uses_the_rooms_own_cameras(self):
        from sbv import calibrate
        from sbv.appearance import views as App
        from sbv.common.scene import Scene, PHYSICS

        with tempfile.TemporaryDirectory() as tmp:
            from tests import rooms

            room = os.path.join(tmp, "room")
            xml = rooms.write(room)
            s = Scene(xml, PHYSICS, probes=True)
            s.reset(s.q0)
            bounds = calibrate.room_bounds(s)
            s.settle()
            inputs = App.cameras(s, bounds, height=1.6, res=512, kind="input")[:8]
            shifted = H.build(s, bounds, inputs)  # the shifted rule turns two cameras here; the room's own cameras win
            own = (dict(protocol="test", cameras=inputs), searched(s, bounds, inputs))
            for name, doc in zip((H.INPUTS_FILE, H.HELDOUT_FILE), own):
                with open(os.path.join(room, name), "w") as f:
                    json.dump(doc, f)
            case = calibrate.build(
                room,
                procs=2,
                draws=3,
                need=2,
                jitter=1,
                robot_tasks=0,
                push_tasks=0,
                require_tasks=False,
                checkpoint=os.path.join(tmp, "cache"),
                log=lambda *a: None,
            )
            a = case["appearance"]
            self.assertEqual(a["heldout_protocol"]["protocol"], H.SUPPLIED)
            self.assertNotIn("camera_seed", a)
            self.assertEqual(a["input_cameras"], own[0]["cameras"])
            self.assertEqual([c["name"] for c in a["cameras"]], [c["name"] for c in own[1]["cameras"]])
            self.assertNotEqual([c["position"] for c in a["cameras"]], [c["position"] for c in shifted["cameras"]])

    def test_public_tasks_refuse_private_camera_files(self):
        from scenebench_harbor import make_tasks as M

        with tempfile.TemporaryDirectory() as tmp:
            pics = os.path.join(tmp, "pics")
            os.makedirs(pics)
            with open(os.path.join(pics, "input_0.png"), "wb") as f:
                f.write(b"\x89PNG")
            with open(os.path.join(pics, "cameras.json"), "w") as f:
                json.dump(dict(cameras=[]), f)
            M.write_task(os.path.join(tmp, "ok"), "pranalabs", "t", "room", "1", pics, ["cup"])
            with open(os.path.join(pics, H.HELDOUT_FILE), "w") as f:
                f.write("{}")
            with self.assertRaisesRegex(SystemExit, "private camera files"):
                M.write_task(os.path.join(tmp, "bad"), "pranalabs", "t", "room", "1", pics, ["cup"])

    def test_add_input_cameras_needs_twelve_in_the_rooms_file(self):
        from sbv import add_input_cameras as AI
        from sbv.common.scene import PHYSICS

        with tempfile.TemporaryDirectory() as room:
            with open(os.path.join(room, "scene.xml"), "w") as f:
                f.write(ROOM)
            s = scene(ROOM)
            b = [[-3, -3, -0.05], [3, 3, 2.9]]
            ins = [
                camera(
                    f"input_{i}",
                    np.array([2.4 * math.sin(i * math.pi / 4), 2.4 * math.cos(i * math.pi / 4), 1.65]),
                    np.array([0.0, 0.0, 0.8]),
                    60,
                    512,
                )
                for i in range(8)
            ]
            for name, doc in zip((H.INPUTS_FILE, H.HELDOUT_FILE), docs(s, b, ins)):
                with open(os.path.join(room, name), "w") as f:
                    json.dump(doc, f)
            case = dict(physics=dict(PHYSICS), room=dict(bounds=b), appearance=dict(resolution=512, input_cameras=ins))
            with self.assertRaisesRegex(SystemExit, "has 8 cameras; choose twelve"):
                AI.add(case, room)


if __name__ == "__main__":
    unittest.main()
