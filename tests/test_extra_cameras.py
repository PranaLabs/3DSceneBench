"""Extra pictures (sbv/appearance/extra.py): aimed at what the main pictures miss, chosen after the held-out cameras and
apart from them, so adding them never moves a held-out camera nor the reference renders' cache."""

import copy, json, os, unittest
import numpy as np
from tests.test_input_cameras import scene, ROOM
from sbv.appearance import views as V, heldout as H, extra as E

HIDDEN = ROOM.replace(
    "</worldbody>",
    '<body name="bookcase__0" pos="2.2 2.0 1.0"><geom type="box" size="0.05 0.9 1.0"/></body>'
    '<body name="cup__0" pos="2.7 2.6 0.05"><freejoint/><geom type="box" size="0.04 0.04 0.05" mass="0.2"/></body></worldbody>',
)
B = [[-3.0, -3.0, -0.05], [3.0, 3.0, 3.0]]


def room():
    s = scene(HIDDEN)
    s.settle()
    inputs = V.cameras(s, B, height=1.6, kind="input")
    made = H.build(s, B, inputs)
    a = dict(resolution=512, input_cameras=inputs, cameras=made["cameras"], camera_groups=made["camera_groups"])
    return s, a


class ExtraCameras(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s, cls.a = room()

    def add(self, a=None):
        return E.add(a or self.a, self.s, B, self.s.xml)

    def test_extras_show_the_hidden_cup_and_stand_apart(self):
        """No main picture shows the cup behind the partition; an extra does. Every extra stands apart from every
        held-out camera by the held-out rule, views.SPACING from the other extras, where a camera may stand."""
        a = self.add()
        extras = a["extra_cameras"]
        self.assertTrue(extras)
        self.assertLessEqual(len(extras), E.MAX)
        self.assertEqual([c["name"] for c in extras], [f"extra_{k}" for k in range(len(extras))])
        self.assertIn("cup__0", a["extra_protocol"]["adds"])
        self.assertNotIn("cup__0", a["extra_protocol"]["unseen"])
        center = np.mean(np.asarray(B, float), axis=0)[:2]
        for k, c in enumerate(extras):
            if k:
                self.assertGreaterEqual(
                    np.linalg.norm(
                        np.subtract([x["position"][:2] for x in extras[:k]], c["position"][:2]), axis=1
                    ).min(),
                    V.SPACING - 1e-3,
                )
            self.assertFalse(H.separated(c["position"], a["cameras"], center), c["name"])
            self.assertIsNone(H.placement(self.s, B, c["position"], c["look_at"]), c["name"])
            self.assertIsNone(H.record_problem(c, 512), c["name"])

    def test_main_pictures_are_the_first_eight(self):
        """The main set is input_0-7 even when the case has twelve input cameras; a case with fewer cannot have
        extras."""
        self.assertEqual([c["name"] for c in E.main_cameras(self.a)], [f"input_{k}" for k in range(8)])
        with self.assertRaisesRegex(ValueError, "the case has 4"):
            E.main_cameras(dict(self.a, input_cameras=self.a["input_cameras"][:4]))

    def test_held_out_cameras_and_reference_renders_stay(self):
        """Adding extras keeps every input and held-out camera, and the reference-render cache key."""
        a = self.add()
        case = dict(appearance=self.a, room=dict(bounds=B), physics={}, reference=self.s.xml)
        for k in ("input_cameras", "cameras", "camera_groups"):
            self.assertEqual(a[k], self.a[k], k)
        from unittest import mock

        with (
            mock.patch("sbv.common.dependencies.model_files", lambda p: {}),
            mock.patch("sbv.common.dependencies.render_identity", lambda: "r"),
        ):
            self.assertEqual(V.ref_key(case), V.ref_key(dict(case, appearance=a)))

    def test_the_rooms_own_extras(self):
        """extra_cameras.json is used as given when valid; one too near a held-out camera is refused with its name."""
        made = self.add()["extra_cameras"]
        p = os.path.join(os.path.dirname(self.s.xml), E.FILE)
        try:
            with open(p, "w") as f:
                json.dump(dict(cameras=made), f)
            a = self.add()
            self.assertEqual(a["extra_cameras"], made)
            self.assertEqual(a["extra_protocol"]["protocol"], E.SUPPLIED)
            h = self.a["cameras"][0]
            bad = V.camera("extra_0", np.array(h["position"]) + [0.1, 0, 0], np.array(h["look_at"]), 60.0, 512)
            with open(p, "w") as f:
                json.dump(dict(cameras=[bad]), f)
            with self.assertRaisesRegex(ValueError, f"extra_0: .*held-out {h['name']}"):
                self.add()
            with open(p, "w") as f:
                json.dump(dict(cameras=[dict(bad, name="extra_3")]), f)
            with self.assertRaisesRegex(ValueError, "named extra_0"):
                self.add()
        finally:
            os.remove(p)

    def test_visible_scope_gains_what_the_extras_show(self):
        a = self.add()
        case = dict(appearance=a, objects=[dict(slot="table__0"), dict(slot="cup__0"), dict(slot="bookcase__0")])
        self.assertIn("cup__0", E.visible(case, ["table__0"]))
        self.assertEqual(E.visible(dict(case, appearance=self.a), ["table__0"]), ["table__0"])

    def test_new_held_out_cameras_drop_the_extras(self):
        """Extras were chosen apart from the former held-out cameras: a held-out rebuild drops them."""
        case = dict(appearance=self.add(), room=dict(bounds=B))
        new = H.update_case(case, self.s)["appearance"]
        self.assertNotIn("extra_cameras", new)
        self.assertNotIn("extra_protocol", new)


if __name__ == "__main__":
    unittest.main()
