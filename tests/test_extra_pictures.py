"""Extra pictures in the tasks (sbv/appearance/extra.py): pictures/extra/ with its own cameras.json beside the main
pictures, one Inputs line that says so, L1 depth for each extra picture, and the audit's pasted-picture check over
them."""

import json, os, shutil, tempfile, unittest
from unittest import mock
from scenebench_harbor import make_tasks as M
from sbv.common import case as K
from tests import rooms
from tests.test_levels import CAM, case_of, pictures
from tests.harbor_fixtures import metric_environment, receipt

EXTRA = [dict(CAM, name="extra_0"), dict(CAM, name="extra_1", position=[0.5, -2.5, 1.6])]


def extra_pictures(case, out):  # stands in for Blender: the case's extra cameras, one blank picture each
    cams = case["appearance"].get("extra_cameras") or []
    if not cams:
        return []
    os.makedirs(out, exist_ok=True)
    for c in cams:
        open(os.path.join(out, c["name"] + ".png"), "w").write(c["name"])
    with open(os.path.join(out, "cameras.json"), "w") as f:
        json.dump(dict(resolution=512, cameras=cams), f)
    return [c["name"] for c in cams]


def tasks(extra, levels=("l0", "l1")):
    work = tempfile.mkdtemp()
    out = os.path.join(work, "tasks")
    res = os.path.join(work, "results", "room")
    os.makedirs(res)
    xml = rooms.write(os.path.join(work, "rooms", "room"))
    case = case_of(xml)
    case.update(reference="../../rooms/room/scene.xml", appearance=dict(extra_cameras=extra) if extra else {})
    K.save(case, os.path.join(res, "case.json"))
    receipt(work)
    with (
        metric_environment(),
        mock.patch.object(M.App, "input_pictures", pictures),
        mock.patch.object(M.App, "extra_pictures", extra_pictures),
    ):
        M.main(["--work", work, "--out", out, "--pictures", "1", "--levels", *levels])
    return out


class ExtraPictures(unittest.TestCase):
    def test_tasks_carry_the_extra_pictures_and_say_so(self):
        out = tasks(EXTRA)
        for side in ("public", "private"):
            for n in sorted(e.name for e in os.scandir(os.path.join(out, side)) if e.is_dir()):
                env = os.path.join(out, side, n, "environment")
                self.assertEqual(
                    sorted(os.listdir(os.path.join(env, "pictures", "extra"))),
                    ["cameras.json", "extra_0.png", "extra_1.png"],
                )
                self.assertEqual(
                    json.load(open(os.path.join(env, "pictures", "extra", "cameras.json")))["cameras"], EXTRA
                )
                text = open(os.path.join(out, side, n, "instruction.md")).read()
                self.assertIn(
                    "| `/task/pictures/extra/extra_0.png`, `/task/pictures/extra/extra_1.png` | png | 2 more pictures. Each is aimed at objects",
                    text,
                )
                self.assertIn("| `/task/pictures/extra/cameras.json` | json |", text)
                self.assertIn("main picture", text)
                self.assertLess(
                    text.index("`/task/pictures/input_0.png`"), text.index("`/task/pictures/extra/extra_0.png`")
                )
                card = json.load(
                    open(os.path.join(out, side, n, "task_card.json"))
                )  # the card's routine points at them too
                self.assertIn("2 extra pictures", card["agentMustDo"][0])
        l1 = [n for n in os.listdir(os.path.join(out, "public")) if n.endswith("-l1")][0]
        depth = os.path.join(out, "public", l1, "environment", "files", "depth")
        self.assertEqual(sorted(os.listdir(depth)), ["extra_0.json", "extra_1.json", "input_0.json"])
        pics = os.path.join(
            out, "private", l1, "tests", "pictures", "extra"
        )  # the audit's copy of what the entrant got
        self.assertTrue(os.path.exists(os.path.join(pics, "extra_1.png")))

    def test_a_room_without_extras_has_no_line_and_no_folder(self):
        out = tasks([], levels=("l0",))
        n = os.listdir(os.path.join(out, "public"))[0]
        self.assertFalse(os.path.exists(os.path.join(out, "public", n, "environment", "pictures", "extra")))
        self.assertNotIn("pictures/extra", open(os.path.join(out, "public", n, "instruction.md")).read())

    def test_new_extra_cameras_render_the_pictures_again(self):
        """input_key changes with the extra cameras, and stays as before for a case without them."""
        xml = rooms.write(tempfile.mkdtemp())
        base = dict(case_of(xml), appearance=dict(input_cameras=[CAM]), physics={})
        plain = M.App.input_key(base)
        self.assertEqual(plain, M.App.input_key(dict(base, appearance=dict(input_cameras=[CAM], extra_cameras=[]))))
        self.assertNotEqual(
            plain, M.App.input_key(dict(base, appearance=dict(input_cameras=[CAM], extra_cameras=EXTRA)))
        )

    def test_audit_finds_a_pasted_extra_picture(self):
        from sbv import audit as A

        d = tempfile.mkdtemp()
        xml = rooms.write(os.path.join(d, "copy"))
        pics = os.path.join(d, "pictures", "extra")
        os.makedirs(pics)
        open(os.path.join(pics, "extra_0.png"), "w").write("seen from the corner")
        shutil.copy(os.path.join(pics, "extra_0.png"), os.path.join(d, "copy", "wall_texture.png"))
        signs = []
        out = A.scene_layer(xml, None, os.path.join(d, "pictures"), signs)
        self.assertEqual(out["input_pictures"], {"wall_texture.png": os.path.join("extra", "extra_0.png")})
        self.assertIn("input picture in the copy: wall_texture.png = extra/extra_0.png", signs)


if __name__ == "__main__":
    unittest.main()
