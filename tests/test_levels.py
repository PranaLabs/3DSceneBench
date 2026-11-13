"""Instruction levels (scenebench_harbor/levels.py): L1 names only what a kept unit tests, never an instance id; L2's
files match the reference room and give no position; a declared function the case did not keep blocks levelled tasks."""

import contextlib, io, json, os, re, tempfile, unittest
from unittest import mock
from scenebench_harbor import levels as L, make_tasks as M
from sbv.common import case as K
from tests import rooms
from tests.harbor_fixtures import metric_environment, receipt

CAM = dict(
    name="input_0",
    position=[0.0, -2.5, 1.6],
    intrinsics=[[443.405, 0, 256.0], [0, 443.405, 256.0], [0, 0, 1]],
    cam_to_world=[[1, 0, 0, 0.0], [0, 0.6, -0.8, -2.5], [0, 0.8, 0.6, 1.6], [0, 0, 0, 1]],
)


def pictures(case, out, condition):  # stands in for Blender: one picture and its camera
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "input_0.png"), "w").close()
    with open(os.path.join(out, "cameras.json"), "w") as f:
        json.dump(dict(resolution=512, cameras=[CAM]), f)


def case_of(xml):
    objs = [
        dict(slot=n, category=n.split("__")[0])
        for n in ("table__0", "cup__0", "book__0", "chair__0", "crate__0", "vase__0", "cabinet__0")
    ]
    units = [
        dict(
            id="joint|cabinet__0|door_hinge",
            kind="joint",
            slots=["cabinet__0"],
            named=True,
            params=dict(ref=dict(name="door_hinge")),
        ),
        dict(id="opening|crate__0|0", kind="opening", slots=["crate__0"], named=True),
    ]
    return dict(
        schema=K.SCHEMA,
        reference=xml,
        vocabulary=sorted({o["category"] for o in objs}),
        objects=objs,
        functionality=dict(units=units),
    )


class Levels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.room = tempfile.mkdtemp(prefix="sbv_levels_")
        cls.xml = rooms.write(cls.room)
        cls.case = case_of(cls.xml)
        cls.objs = L.objects(cls.case, cls.xml)

    def test_l1_names_only_kept_units_in_words(self):
        text = L.l1_text(self.case, self.objs)
        self.assertIn("cabinet doors (hinged)", text)
        self.assertIn("into a crate", text)
        self.assertNotIn("drawer", text)  # the drawer has no kept unit: not named
        self.assertNotIn("cabinets", text)
        self.assertNotIn("the cabinet's", text)  # no count, not even one or many
        self.assertIsNone(re.search(r"[a-z]__\d", text))
        self.assertIsNone(re.search(r"\d", text))  # no ids, no numbers

    def test_part_words_are_whole_words_and_an_empty_level_says_so(self):
        self.assertEqual(
            L.part_word("lamp__0_hinge_bottom_with_sliding_0", "hinge"), "hinged part"
        )  # "sliding" holds no lid
        self.assertEqual(L.part_word("cabinet__0_cabinet_hinge_0", "hinge"), "hinged part")
        self.assertEqual(L.part_word("box__0_F1_top_lid_axis_0", "hinge"), "lid")
        self.assertEqual(L.part_word("drawer__0_drawer_slider_0", "slide"), "drawer")
        self.assertEqual(
            L.part_word("bed__0_bed__0_drawer0_wheel1_axle", "hinge"), "wheel"
        )  # the last part word: a wheel under a drawer
        self.assertEqual(L.part_word("door__0_door__0_latch_slide", "slide"), "latch")
        self.assertEqual(
            L.part_entry(dict(name="toy__3_wheel0_axle", type="hinge", travel=None)),
            dict(part="wheel", type="hinge", no_stops=True),
        )
        self.assertEqual(
            L.part_entry(dict(name="desk__0_drawer0_slide", type="slide", travel=0.21)),
            dict(part="drawer", type="slide", travel=0.21, unit="m"),
        )
        bare = dict(self.case, functionality=dict(units=[]))
        self.assertEqual(L.l1_text(bare, self.objs), "")
        self.assertEqual(L.level("l1", bare, self.xml, [])[0], L.NONE)
        self.assertEqual(L.level("l0", bare, self.xml, []), ("", {}))

    def test_l2_files_match_the_room_and_give_no_position(self):
        files = L.l2_files(self.objs, self.xml, [CAM])
        self.assertEqual(sum(v["count"] for v in files["inventory.json"].values()), len(self.case["objects"]))
        door = [p for p in files["measurements.json"]["cabinet__0"]["parts"] if p["type"] == "hinge"]
        self.assertTrue(door and door[0]["travel"] > 0 and door[0]["unit"] == "rad")
        grid = files["depth/input_0.json"]["depth"]
        self.assertEqual((len(grid), len(grid[0])), (16, 16))
        self.assertTrue(any(x for row in grid for x in row))
        self.assertNotIn("pos", json.dumps(files["measurements.json"]))
        self.assertEqual(
            files["measurements.json"]["cup__0"]["size_factor"], 1.25
        )  # the bar G2.3 scores sizes by, not a made-up tolerance

    def test_declared_function_not_kept_is_listed(self):
        d = tempfile.mkdtemp()
        rooms.write(d)
        with open(os.path.join(d, "functions.json"), "w") as f:
            json.dump(dict(containers=dict(vase__0=dict(openings=[{}]), crate__0=dict(openings=[{}]))), f)
        with open(os.path.join(d, "tasks.json"), "w") as f:
            json.dump(dict(push=[dict(object="cup__0")]), f)
        self.assertEqual(L.audit(d, self.case), ["opening of vase__0", "push cup__0"])

    def test_make_tasks_writes_one_task_per_level(self):
        work = tempfile.mkdtemp()
        out = os.path.join(work, "tasks")
        res = os.path.join(work, "results", "room")
        os.makedirs(res)
        rooms.write(os.path.join(work, "rooms", "room"))
        with open(
            os.path.join(work, "rooms", "room", "functions.json"), "w"
        ) as f:  # the vase's opening was dropped at calibration
            json.dump(dict(containers=dict(crate__0=dict(openings=[{}]), vase__0=dict(openings=[{}]))), f)
        K.save(dict(self.case, reference="../../rooms/room/scene.xml"), os.path.join(res, "case.json"))
        receipt(work)
        err = io.StringIO()
        with (
            metric_environment(),
            mock.patch.object(M.App, "input_pictures", pictures),
            contextlib.redirect_stderr(err),
        ):
            M.main(["--work", work, "--out", out, "--pictures", "1", "--levels", "l0", "l1"])
        self.assertIn("warning: room room: declared functions the reference room fails", err.getvalue())
        self.assertIn("opening of vase__0", err.getvalue())
        pub = sorted(e.name for e in os.scandir(os.path.join(out, "public")))
        self.assertEqual([n[-3:] for n in pub], ["-l0", "-l1"])  # two levels
        texts = {n[-2:]: open(os.path.join(out, "public", n, "instruction.md")).read() for n in pub}
        for f in ("inventory.json", "measurements.json", "depth/"):  # each level lists only the files it gives
            self.assertIn(f"| `/task/{f}` | json |", texts["l1"])
            self.assertNotIn(f"`/task/{f}`", texts["l0"])
        self.assertNotIn("This task also gives `inventory.json`", texts["l0"])
        self.assertIn("This task also gives `inventory.json`", texts["l1"])
        for t in texts.values():  # the card's order (Agents' Last Exam)
            heads = [h for h in re.findall(r"^## (.+)$", t, re.M)]
            self.assertEqual(heads, ["Task prompt", "Input files", "Output files", "Evaluation"])
        card = json.load(open(os.path.join(out, "public", pub[1], "task_card.json")))
        self.assertEqual(card["taskPrompt"], texts["l1"])
        self.assertEqual([r["path"] for r in card["referenceFiles"]][0], "/output/scene/scene.xml")
        self.assertIn("/task/inventory.json", [r["path"] for r in card["inputFiles"]])
        self.assertTrue(2 <= len(card["agentMustDo"]) <= 5)
        self.assertFalse(
            os.path.exists(os.path.join(out, "public", pub[1], "environment", "task_card.json"))
        )  # never in the container
        self.assertIn("What the verifiers do in this room", texts["l1"])
        self.assertNotIn("What the verifiers do in this room", texts["l0"])
        for t in texts.values():
            self.assertEqual(set(re.findall(r"\b[a-z_]+__\d+\b", t)) - {"floor__0"}, set())
        env = os.path.join(out, "private", pub[1], "environment")
        self.assertTrue(os.path.exists(os.path.join(env, "files", "depth", "input_0.json")))
        self.assertIn("COPY files /task/", open(os.path.join(env, "Dockerfile")).read())


if __name__ == "__main__":
    unittest.main()
