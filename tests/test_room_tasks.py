"""Each room's own robot tasks are announced in its instruction, in plain words: the tasks named in tasks.json that
calibration kept, in the builder's words or made from the objects' categories, never by object id."""

import os, tempfile, unittest
from scenebench_harbor.make_tasks import room_tasks, CASE_NONE
from sbv.common.scene import Scene, PHYSICS
from sbv.calibrate import room_bounds, task_kinds
from sbv.functionality import f2_calibrate as C
from sbv.functionality.f4_robot import tasks as T
from tests import rooms
from tests.harbor_fixtures import metric_environment, receipt

OBJECTS = [
    dict(slot=n, category=n.split("__")[0])
    for n in ("mug__0", "cabinet__0", "tissue_box__0", "book__19", "bookcase__0")
]


def unit(kind, slots, named=True, **kw):
    return dict(id="|".join([kind] + slots), kind=kind, slots=slots, named=named, **kw)


class RoomTasks(unittest.TestCase):
    def test_kept_named_tasks_in_words_without_ids(self):
        case = dict(
            objects=OBJECTS,
            functionality=dict(
                units=[
                    unit("robot", ["mug__0", "cabinet__0"]),
                    unit("robot_push", ["tissue_box__0"]),
                    unit(
                        "robot_shelve", ["book__19", "bookcase__0"], words="Stand the red book upright in the top bay."
                    ),
                    unit("robot", ["book__19", "cabinet__0"], named=False),
                    unit("push", ["mug__0"], named=False),
                ]
            ),
        )
        text = room_tasks(case)
        lines = [x for x in text.splitlines() if x.startswith("- ")]
        self.assertEqual(
            lines,
            [
                "- Pick up the mug and put it on top of the cabinet.",
                "- Push the tissue box 20 cm across the top it stands on.",
                "- Stand the red book upright in the top bay.",
            ],
        )  # the tasks it found itself stay hidden
        self.assertNotIn("__", text)

    def test_an_activity_step_counts_as_a_task_of_its_kind(self):  # no standalone pick is needed
        act = unit(
            "robot_activity",
            ["mug__0", "cabinet__0", "tissue_box__0"],
            params=dict(
                steps=[
                    dict(kind="pick_place", object="mug__0", goal="cabinet__0"),
                    dict(kind="push", object="tissue_box__0"),
                ]
            ),
        )
        have = task_kinds([act, unit("push", ["book__19"], named=False)])
        self.assertEqual(
            (have["robot"], have["robot_push"], have["robot_shelve"], have["robot_activity"]), (1, 1, 0, 1)
        )
        self.assertEqual(task_kinds([unit("robot_push", ["tissue_box__0"], params={})])["robot"], 0)

    def test_rewording_replaces_a_sentence_and_refuses_a_stale_one(self):
        case = dict(
            objects=OBJECTS,
            functionality=dict(units=[unit("robot_push", ["mug__0"], words="Push the mug to the left.")]),
        )
        text = room_tasks(case, reword={"Push the mug to the left.": "Push the mug 10 cm to the left."})
        self.assertIn("- Push the mug 10 cm to the left.", text)
        self.assertNotIn("- Push the mug to the left.", text)
        self.assertEqual(room_tasks(case, reword={}), room_tasks(case))
        with self.assertRaises(ValueError):
            room_tasks(case, reword={"Lift the mug.": "Lift the mug 10 cm."})

    def test_no_named_task_and_ids_in_words(self):
        self.assertEqual(
            room_tasks(
                dict(objects=OBJECTS, functionality=dict(units=[unit("robot", ["mug__0", "cabinet__0"], named=False)]))
            ),
            CASE_NONE,
        )
        with self.assertRaises(ValueError):
            room_tasks(
                dict(
                    objects=OBJECTS,
                    functionality=dict(units=[unit("robot_push", ["mug__0"], words="Push mug__0 to the left.")]),
                )
            )

    def test_declared_functions_are_not_announced_as_robot_tasks(self):
        # functions.json openings and holds are named units too; without words of their own they are not robot tasks
        # (one room had 23 lines "A robot task with the bag."); a unit that brings its own words (a room plugin) is kept
        case = dict(
            objects=OBJECTS,
            functionality=dict(
                units=[
                    unit("opening", ["mug__0"]),
                    unit("holds", ["cabinet__0", "mug__0"]),
                    unit("joint", ["cabinet__0"]),
                    unit("plugin", ["cabinet__0"], words="Open the cabinet and let go."),
                    unit("robot_push", ["tissue_box__0"]),
                ]
            ),
        )
        lines = [x for x in room_tasks(case).splitlines() if x.startswith("- ")]
        self.assertEqual(
            lines, ["- Open the cabinet and let go.", "- Push the tissue box 20 cm across the top it stands on."]
        )
        self.assertEqual(
            room_tasks(dict(objects=OBJECTS, functionality=dict(units=[unit("opening", ["mug__0"])]))), CASE_NONE
        )

    def test_every_task_of_the_room_lists_them(self):
        from unittest import mock
        from scenebench_harbor import make_tasks as M
        from sbv.common import case as K
        from tests.test_room_ids import pictures

        work = tempfile.mkdtemp()
        out = os.path.join(work, "tasks")
        res = os.path.join(work, "results", "room")
        os.makedirs(res)
        rooms.write(os.path.join(work, "rooms", "room"))
        named = [
            dict(o, slot=o["slot"].replace("mug", "cup"), category=o["category"].replace("mug", "cup")) for o in OBJECTS
        ]  # a reference room uses main words
        K.save(
            dict(
                schema=K.SCHEMA,
                reference="../../rooms/room/scene.xml",
                vocabulary=["cup", "table"],
                objects=named,
                functionality=dict(units=[unit("robot", ["cup__0", "cabinet__0"])]),
            ),
            os.path.join(res, "case.json"),
        )
        receipt(work)
        with metric_environment(), mock.patch.object(M.App, "input_pictures", pictures):
            M.main(["--work", work, "--out", out, "--pictures", "1"])
        for side in ("public", "private"):
            task = next(e.path for e in os.scandir(os.path.join(out, side)) if e.is_dir())
            with open(os.path.join(task, "instruction.md")) as f:  # the reference room's cup, by its main word
                self.assertIn("| 1 | Pick up the cup and put it on top of the cabinet. |", f.read())
            self.assertEqual(
                os.path.exists(os.path.join(task, "PRIVATE")), side == "private"
            )  # the marker a publish step refuses

    def test_about_md_says_what_kind_of_room_it_is(self):
        # bout.md heads the task list, so the tasks make sense; without it the list stands alone, with a warning
        import contextlib, io
        from unittest import mock
        from scenebench_harbor import make_tasks as M
        from sbv.common import case as K
        from tests.test_room_ids import pictures

        def run(about):
            work = tempfile.mkdtemp()
            out = os.path.join(work, "tasks")
            res = os.path.join(work, "results", "room")
            os.makedirs(res)
            rooms.write(os.path.join(work, "rooms", "room"))
            if about is not None:
                with open(os.path.join(work, "rooms", "room", "about.md"), "w") as f:
                    f.write(about)
            named = [
                dict(o, slot=o["slot"].replace("mug", "cup"), category=o["category"].replace("mug", "cup"))
                for o in OBJECTS
            ]
            K.save(
                dict(
                    schema=K.SCHEMA,
                    reference="../../rooms/room/scene.xml",
                    vocabulary=["cup", "table"],
                    objects=named,
                    functionality=dict(units=[unit("robot", ["cup__0", "cabinet__0"])]),
                ),
                os.path.join(res, "case.json"),
            )
            receipt(work)
            err = io.StringIO()
            with (
                metric_environment(),
                mock.patch.object(M.App, "input_pictures", pictures),
                contextlib.redirect_stderr(err),
            ):
                M.main(["--work", work, "--out", out, "--pictures", "1"])
            task = os.path.join(out, "public", os.listdir(os.path.join(out, "public"))[0])
            with open(os.path.join(task, "instruction.md")) as f:
                return f.read(), err.getvalue()

        text, err = run("A children's playroom\nwith a study corner.\n")
        self.assertIn(
            "### Tasks done in your world\n\nA children's playroom with a study corner.\n\n" + M.CASE_TASKS.strip(),
            text,
        )
        self.assertNotIn("has no about.md", err)
        text, err = run(None)
        self.assertIn("### Tasks done in your world\n\n" + M.CASE_TASKS.strip(), text)
        self.assertIn("has no about.md", err)
        with self.assertRaises(SystemExit):
            run("A playroom around cup__0.")

    def test_the_builders_words_reach_the_case(self):
        xml = rooms.write(tempfile.mkdtemp(), rooms.robot_room_xml())
        s = Scene(xml, dict(PHYSICS), probes=True)
        s.settle()
        b = room_bounds(s)
        c = [(b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2]
        spec = dict(
            push=[dict(object="cup__0", words="Push the cup on the table toward the wall.")],
            pick_place=[dict(object="cup__0", goal="cabinet__0")],
        )
        units, _ = T.named(s, C.context(s, c), spec, c, b)
        said = {u["kind"]: u.get("words") for u in units}
        self.assertEqual(said.get("robot_push"), "Push the cup on the table toward the wall.")
        self.assertIsNone(said.get("robot"))

    def test_a_shelving_task_without_its_bay_is_refused_with_a_reason(self):  # it used to raise KeyError
        xml = rooms.write(tempfile.mkdtemp(), rooms.robot_room_xml())
        s = Scene(xml, dict(PHYSICS), probes=True)
        s.settle()
        b = room_bounds(s)
        c = [(b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2]
        ctx = C.context(s, c)

        def refused(task):
            units, bad = T.named(s, ctx, dict(shelve=[dict(object="cup__0", goal="cabinet__0", **task)]), c, b)
            self.assertEqual(units, [])
            return bad[0][1]

        self.assertIn("needs bay, spine, front", refused({}))
        self.assertIn("needs front", refused(dict(bay=[[3.3, 1.6, 0.8], [3.9, 2.4, 1.1]], spine=[-1, 0])))
        self.assertIn(
            "bay must be a box", refused(dict(bay=[[3.3, 1.6, 0.8], [3.9, 2.4, 0.8]], spine=[-1, 0], front=[0, 1]))
        )
        self.assertIn(
            "non-zero directions", refused(dict(bay=[[3.3, 1.6, 0.8], [3.9, 2.4, 1.1]], spine=[0, 0], front=[0, 1]))
        )
        units, bad = T.named(
            s,
            ctx,
            dict(
                shelve=[
                    dict(
                        object="cup__0",
                        goal="cabinet__0",
                        bay=[[3.3, 1.6, 0.8], [3.9, 2.4, 1.1]],
                        spine=[-1, 0],
                        front=[0, 1],
                    )
                ]
            ),
            c,
            b,
        )
        self.assertEqual((len(units), bad), (1, []))


if __name__ == "__main__":
    unittest.main()
