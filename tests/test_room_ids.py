"""Harbor tasks never name the reference room: public tasks carry only its opaque id, the private tasks keep the same
task names (so `harbor job regrade -p OUT/private` pairs them), and every task gives the same public category words.
Needs requirements/core.txt. Run from the repository root:
    python -m unittest tests.test_room_ids -v
"""

import json, os, sys, tempfile, unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from scenebench_harbor import make_tasks as M  # noqa: E402
from sbv.common import case as K  # noqa: E402
from tests import rooms  # noqa: E402
from tests.harbor_fixtures import metric_environment, receipt  # noqa: E402

ROOM = "secretstudyroom"  # no number can contain it


def ready_room(work, room, words, slots=()):
    """A room as the pipeline leaves it: the reference room, its case (only what make_tasks reads) and check.json."""
    rooms.write(os.path.join(work, "rooms", room))
    res = os.path.join(work, "results", room)
    os.makedirs(res)
    K.save(
        dict(
            schema=K.SCHEMA,
            reference=f"../../rooms/{room}/scene.xml",
            vocabulary=words,
            objects=[dict(slot=s, category=s.split("__")[0]) for s in slots],
        ),
        os.path.join(res, "case.json"),
    )
    receipt(work, room)


def pictures(case, out, condition):  # stands in for Blender
    os.makedirs(out, exist_ok=True)
    for f in ("input_0.png", "cameras.json"):
        open(os.path.join(out, f), "w").close()


class RoomIds(unittest.TestCase):
    def setUp(self):
        self.enterContext(metric_environment())

    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp(prefix="sbv_room_ids_")
        cls.out = os.path.join(cls.work, "tasks")
        ready_room(cls.work, ROOM, ["book", "cabinet", "chair", "crate", "cup", "table", "vase"])
        run = ["--work", cls.work, "--out", cls.out, "--pictures", "1"]
        with metric_environment(), mock.patch.object(M.App, "input_pictures", pictures):
            M.main(run)
            cls.first = sorted(os.listdir(os.path.join(cls.out, "public")))
            M.main(run)  # the second run reuses the ids
        with open(os.path.join(cls.out, "private", "room_ids.json")) as f:
            cls.rid = json.load(f)[ROOM]

    def test_no_public_file_names_the_true_room(self):
        hits = []
        for d, ds, fs in os.walk(self.out):
            if d == self.out:
                ds.remove("private")
            for f in fs:
                with open(os.path.join(d, f), "rb") as fh:
                    text = fh.read().lower()
                if ROOM in os.path.join(d, f) or ROOM.encode() in text:
                    hits.append(os.path.join(d, f))
        self.assertEqual(hits, [])
        task = os.path.join(self.out, "public", f"3dscenebench-{self.rid}")
        with open(os.path.join(task, "environment", "room.txt")) as f:
            text = f.read()
        self.assertEqual(text.strip(), self.rid)
        self.assertEqual(len(text), len(self.rid) + 1 + int(self.rid))  # padded: a different size in every room
        with open(os.path.join(task, "instruction.md")) as f:
            self.assertIn(f"build room {self.rid} as", f.read())

    def test_a_task_is_named_by_its_room_number(self):
        self.assertEqual(self.rid, "001")
        self.assertEqual(M.task_name("001"), "3dscenebench-001")  # no picture count in the name
        with self.assertRaisesRegex(SystemExit, "one picture condition"):
            M.main(["--work", self.work, "--out", self.out, "--pictures", "1", "4"])

    def test_rooms_keep_their_numbers_and_old_codes_are_numbered_anew(self):
        path = os.path.join(tempfile.mkdtemp(prefix="sbv_room_ids_n_"), "room_ids.json")
        with open(path, "w") as f:
            json.dump(dict(den="3f9a1c2e", hall="002"), f)
        ids = M.room_ids(path, ["den", "hall", "attic", "porch"])
        self.assertEqual(ids["hall"], "002")  # kept
        self.assertEqual(sorted(ids.values()), ["001", "002", "003", "004"])
        self.assertEqual(M.room_ids(path, ["porch", "den"]), ids)  # the same on the next run
        with open(os.path.join(os.path.dirname(path), "room_ids.before_numbers.json")) as f:
            self.assertEqual(json.load(f)["den"], "3f9a1c2e")  # the old code is not lost

    def test_private_and_public_task_names_match(self):
        tasks = lambda side: sorted(
            x for x in os.listdir(os.path.join(self.out, side)) if os.path.isdir(os.path.join(self.out, side, x))
        )
        self.assertEqual(tasks("public"), [f"3dscenebench-{self.rid}"])
        self.assertEqual((tasks("private"), self.first), (tasks("public"), tasks("public")))

        def name(side, t):
            with open(os.path.join(self.out, side, t, "task.toml")) as f:
                return [x for x in f if x.startswith("name =")]

        for t in tasks("public"):
            self.assertEqual(name("public", t), name("private", t))

    def test_every_task_gives_the_public_list(self):
        with open(os.path.join(self.out, "public", f"3dscenebench-{self.rid}", "environment", "vocabulary.json")) as f:
            self.assertEqual(json.load(f), M.vocabulary(M.categories()))
        self.assertGreater(len(M.categories()), 100)
        env = os.path.join(self.out, "public", f"3dscenebench-{self.rid}", "environment")
        with open(os.path.join(env, "categories.txt")) as f:
            self.assertEqual(f.read(), M.word_file())
        with open(os.path.join(env, "Dockerfile")) as f:
            self.assertIn("categories.txt /task/", f.read())
        with open(os.path.join(env, "instruction.md")) as f:
            text = f.read()
        self.assertIn("`/task/categories.txt`", text)
        self.assertNotIn("bookcase (also", text)  # the list is in the file, not inline

    def test_a_room_with_an_unknown_word_is_refused(self):
        work = tempfile.mkdtemp(prefix="sbv_room_ids_odd_")
        ready_room(work, "oddroom", ["table", "zorbleflux"])
        with self.assertRaises(SystemExit) as e:
            M.main(["--work", work, "--out", os.path.join(work, "tasks"), "--pictures", "1"])
        self.assertIn("zorbleflux", str(e.exception))
        self.assertFalse(os.path.exists(os.path.join(work, "tasks")))

    def test_waiting_words_allow_a_look_at_the_tasks(self):
        work = tempfile.mkdtemp(prefix="sbv_room_ids_wait_")
        ready_room(work, "waitroom", ["table", "zorbleflux"])
        waiting = os.path.join(work, "words-to-add.json")
        with open(waiting, "w") as f:
            json.dump(dict(words=[dict(word="zorbleflux", topic="other household things")]), f)
        with mock.patch.object(M.App, "input_pictures", pictures):
            M.main(
                ["--work", work, "--out", os.path.join(work, "tasks"), "--pictures", "1", "--waiting-words", waiting]
            )
        self.assertTrue(os.path.isdir(os.path.join(work, "tasks", "public")))

    def test_a_room_that_names_an_object_by_an_alias_still_gets_its_tasks(self):
        """An alias in a reference room is scored as its main word, as in a copy, so the room is not refused."""
        import contextlib, io

        work = tempfile.mkdtemp(prefix="sbv_room_ids_alias_")
        ready_room(work, "aliasroom", ["ornament", "table"], ["trinket__0", "table__0"])
        err = io.StringIO()
        with mock.patch.object(M.App, "input_pictures", pictures), contextlib.redirect_stderr(err):
            M.main(["--work", work, "--out", os.path.join(work, "tasks"), "--pictures", "1"])
        self.assertIn("'trinket__0': 'ornament'", err.getvalue())
        self.assertEqual(len(os.listdir(os.path.join(work, "tasks", "public"))), 1)

    def test_pictures_are_rendered_again_only_when_the_room_or_a_picture_changes(self):
        work = tempfile.mkdtemp(prefix="sbv_room_ids_pics_")
        ready_room(work, "picroom", ["table"])
        calls = []

        def render(case, out, condition):
            calls.append(condition)
            pictures(case, out, condition)

        run = ["--work", work, "--out", os.path.join(work, "tasks"), "--pictures", "1"]
        with mock.patch.object(M.App, "input_pictures", render):
            M.main(run)
            M.main(run)
            self.assertEqual(calls, ["1"])  # unchanged: kept
            with open(os.path.join(work, "rooms", "picroom", "scene.xml"), "a") as f:
                f.write("<!-- rebuilt -->\n")
            receipt(work, "picroom")  # explicit replacement evidence for the changed fixture room
            M.main(run)
            self.assertEqual(calls, ["1"] * 2)  # the room changed: again
            with open(os.path.join(work, "tasks", "private", "room_ids.json")) as f:
                rid = json.load(f)["picroom"]
            os.remove(
                os.path.join(work, "tasks", "pictures", f"{rid}-p1", "input_0.png")
            )  # a render that stopped half way
            M.main(run)
            self.assertEqual(calls, ["1"] * 3)

    def test_files_docker_could_mix_up_between_tasks_are_found(self):
        d = tempfile.mkdtemp(prefix="sbv_same_size_")
        for task, cpus in (("3dscenebench-003", 4), ("3dscenebench-034", 8)):
            for ctx in ("environment", "tests"):
                os.makedirs(os.path.join(d, task, ctx))
            with open(os.path.join(d, task, "environment", "categories.txt"), "w") as f:
                f.write("same in every task\n")  # same size, same content: no risk
            with open(os.path.join(d, task, "tests", "test.sh"), "w") as f:
                f.write(f"procs={cpus}\n")  # same size, different content: the file Docker reused
        self.assertEqual(
            M.same_size_files(d), [(os.path.join("tests", "test.sh"), "3dscenebench-003", "3dscenebench-034")]
        )
        for task in os.listdir(d):
            M.pad_test(os.path.join(d, task, "tests", "test.sh"), task.split("-")[1])
        self.assertEqual(M.same_size_files(d), [])  # padded by the room's number: every size differs

    def test_a_stopped_render_leaves_no_spec_behind(self):
        from sbv.appearance import views as App

        out = os.path.join(tempfile.mkdtemp(prefix="sbv_render_"), "pics")
        with self.assertRaises(RuntimeError):
            App.render("/nonexistent/scene.xml", dict(cameras=[]), out)  # no file (or no Blender): it fails
        self.assertEqual(os.listdir(out), [])


if __name__ == "__main__":
    unittest.main()
