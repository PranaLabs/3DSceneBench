import json, os, shutil, tempfile, unittest

from sbv.nonrigid import declare
from sbv.package_tasks import verify as pkg

SCENE = """<mujoco><worldbody>
  <body name="towel_rack__0" pos="0 0 1"><geom type="box" size="0.3 0.02 0.02"/>
    <body name="towel__0" pos="0 0 -0.1"><geom type="box" size="0.25 0.01 0.2" mass="0.3"/></body></body>
  <body name="towel__1" pos="1 0 0.5"><freejoint/><geom type="box" size="0.25 0.01 0.2" mass="0.3"/></body>
  <body name="cup__3" pos="2 0 0.5"><freejoint/><geom type="cylinder" size="0.04 0.05" mass="0.2"/></body>
  <body name="cup__7" pos="0.2 0 0.5"><freejoint/><geom type="cylinder" size="0.04 0.05" mass="0.2"/></body>
</worldbody></mujoco>"""


def room(nonrigid=None, case=None):
    d = tempfile.mkdtemp(prefix="sbv_nr_")
    r = os.path.join(d, "room")
    os.makedirs(r)
    open(os.path.join(r, "scene.xml"), "w").write(SCENE)
    if nonrigid is not None:
        json.dump(dict(objects=nonrigid), open(os.path.join(r, "nonrigid.json"), "w"))
    if case is not None:
        json.dump(case, open(os.path.join(d, "case.json"), "w"))
    return d, r


class Declare(unittest.TestCase):
    def test_lists_only_undeclared_deformable_words(self):
        d, r = room(nonrigid=[dict(name="towel__1", kind="thin_shell")])
        self.addCleanup(shutil.rmtree, d, True)
        self.assertEqual(
            declare.missing(r), [("towel__0", "thin_shell")]
        )  # the cups are rigid words; towel__1 is declared
        rows = declare.draft(r)
        self.assertEqual(
            (rows[0]["name"], rows[0]["kind"], rows[0]["on"], rows[0]["shape"]),
            ("towel__0", "thin_shell", "towel_rack__0", None),
        )  # a cloth's drape is for the builder

    def test_no_scene_no_warning(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        self.assertEqual(declare.missing(d), [])


class RoleBinding(unittest.TestCase):
    def test_an_unpaired_task_object_binds_to_the_nearest_unclaimed_of_its_word(self):
        d, r = room()
        self.addCleanup(shutil.rmtree, d, True)
        at = pkg.body_centres(os.path.join(r, "scene.xml"))
        true_at = {"cup__18": (at["cup__7"][0] * 0, True)}  # the true cup at the origin
        self.assertEqual(
            pkg.by_role({"cup__18": None}, true_at, at, {"cup__7"}), {"cup__18": "cup__3"}
        )  # cup__7 is claimed elsewhere
        self.assertEqual(pkg.by_role({"cup__18": None}, true_at, at, set()), {"cup__18": "cup__7"})  # nearest free cup
        self.assertEqual(
            pkg.by_role({"cup__18": None, "cup__2": "cup__7"}, true_at, at, set()),
            {"cup__18": "cup__3", "cup__2": "cup__7"},
        )  # not one this task uses
        self.assertEqual(
            pkg.by_role({"cup__18": "cup__3"}, true_at, at, set()), {"cup__18": "cup__3"}
        )  # nothing missing: unchanged

    def test_tasks_do_not_take_each_others_stand_ins(self):
        d, r = room()
        self.addCleanup(shutil.rmtree, d, True)
        at = pkg.body_centres(os.path.join(r, "scene.xml"))
        true_at = {"cup__18": (at["cup__7"][0] * 0, True)}
        self.assertEqual(pkg.by_role({"cup__18": None}, true_at, at, set())["cup__18"], "cup__7")
        self.assertEqual(pkg.by_role({"cup__18": None, "box__1": None}, true_at, at, set())["cup__18"], "cup__7")


if __name__ == "__main__":
    unittest.main()
