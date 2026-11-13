"""The entrants' checker (checker/check_scene.py) stands alone and agrees with the F1 gate. Run from the repository
root:
python -m unittest tests.test_checker -v
"""

import ast, json, os, shutil, subprocess, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from sbv.common import scene as S  # noqa: E402
from sbv.functionality.f1_gate import gate, BLEW_UP  # noqa: E402
from sbv.nonrigid import schema as Nonrigid  # noqa: E402
from tests import rooms  # noqa: E402

CHECKER = os.path.join(ROOT, "checker", "check_scene.py")
sys.path.insert(0, os.path.dirname(CHECKER))
import check_scene as C  # noqa: E402

BOOK = 'name="book_g" type="box" pos="0 0 0" size="0.075 0.11 0.015" density="700"'
BAD_BOOK = rooms.room_xml().replace(BOOK, BOOK + ' friction="nan 0.005 0.0001"')
SPRING = rooms.room_xml().replace(
    'range="0 1.6" damping="0.5"', 'range="0 1.6" stiffness="1e9" springref="1"'
)  # blows up
NO_FLOOR = rooms.room_xml().replace('type="plane" size="0 0 0.1"', 'type="plane" size="0 0 0.1" quat="0.707 0.707 0 0"')
NO_MESH = rooms.room_xml().replace("<worldbody>", '<asset><mesh name="m" file="nothing.obj"/></asset><worldbody>')


def copy(xml):
    d = tempfile.mkdtemp(prefix="sbv_checker_")
    rooms.write(d, xml)
    return os.path.join(d, "scene.xml")


class Checker(unittest.TestCase):
    def test_the_checker_imports_nothing_from_sbv(self):
        with open(CHECKER) as f:
            tree = ast.parse(f.read())
        names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names] + [
            n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
        ]
        self.assertEqual([n for n in names if n.split(".")[0] in ("sbv", "scenebench_harbor", "tests")], [])
        alone = tempfile.mkdtemp(prefix="sbv_alone_")
        shutil.copy(CHECKER, alone)
        xml = copy(None)
        r = subprocess.run(
            [sys.executable, "-I", os.path.join(alone, "check_scene.py"), os.path.dirname(xml)],
            capture_output=True,
            text=True,
            cwd=alone,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(json.loads(r.stdout)["loads_and_runs"])
        self.assertNotEqual(
            subprocess.run([sys.executable, "-I", "-c", "import sbv"], capture_output=True, cwd=alone).returncode,
            0,
            "sbv was importable from an empty folder",
        )

    def test_the_constants_mirror_sbv(self):
        self.assertEqual(C.PHYSICS, {k: v for k, v in S.PHYSICS.items() if k != "mujoco"})
        self.assertEqual(C.ARCH, S.ARCH)
        self.assertEqual(C.ARENA, S.ARENA)
        self.assertEqual(
            (C.KINDS, C.VISCOSITY, C.FABRIC), (Nonrigid.KINDS, tuple(Nonrigid.VISCOSITY), tuple(Nonrigid.FABRIC))
        )

    def test_the_checker_agrees_with_the_gate(self):
        """Same verdict and same findings; a copy the gate rescues by making the objects that blow it up inert fails the
        checker instead, naming those objects among its suspects."""
        for xml, what in (
            (None, "the room"),
            (NO_FLOOR, "no floor"),
            (NO_MESH, "a missing mesh"),
            (BAD_BOOK, "a bad value"),
            (SPRING, "a door spring"),
        ):
            path = copy(xml)
            ok_g, rep_g, _ = gate(path, dict(S.PHYSICS))
            ok_c, rep_c, _ = C.gate(path)
            for k in ("package", "definitions", "units_frames"):
                self.assertEqual(rep_c.get(k), rep_g.get(k), what)
            own = {
                n: [x for x in v if x != BLEW_UP] for n, v in rep_g.get("object_gate", {}).items()
            }  # the gate's findings, without its blow-up note
            self.assertEqual(
                rep_c.get("object_gate"), {n: v for n, v in own.items() if v} if "object_gate" in rep_g else None, what
            )
            if not rep_g.get("inert"):
                self.assertEqual(ok_c, ok_g, what)
                continue
            self.assertTrue(ok_g and not ok_c, what)
            self.assertEqual(rep_c["load_and_step"], [C.UNSTABLE], what)
            self.assertTrue(
                set(rep_g["inert"]) <= set(rep_c["blow_up_suspects"]), (what, rep_g["inert"], rep_c["blow_up_suspects"])
            )

    def test_aliases_and_words_off_the_list(self):
        xml = copy(rooms.room_xml().replace('"cup__0"', '"mug__0"').replace('"crate__0"', '"zorbleflux__0"'))
        words = os.path.join(os.path.dirname(xml), "vocabulary.json")
        with open(words, "w") as f:
            json.dump(
                {"words": ["cup", "mug", "table", "book", "chair", "vase", "cabinet"], "aliases": {"mug": "cup"}}, f
            )
        r = subprocess.run(
            [sys.executable, CHECKER, os.path.dirname(xml), "--vocabulary", words], capture_output=True, text=True
        )
        out = json.loads(r.stdout)
        self.assertEqual(
            (r.returncode, out["aliases"], out["words_not_on_the_list"]),
            (0, ["mug__0 is scored as cup"], ["zorbleflux__0"]),
        )


if __name__ == "__main__":
    unittest.main()
