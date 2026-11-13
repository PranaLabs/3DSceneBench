"""Each cached stage depends on the code its entry points import (sbv/common/code_dependencies.py). What an import
cannot show is listed there; these tests fail when a process start, a data file or an ignored import is not."""

import ast, json, os, re, shutil, tempfile, unittest
from pathlib import Path
from unittest import mock
from sbv.common import code_dependencies as CD

ROOT = CD.ROOT
MODULE = re.compile(r"\bsbv(?:\.[A-Za-z_]\w*)+")
# Non-code files no cached stage reads, with where they are read instead.
NOT_DATA = {"common/categories_released.txt": "sbv/words.py compares a release's word list; no stage reads it"}


def started(path):
    """The sbv files a module starts as a new process: `-m sbv.x`, code given to `-c` that imports sbv, or a script
    path built as os.path.join(..., "sbv", ..., "x.py")."""
    out = set()
    for call in (n for n in ast.walk(ast.parse(path.read_bytes())) if isinstance(n, ast.Call)):
        if not ast.unparse(call.func).startswith("subprocess."):
            continue
        for node in ast.walk(call):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for name in MODULE.findall(node.value):
                    f = CD._module(name, ROOT)
                    if f:
                        out.add(f.relative_to(ROOT).as_posix())
            if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("path.join"):
                parts = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
                if "sbv" in parts and parts[-1].endswith(".py"):
                    out.add("/".join(parts[parts.index("sbv") + 1 :]))
    return out


class Coverage(unittest.TestCase):
    def test_every_process_start_is_listed(self):
        for path in sorted(ROOT.rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            if rel in CD.BOOKKEEPING or rel == "pipeline.py":
                continue  # pipeline's selftest starts unittest only
            listed = {f for f, _ in CD.SUBPROCESS.get(rel, ())}
            self.assertLessEqual(started(path), listed, f"{rel} starts code by name that SUBPROCESS does not list")
        for rel, targets in CD.SUBPROCESS.items():
            self.assertEqual(
                {f for f, _ in targets}, started(ROOT / rel), f"SUBPROCESS lists for {rel} what it does not start"
            )

    def test_every_data_file_is_listed(self):
        data = {f for files in CD.DATA.values() for f in files}
        for path in sorted(
            p
            for p in ROOT.rglob("*")
            if p.is_file() and p.suffix not in (".py", ".pyc") and "__pycache__" not in p.parts
        ):
            rel = path.relative_to(ROOT).as_posix()
            self.assertTrue(
                rel in data or rel in NOT_DATA, f"{rel}: list it in DATA beside the module that reads it, or here"
            )
        for reader, files in CD.DATA.items():
            text = (ROOT / reader).read_text()
            for f in files:
                self.assertIn(Path(f).name, text, f"DATA says {reader} reads {f}")

    def test_ignored_imports_are_real_imports(self):
        for stage, edges in CD.IGNORE.items():
            self.assertIn(stage, CD.STAGES)
            for src, dst in edges:
                self.assertIn(ROOT / dst, CD.imports(ROOT / src, ROOT), f"{src} does not import {dst}")

    def test_entry_points_exist(self):
        for stage, entries in CD.STAGES.items():
            for e in entries:
                self.assertTrue((ROOT / e).is_file(), f"{stage}: {e}")


class Closures(unittest.TestCase):
    def test_what_each_stage_runs_is_in_and_what_it_does_not_is_out(self):
        rigid, render, appearance, nonrigid = (set(CD.files(s)) for s in ("rigid", "render", "appearance", "nonrigid"))
        # imported by calibrate.build's code path, missed by the hand-written list this replaces
        self.assertLessEqual({"common/calibration_jobs.py", "common/gravity_cache.py", "common/categories.txt"}, rigid)
        self.assertTrue(
            {"reward.py", "appearance/render.py", "appearance/metrics.py", "common/phase_cache.py"}.isdisjoint(rigid)
        )
        self.assertLessEqual({"appearance/views.py", "appearance/render.py"}, render)
        self.assertTrue(
            {"appearance/metrics.py", "functionality/f4_robot/plan.py"}.isdisjoint(render)
        )  # renders never run them
        self.assertIn("appearance/metrics.py", appearance)
        self.assertIn("nonrigid/world.py", nonrigid)
        self.assertNotIn("common/scene.py", nonrigid)

    def test_a_change_in_an_imported_module_invalidates_and_an_unrelated_one_does_not(self):
        before = CD.native("rigid")
        original = CD.sha

        def changed(name):
            return lambda p: "changed" if str(p).endswith(name) else original(p)

        with mock.patch.object(CD, "sha", side_effect=changed("common/calibration_jobs.py")):
            self.assertNotEqual(before, CD.native("rigid"))
        with mock.patch.object(CD, "sha", side_effect=changed("common/categories.txt")):
            self.assertNotEqual(before, CD.native("rigid"))
        with mock.patch.object(CD, "sha", side_effect=changed("sbv/reward.py")):
            self.assertEqual(before, CD.native("rigid"))

    def test_a_new_module_joins_without_being_listed(self):
        tmp = Path(tempfile.mkdtemp()) / "sbv"
        shutil.copytree(ROOT, tmp, ignore=shutil.ignore_patterns("__pycache__"))
        (tmp / "common/newhelper.py").write_text("X = 1\n")
        f = tmp / "calibrate.py"
        f.write_text(f.read_text() + "\ndef _new():\n    from sbv.common.newhelper import X\n    return X\n")
        self.assertIn("common/newhelper.py", CD.files("rigid", tmp))
        self.assertNotIn("common/newhelper.py", CD.files("nonrigid", tmp))

    def test_package_controllers_depend_on_the_whole_package(self):
        self.assertEqual(
            set(CD.native("package", ["x.py"])),
            {
                p.relative_to(ROOT).as_posix()
                for p in ROOT.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
            },
        )


if __name__ == "__main__":
    unittest.main()
