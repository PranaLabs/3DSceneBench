"""They are never written."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sbv import reward as R  # noqa: E402
from sbv.nonrigid import schema as S  # noqa: E402
from sbv.package_tasks import schema as PS, verify as PV  # noqa: E402

COFFEE = {
    "name": "coffee__0",
    "kind": "liquid",
    "on": "mug__0",
    "material": {"viscosity": "water", "density": 1000},
    "shape": {"type": "cylinder", "pos": [0, 0, 0.8], "radius": 0.035, "height": 0.06},
}
CUSHION = {
    "name": "cushion__0",
    "kind": "soft_solid",
    "on": "sofa__0",
    "material": {"youngs_kpa": 30, "poisson": 0.3, "density": 200},
    "shape": {"type": "box", "pos": [0, 0, 0.5], "size": [0.4, 0.4, 0.1]},
}


def checker():
    spec = importlib.util.spec_from_file_location(
        "check_scene",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checker", "check_scene.py"),
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class OldNames(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, data):
        with open(os.path.join(self.d, name), "w") as f:
            json.dump(data, f)

    def test_side_file_finds_the_old_name_only_when_the_new_one_is_absent(self):
        new, old = os.path.join(self.d, "nonrigid.json"), os.path.join(self.d, "soft_liquid.json")
        self.assertEqual(S.side_file(self.d), new)  # neither: the new name's path
        self.write("soft_liquid.json", {"objects": []})
        self.assertEqual(S.side_file(self.d), old)
        self.write("nonrigid.json", {"objects": []})
        self.assertEqual(S.side_file(self.d), new)
        self.write("soft_case.json", {})
        self.assertEqual(S.side_file(self.d, S.CASE), os.path.join(self.d, "soft_case.json"))

    def test_an_old_side_file_loads_with_the_new_kinds(self):
        self.write("soft_liquid.json", {"schema": "soft_liquid/v0", "objects": [COFFEE, CUSHION]})
        objs, problems = S.load(S.side_file(self.d))
        self.assertEqual((problems, [o["kind"] for o in objs]), ([], ["fluid", "soft_body"]))

    def test_upgrade_renames_every_kind_in_a_case_and_leaves_the_others(self):
        case = {
            "units": [
                {"id": "S1.coffee__0", "kind": "liquid", "truth_object": dict(COFFEE)},
                {"id": "contain.coffee__0", "kind": "contain"},
                {"id": "S1.jacket__0", "kind": "cloth"},
            ]
        }
        S.upgrade(case)
        self.assertEqual([u["kind"] for u in case["units"]], ["fluid", "contain", "thin_shell"])
        self.assertEqual(case["units"][0]["truth_object"]["kind"], "fluid")

    def test_old_report_rows_count_in_the_same_parts(self):
        for old, new in S.OLD_KINDS.items():
            self.assertEqual(
                R.nonrigid_part({"id": "S1.x__0", "kind": old}), R.nonrigid_part({"id": "S1.x__0", "kind": new})
            )

    def test_package_code_still_gets_ctx_soft_liquid_and_negatives_may_say_soft_set(self):
        self.write("soft_liquid.json", {"objects": [CUSHION]})
        ctx = PV.context(self.d, {}, {}, 0)
        self.assertEqual(ctx["nonrigid"], ctx["soft_liquid"])
        self.assertTrue(ctx["nonrigid"].endswith("soft_liquid.json"))
        neg = [
            {"name": "stiff", "why": "ten times stiffer", "soft_set": {"cushion__0": {"material": {"youngs_kpa": 300}}}}
        ]
        self.assertEqual(PS.negative_problems(neg, {"cushion__0"}), [])

    def test_the_old_interpreter_variable_is_still_read(self):
        with mock.patch.dict(os.environ, {"SBV_SOFT_PYTHON": "/old/python"}, clear=False):
            os.environ.pop("SBV_NONRIGID_PYTHON", None)
            self.assertEqual(PV.python("genesis"), "/old/python")
            with mock.patch.dict(os.environ, {"SBV_NONRIGID_PYTHON": "/new/python"}):
                self.assertEqual(PV.python("genesis"), "/new/python")

    def test_the_checker_names_the_new_kind_and_the_new_file(self):
        C = checker()
        self.write("nonrigid.json", {"objects": [COFFEE]})
        problems = C.nonrigid_problems(os.path.join(self.d, "nonrigid.json"), {"mug__0"})[1]
        self.assertIn("coffee__0: kind 'liquid' is now 'fluid'", problems)
        self.assertEqual(C.KINDS, S.KINDS)
        self.assertEqual(C.OLD_KINDS, S.OLD_KINDS)


if __name__ == "__main__":
    unittest.main()
