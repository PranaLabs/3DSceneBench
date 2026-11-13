"""The object gate: a bad value, a hidden constraint or a blow-up in the 6 s run fails only the object that has it, not
the whole copy, as one anywhere used to."""

import os, tempfile, unittest
from sbv import calibrate, evaluate
from sbv.common import case as K
from sbv.functionality.f1_gate import gate
from tests import rooms

BOOK = 'name="book_g" type="box" pos="0 0 0" size="0.075 0.11 0.015" density="700"'
NAN_BOOK = (BOOK, BOOK + ' friction="nan 0.005 0.0001"')
SPRING = (
    'range="0 1.6" damping="0.5"',
    'range="0 1.6" stiffness="1e9" springref="1"',
)  # a door spring that blows up the run


def copy(xml):
    d = tempfile.mkdtemp()
    rooms.write(d, xml)
    return d


def failed(report, name):
    return {
        k
        for k, v in report["units"].items()
        if name in k.split("|")[1:] and not all(x["passed"]["physics_hint"] for x in v)
    }


class ObjectGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ref = copy(None)
        case = calibrate.build(
            ref,
            procs=4,
            draws=2,
            need=2,
            jitter=1,
            robot_tasks=0,
            push_tasks=0,
            require_tasks=False,
            log=lambda *a: None,
        )
        out = os.path.join(tempfile.mkdtemp(), "case.json")
        case["reference"] = os.path.join(ref, "scene.xml")
        K.save(case, out)
        cls.case = K.load(out)

    def score(self, xml):
        return evaluate.evaluate(self.case, copy(xml), procs=4, log=lambda *a: None)

    def test_true_room_records_its_constraints_and_passes(self):
        self.assertTrue(
            all(o["constraints"] == dict(equality=0, gravcomp=0.0, exclude=0) for o in self.case["objects"])
        )
        r = self.score(rooms.room_xml())
        self.assertFalse(r.get("object_gate"))
        self.assertEqual(r["functionality"]["physics_hint"]["functionality"], 100.0)

    def test_a_bad_value_fails_only_its_object(self):
        r = self.score(rooms.room_xml().replace(BOOK, BOOK + ' friction="-0.5 0.005 0.0001"'))
        self.assertTrue(r["F1"]["passed"])  # the copy gate passes: it loads and runs
        self.assertIn("friction", " ".join(r["object_gate"]["book__0"]))
        self.assertTrue(failed(r, "book__0"))
        self.assertFalse(failed(r, "vase__0"))
        self.assertGreater(
            r["functionality"]["physics_hint"]["functionality"], 50
        )  # not 0, as one bad value used to give

    def test_a_hidden_weld_fails_the_welded_objects(self):
        r = self.score(
            rooms.room_xml().replace(
                "</worldbody>", '</worldbody><equality><weld body1="cup__0" body2="table__0"/></equality>'
            )
        )
        self.assertIn("hidden constraint: equality 1 (reference room 0)", r["object_gate"]["cup__0"])
        self.assertTrue(failed(r, "cup__0"))
        self.assertFalse(failed(r, "vase__0"))

    def test_cases_from_before_skip_the_constraint_check(self):
        old = dict(
            self.case, objects=[{k: v for k, v in o.items() if k != "constraints"} for o in self.case["objects"]]
        )
        r = evaluate.evaluate(
            old,
            copy(
                rooms.room_xml().replace(
                    "</worldbody>", '</worldbody><equality><weld body1="cup__0" body2="table__0"/></equality>'
                )
            ),
            procs=4,
            log=lambda *a: None,
        )
        self.assertNotIn("cup__0", r.get("object_gate", {}))

    def test_an_object_that_blows_up_the_run_fails_only_its_own_units(self):
        r = self.score(rooms.room_xml().replace(*SPRING))
        self.assertTrue(r["F1"]["passed"])
        self.assertEqual(r["F1"]["inert"], ["cabinet__0"])
        self.assertIn("blows up", " ".join(r["object_gate"]["cabinet__0"]))
        bad = {k for k, v in r["units"].items() if not all(x["passed"]["physics_hint"] for x in v)}
        self.assertTrue(bad)
        self.assertEqual(bad, failed(r, "cabinet__0"))  # every failed unit is the cabinet's
        self.assertGreater(r["functionality"]["physics_hint"]["functionality"], 50)  # not 0, as a blow-up used to give

    def test_a_bad_value_that_blows_up_the_run_fails_only_its_object(self):
        r = self.score(rooms.room_xml().replace(*NAN_BOOK))  # F1.3 and F1.4 at once
        self.assertEqual(r["F1"]["inert"], ["book__0"])
        self.assertTrue(failed(r, "book__0"))
        self.assertFalse(failed(r, "vase__0"))
        self.assertGreater(r["functionality"]["physics_hint"]["functionality"], 50)

    def test_each_object_that_blows_up_is_found_and_a_blow_up_no_object_explains_fails_the_copy(self):
        two = gate(
            rooms.write(tempfile.mkdtemp(), rooms.room_xml().replace(*SPRING).replace(*NAN_BOOK)), self.case["physics"]
        )[1]
        self.assertEqual(sorted(two["inert"]), ["book__0", "cabinet__0"])
        wall = '<body name="wall__9" pos="2 2.9 1"><joint type="slide" axis="1 0 0" stiffness="1e9" springref="1"/><geom type="box" size="0.1 0.01 0.1"/></body>'
        E, rep, _ = gate(
            rooms.write(tempfile.mkdtemp(), rooms.room_xml().replace("</worldbody>", wall + "</worldbody>")),
            self.case["physics"],
        )
        self.assertFalse(E)
        self.assertEqual(rep["load_and_step"], ["unstable simulation"])
        self.assertNotIn("inert", rep)


if __name__ == "__main__":
    unittest.main()
