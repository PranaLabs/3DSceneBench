"""Run rule R1, one limit for every room: no object stays sunk more than 5 mm into another for 0.3 s or longer."""

import tempfile, unittest
from sbv.common.scene import Scene, PHYSICS
from sbv.functionality import f2_physical as P
from sbv.functionality.integrity import HOLD_S
from sbv.functionality.units import judge, RULES
from tests import rooms


def settle(xml):
    d = tempfile.mkdtemp()
    rooms.write(d, xml)
    S = Scene(d + "/scene.xml", dict(PHYSICS))
    return P.init_record(S)


class RunRules(unittest.TestCase):
    def test_true_room_never_sinks_a_mushy_copy_does(self):
        true = settle(rooms.room_xml())
        self.assertTrue(all(r["integrity"]["sunk_s"] < HOLD_S for r in true.values()))
        mushy = settle(
            rooms.room_xml().replace("<default><geom ", '<default><geom solref="0.2 1" ')
        )  # contacts 10x softer
        sunk = {n for n, r in mushy.items() if r["integrity"]["sunk_s"] >= HOLD_S}
        self.assertTrue(sunk, "a room of mushy contacts should leave something sunk more than 5 mm")
        unit = dict(id="init|x", kind="init", expect=dict(role=mushy[next(iter(sunk))]["role"]), tol=1.0, rules=RULES)
        self.assertFalse(
            P.init_passes(unit, mushy[next(iter(sunk))])
        )  # fails under R1 although its old tolerance (1 m) is huge

    def test_old_cases_keep_their_tolerance(self):
        class Mon:
            owners, ok_rules = {1}, lambda self, owners, sink=True: False

            def ok(self, owners, tol):
                return tol > 0.5

            def report(self, owners):
                return {}

        old = dict(kind="push", expect=[True], tol=1.0)
        new = dict(old, rules=RULES)
        self.assertTrue(judge(old, [True], [Mon()])["passed"]["physics_hint"])
        self.assertFalse(judge(new, [True], [Mon()])["passed"]["physics_hint"])


if __name__ == "__main__":
    unittest.main()
