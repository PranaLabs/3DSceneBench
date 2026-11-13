"""Doors and drawers need a handle when the true part has one: calibration records whether the true part has a handle
the robot hand can pinch, and a copy whose part has none fails the "handle" component."""

import tempfile, unittest
from sbv.common.scene import Scene
from sbv.functionality import f3_functions as Fn, f3_calibrate as CF
from tests import rooms


def cabinet(knob):
    s = Scene(rooms.write(tempfile.mkdtemp(prefix="sbv_handle_"), rooms.cabinet_room_xml(knob)), probes=True)
    s.settle()
    return s


def joint(s, name):
    return next(j for j in s.joints["cabinet__0"] if Fn.jname(s.m, j) == name)


class HandleCheck(unittest.TestCase):
    def test_a_flush_door_fails_where_the_true_door_has_a_knob(self):
        true = cabinet(True)
        unit, why = CF.cal_joint(true, "cabinet__0", joint(true, "door_hinge"))
        self.assertIsNone(why)
        self.assertTrue(unit["params"]["ref"]["handle"])
        self.assertTrue(Fn.joint_trial(true, unit)[0]["handle"])
        flush = cabinet(False)
        self.assertFalse(Fn.joint_trial(flush, unit)[0]["handle"])
        self.assertTrue(Fn.has_handle(flush, "cabinet__0", joint(flush, "drawer_slide")))  # the drawer keeps its bar

    def test_no_requirement_without_a_true_handle(self):
        flush = cabinet(False)
        unit, why = CF.cal_joint(flush, "cabinet__0", joint(flush, "door_hinge"))
        self.assertIsNone(why)
        self.assertFalse(unit["params"]["ref"]["handle"])
        self.assertTrue(Fn.joint_trial(cabinet(False), unit)[0]["handle"])


if __name__ == "__main__":
    unittest.main()
