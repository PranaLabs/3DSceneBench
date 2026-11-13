"""The at-rest test's role rule: a piece the reference room fixes may stand free if it stays put; a movable reference
object welded in place fails; the joints must match."""

import unittest
from sbv.functionality.f2_physical import init_passes, role_ok


def rec(role, still=True):
    return dict(
        role=role,
        unstable=False,
        mass=True,
        inertia=True,
        com=True,
        contact=True,
        still=still,
        joints_still=True,
        mass_kg=10.0,
        integrity=dict(crossed=[], max_penetration=0.0, penetration_upper=0.0, sunk_s=0.0),
    )


def unit(role):
    return dict(expect=dict(role=role, joints_still=True, mass_range=None), tol=0.005, rules="R1-sink5mm-0.3s/R2")


class RoleRule(unittest.TestCase):
    def test_roles(self):
        for true, copy, ok in (
            ("fixed", "free", True),
            ("fixed+joints", "free+joints", True),
            ("free", "free", True),
            ("free", "fixed", False),
            ("free+joints", "fixed+joints", False),
            ("fixed+joints", "free", False),
            ("fixed", "free+joints", False),
            ("free+joints", None, False),
        ):
            self.assertEqual(role_ok(true, copy), ok, (true, copy))

    def test_a_free_standing_cabinet_must_still_stay_put(self):
        self.assertTrue(init_passes(unit("fixed+joints"), rec("free+joints")))
        self.assertFalse(init_passes(unit("fixed+joints"), rec("free+joints", still=False)))
        self.assertFalse(init_passes(unit("free"), rec("fixed")))


if __name__ == "__main__":
    unittest.main()
