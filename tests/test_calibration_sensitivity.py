"""Sensitivity early stopping preserves every pass bar and final physics state."""

import itertools
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from sbv.functionality import f2_calibrate as C


class Sensitivity(unittest.TestCase):
    def run_sequence(self, outcomes, need, expect=True, params=None):
        arrays = [np.array([1.0]), np.array([2.0]), np.array([[0.5, 0.0, 0.0]])]
        model = SimpleNamespace(body_mass=arrays[0], body_inertia=arrays[1], geom_friction=arrays[2])
        scene = SimpleNamespace(m=model, d=None)
        state = {"draw": -1, "calls": [], "last": None}

        def perturb(s, rng):
            saved = tuple(a.copy() for a in arrays)
            state["draw"] += 1
            for a in arrays:
                a[:] = rng.random()
            return saved

        def trial(s, u):
            i = state["draw"]
            state["calls"].append(i)
            state["last"] = i
            if outcomes[i] is None:
                raise C.Unstable("fixture")
            return outcomes[i], []

        with patch.object(C, "perturb", side_effect=perturb), patch.object(C.mujoco, "mj_setConst"):
            result = C.decidable(
                scene, {"id": "fixture", "expect": expect, "params": params or {}}, len(outcomes), need, run=trial
            )
        for actual, expected in zip(arrays, (1.0, 2.0, np.array([[0.5, 0.0, 0.0]]))):
            np.testing.assert_array_equal(actual, np.full_like(actual, expected))
        return result, state

    def test_exhaustive_decision_equivalence(self):
        for xs in itertools.product((False, True, None), repeat=5):
            for need in (0, 3, 5, 6):
                got, state = self.run_sequence(xs, need)
                self.assertEqual(got, [sum(x is True for x in xs) >= need])
                self.assertEqual(state["last"], 4)  # downstream state matches legacy last draw

    def test_two_failures_skip_redundant_rollouts(self):
        got, state = self.run_sequence([False, False] + [True] * 8, 9)
        self.assertEqual(got, [False])
        self.assertEqual(state["calls"], [0, 1, 9])

    def test_two_bars_stop_only_when_both_are_decided(self):
        # Outcomes hold both bars one after the other (main bar first), as push units do with P.BARS.
        cases = list(
            itertools.product(
                ([False, True, False, True], [True, False, True, False], [False, True, True, False], None), repeat=4
            )
        )
        for xs in cases:
            for need in (0, 2, 3, 4):
                got, state = self.run_sequence(
                    xs, need, expect=[False, True, False, True], params=dict(bars=[2.0, 1.25])
                )
                hold = [sum(x is not None and x[2 * b : 2 * b + 2] == [False, True] for x in xs) for b in (0, 1)]
                self.assertEqual(got, [h >= need for h in hold])
                self.assertEqual(state["last"], 3)
        # The main bar fails twice but the reference bar can still pass: no draw is skipped.
        ok, miss = [False, True, False, True], [True, False, False, True]
        got, state = self.run_sequence([miss, miss] + [ok] * 8, 9, expect=ok, params=dict(bars=[2.0, 1.25]))
        self.assertEqual(got, [False, True])
        self.assertEqual(state["calls"], list(range(10)))

    def test_unexpected_errors_restore_parameters(self):
        # Parameter restoration also runs when the simulator fails unexpectedly.
        m = SimpleNamespace(body_mass=np.ones(1), body_inertia=np.ones(1), geom_friction=np.ones((1, 3)))
        s = SimpleNamespace(m=m, d=None)

        def perturb(s, rng):
            saved = (m.body_mass.copy(), m.body_inertia.copy(), m.geom_friction.copy())
            m.body_mass[:] = 20
            return saved

        def broken(*a):
            raise RuntimeError("unexpected")

        with patch.object(C, "perturb", side_effect=perturb), patch.object(C.mujoco, "mj_setConst"):
            with self.assertRaises(RuntimeError):
                C.decidable(s, {"id": "x", "expect": True, "params": {}}, 10, 9, run=broken)
        self.assertEqual(m.body_mass[0], 1)


class PhysicsParity(unittest.TestCase):
    def test_sensitivity_keeps_legacy_outcome_and_final_physics_state(self):
        import zlib
        from sbv.common.scene import Scene
        from sbv.functionality import f2_physical as P
        from tests import rooms

        xml = """<mujoco><worldbody><geom type="plane" size="0 0 .1"/>
        <body name="box__0" pos="0 0 .05"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
        </worldbody></mujoco>"""
        with tempfile.TemporaryDirectory() as tmp:
            path = rooms.write(tmp, xml)
            bars = [2.0, 1.25]
            for kind, params in [
                ("push", dict(F=3.0, dir=[1.0, 0.0], bars=bars)),
                ("tilt", dict(angle=10.0, bars=bars, angles=[[7.0, 13.0], [8.0, 12.0]], dir=[1.0, 0.0])),
                ("slide", dict(v=0.3, dir=[1.0, 0.0], bars=bars)),
            ]:
                # Deliberately impossible expectation exercises the skipped middle draws.
                unit = dict(id="parity|" + kind, kind=kind, slots=["box__0"], params=params, expect=["impossible"])
                old = Scene(path, probes=True)
                old.settle()
                rng = np.random.default_rng(zlib.crc32(unit["id"].encode()))
                hold = 0
                for _ in range(10):
                    saved = C.perturb(old, rng)
                    try:
                        hold += P.trial(old, unit)[0] == unit["expect"]
                    finally:
                        old.m.body_mass[:], old.m.body_inertia[:], old.m.geom_friction[:] = saved
                        C.mujoco.mj_setConst(old.m, old.d)
                new = Scene(path, probes=True)
                new.settle()
                self.assertEqual(C.decidable(new, unit, 10, 9), [hold >= 9] * len(bars))
                for a in ("qpos", "qvel", "qacc"):
                    np.testing.assert_allclose(getattr(old.d, a), getattr(new.d, a), rtol=0, atol=1e-12)
