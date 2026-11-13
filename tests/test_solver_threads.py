"""Solver threads (SBV_SOLVER_THREADS, SBV_WORKER_SOLVER_THREADS) leave every state array unchanged, and F1 checks its
runtime budget before every step."""

import os, tempfile, unittest
from unittest import mock
import numpy as np
from sbv.common.scene import Scene, RuntimeBudget, PHYSICS
from sbv.common.workers import Workers
from sbv.functionality.f1_gate import gate
from tests import rooms

STATE = ("qpos", "qvel", "qacc", "qacc_warmstart", "efc_force")
TOKEN = None


def _keep(token):
    global TOKEN
    TOKEN = token


def _report(_):
    return os.environ.get("SBV_SOLVER_THREADS"), TOKEN


class SolverThreads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.xml = rooms.write(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def scene(self, threads):
        with mock.patch.dict(os.environ, {"SBV_SOLVER_THREADS": threads}):
            return Scene(self.xml, probes=True)

    def test_threads_give_the_same_state_every_step(self):
        one, four = self.scene("1"), self.scene("4")
        for _ in range(300):  # the room settling: contacts made and lost
            one.step(0.002)
            four.step(0.002)
            for f in STATE:
                np.testing.assert_array_equal(getattr(one.d, f), getattr(four.d, f))

    def test_bad_thread_counts_are_refused(self):
        for value in ("0", "-1", "two"):
            with self.assertRaises(ValueError):
                self.scene(value)
            with mock.patch.dict(os.environ, {"SBV_WORKER_SOLVER_THREADS": value}), self.assertRaises(ValueError):
                Workers(2, _keep, ())

    def test_a_deadline_stops_before_the_next_step(self):
        stopped, done = self.scene("1"), self.scene("1")
        done.step(0.002)
        with mock.patch("sbv.common.scene.time.process_time", side_effect=[0.0, 2.0]), self.assertRaises(RuntimeBudget):
            stopped.step(0.004, deadline=1.0)
        self.assertEqual(stopped.d.time, done.d.time)
        for f in STATE:
            np.testing.assert_array_equal(getattr(stopped.d, f), getattr(done.d, f))

    def test_a_spent_budget_fails_the_gate_before_any_step(self):
        ok, rep, scene = gate(self.xml, dict(PHYSICS), budget_s=0.0)
        self.assertFalse(ok)
        self.assertEqual(rep["load_and_step"], ["over the runtime budget of 0 s"])
        self.assertEqual(scene.d.time, 0.0)

    def test_worker_threads_apply_in_the_workers_only(self):
        with mock.patch.dict(os.environ, {"SBV_SOLVER_THREADS": "8", "SBV_WORKER_SOLVER_THREADS": "2"}):
            with Workers(2, _keep, ("kept",)) as pool:
                self.assertEqual(list(pool.map(_report, range(4))), [("2", "kept")] * 4)
            with Workers(1, None, ()) as pool:
                self.assertEqual(list(pool.map(_report, range(1))), [("2", None)])  # no initializer
            self.assertEqual(os.environ["SBV_SOLVER_THREADS"], "8")
        with mock.patch.dict(os.environ, {"SBV_SOLVER_THREADS": "3"}):
            os.environ.pop("SBV_WORKER_SOLVER_THREADS", None)
            with Workers(1, _keep, ("kept",)) as pool:
                self.assertEqual(list(pool.map(_report, range(1))), [("3", "kept")])


class CpuBudget(unittest.TestCase):
    def test_the_budget_is_per_solver_thread(self):
        import os
        from unittest import mock
        from sbv.functionality.f1_gate import cpu_budget

        self.assertEqual(cpu_budget(300.0), 300.0)
        with mock.patch.dict(os.environ, {"SBV_SOLVER_THREADS": "4"}):
            self.assertEqual(cpu_budget(300.0), 1200.0)


if __name__ == "__main__":
    unittest.main()
