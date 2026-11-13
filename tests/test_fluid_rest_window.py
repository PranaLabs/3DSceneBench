"""Fluid stillness is read as net movement over the last 0.2 s before the tilt: particles that only jitter from step to
step read as still, while a fluid that drains still reads as moving."""

import unittest
import numpy as np
from sbv.nonrigid import actions, verify

DT, T_START = 0.002, 0.4


class _Tensor:
    def __init__(self, a):
        self.a = np.asarray(a, float)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Container:
    def __init__(self):
        self.q = np.array([0, 0, 0, 1, 0, 0, 0], float)

    def get_qpos(self):
        return _Tensor(self.q)

    def set_qpos(self, q):
        pass

    def set_dofs_velocity(self, v):
        pass


class _Scene:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


def run(motion):
    """_fluid on a mock: 200 particles inside a 10 cm box; motion(steps) moves them, given the steps taken so far."""
    sc, base = _Scene(), np.random.default_rng(0).uniform(0.03, 0.07, (200, 3))
    P = lambda: base + motion(sc.steps)
    a = dict(yaw_deg=0, angle_deg=75, t_start=T_START, t_ramp=0.5)
    within = lambda p: np.all((p > 0) & (p < 0.1), axis=1)
    return actions._fluid(sc, _Container(), P, a, (np.zeros(3), np.full(3, 0.1)), int(0.5 / DT), DT, within)


class RestWindow(unittest.TestCase):
    def test_jitter_reads_as_still(self):
        r = run(lambda n: np.array([0, 0, 0.001]) * (-1) ** n)  # 1 mm up, 1 mm down, every step: 1 m/s over one step
        self.assertLess(r["rest_speed"], 0.02)
        self.assertEqual(verify.contained(dict(r, finite=True)), [])

    def test_draining_reads_as_moving(self):
        r = run(lambda n: np.array([0, 0, -0.1 * DT * n]))  # sinking at 10 cm/s
        self.assertAlmostEqual(r["rest_speed"], 0.1, delta=0.01)
        self.assertTrue(any("not at rest" in w for w in verify.contained(dict(r, finite=True))))


if __name__ == "__main__":
    unittest.main()
