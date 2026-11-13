"""Local runs (Scene.focus): a test steps only the objects near its own, with the same outcome as the whole room."""

import tempfile, unittest
import numpy as np
from sbv.common.scene import Scene, PHYSICS
from sbv.functionality import f2_physical as P
from tests import rooms


class LocalRuns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = tempfile.mkdtemp()
        rooms.write(d)
        cls.S = Scene(d + "/scene.xml", dict(PHYSICS), probes=True)
        cls.S.settle()

    def test_far_objects_stop_colliding_and_everything_is_restored(self):
        S, m = self.S, self.S.m
        before = (m.geom_contype.copy(), m.geom_conaffinity.copy(), m.body_gravcomp.copy())
        with S.focus(["cup__0"], 0.3):
            off = {n for n in S.objects if not any(S.collides(g) for g in S.ogeoms[n])}
            self.assertIn("cabinet__0", off)  # 2 m away
            self.assertNotIn("cup__0", off)
            self.assertNotIn("table__0", off)  # the cup and what it stands on
        for a, b in zip(before, (m.geom_contype, m.geom_conaffinity, m.body_gravcomp)):
            np.testing.assert_array_equal(a, b)
        with S.focus([], 0.3):
            self.assertTrue(all(S.collides(g) for n in S.objects for g in S.ogeoms[n] if before[0][g]))

    def test_same_outcome_as_the_whole_room(self):
        S = self.S
        for F in (0.5, 20.0):  # one push that holds, one that breaks the cup loose
            whole = P.push_once(S, "cup__0", F, (1.0, 0.0), S.start)[0]
            p_whole = S.d.xipos[S.objects["cup__0"]].copy()
            with S.focus(["cup__0"], 0.3):
                local = P.push_once(S, "cup__0", F, (1.0, 0.0), S.start)[0]
                p_local = S.d.xipos[S.objects["cup__0"]].copy()
            self.assertEqual(whole, local)  # the verdict, whatever happens
            if F < 1:
                np.testing.assert_allclose(
                    p_whole, p_local, atol=1e-3
                )  # and the pose, while it stays near its own objects


if __name__ == "__main__":
    unittest.main()
