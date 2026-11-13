"""Small models (Scene.subset): a local run compiles only the objects near its own, with the same outcome and the same
random draws as the whole room, and refuses when leaving an object out would change the physics."""

import tempfile, unittest
import numpy as np
from sbv.common.scene import Scene, PHYSICS
from sbv.functionality import f2_physical as P, f2_calibrate as C
from tests import rooms


class SmallModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = tempfile.mkdtemp()
        rooms.write(d)
        cls.S = Scene(d + "/scene.xml", dict(PHYSICS), probes=True)
        cls.S.settle()
        cls.sub = cls.S.subset(["cup__0"], 0.3)

    def test_only_the_zone_is_compiled_at_its_settled_start(self):
        S, sub = self.S, self.sub
        self.assertEqual(set(sub.objects), S.zone_objects(["cup__0"], 0.3))
        self.assertNotIn("cabinet__0", sub.objects)
        self.assertLess(sub.m.nq, S.m.nq)
        for n in sub.objects:
            np.testing.assert_allclose(sub.d.xipos[sub.objects[n]], S.d.xipos[S.objects[n]], atol=1e-9)

    def test_same_outcome_as_the_whole_room(self):
        S, sub = self.S, self.sub
        for F in (0.5, 20.0):
            self.assertEqual(
                P.push_once(S, "cup__0", F, (1.0, 0.0), S.start)[0],
                P.push_once(sub, "cup__0", F, (1.0, 0.0), sub.start)[0],
            )

    def test_same_random_draws_as_the_whole_room(self):
        S = self.S
        sub = S.subset(["vase__0"], 0.3)  # the cup, book and chair, earlier in the file, are left out
        for sc in (S, sub):
            saved = C.perturb(sc, np.random.default_rng(7))
            sc.got = {
                n: (sc.m.body_mass[sc.links[n][0]] / saved[0][sc.links[n][0]], sc.m.geom_friction[sc.ogeoms[n][0], 0])
                for n in sub.objects
            }
            sc.m.body_mass[:], sc.m.body_inertia[:], sc.m.geom_friction[:] = saved
        for n in sub.objects:
            np.testing.assert_allclose(S.got[n], sub.got[n])
        f, g = S.free_dof("vase__0")[0], sub.free_dof("vase__0")[0]
        np.testing.assert_allclose(
            S.jittered(3)[f : f + 2] - S.start[f : f + 2], sub.jittered(3)[g : g + 2] - sub.start[g : g + 2]
        )

    def test_the_case_switch_picks_small_model_or_whole_room(self):
        from sbv import calibrate as Cal

        Cal.S = self.S
        for small in (True, False):
            Cal.OPT = dict(local_m=0.3, small=small)
            self.assertEqual(Cal._local(["cup__0"], lambda: Cal.S is not self.S), small)
        self.assertIs(Cal.S, self.S)

    def test_refuses_when_a_kept_object_is_tied_to_a_left_out_one(self):
        d = tempfile.mkdtemp()
        rooms.write(
            d,
            rooms.room_xml().replace(
                "</worldbody>", '</worldbody><equality><weld body1="cup__0" body2="cabinet__0"/></equality>'
            ),
        )
        S = Scene(d + "/scene.xml", dict(PHYSICS), probes=True)
        S.settle()
        self.assertIsNone(S.subset(["cup__0"], 0.3))


if __name__ == "__main__":
    unittest.main()
