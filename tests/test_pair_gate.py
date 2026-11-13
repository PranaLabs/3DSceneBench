"""Position gate (sbv/pair_gate.py, sbv/common/case.py gate()): a slot pairs by its word only with a copy object within
its gate, max(0.3 m, the reference object's own box diagonal), whether or not the case froze gates; a robot task's
object reaches at least its true support's diagonal. Pairs across words reach at most min(1 m, max(0.3 m, half the true
box diagonal))."""

import tempfile, unittest
from types import SimpleNamespace
import numpy as np
from sbv.common import case as K
from sbv.common.scene import Scene
from sbv.pair_gate import FACTOR, FLOOR, set_gates
from tests import rooms


def boxes(objs):
    """name -> (centre, size): a stand-in Scene with boxes only (as in tests/test_place_pairing.py)."""
    c = {n: np.array(v[0], float) for n, v in objs.items()}
    h = {n: np.array(v[1], float) / 2 for n, v in objs.items()}
    return SimpleNamespace(
        objects=objs,
        cat={n: n.split("__")[0] for n in objs},
        center=lambda n: c[n],
        aabb=lambda n: (c[n] - h[n], c[n] + h[n]),
        joints={n: [] for n in objs},
    )


class PairGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = Scene(rooms.write(tempfile.mkdtemp(prefix="sbv_gate_")))
        cls.s.reset(cls.s.q0)

    def slots(self, shift):
        return [
            dict(
                slot=n,
                category=n.split("__")[0],
                role=self.s.role[n],
                dims=[0.1, 0.1, 0.1],
                center=(np.asarray(self.s.center(n)) + (shift if n == "cup__0" else 0)).tolist(),
            )
            for n in self.s.objects
        ]

    def cup(self, shift, gates=True):
        """The copy object the true cup pairs with when the true cup stands shift metres from it (a 17 cm cup)."""
        slots = self.slots(np.array([shift, 0, 0]))
        return K.correspond(set_gates(dict(objects=slots))["objects"] if gates else slots, self.s).get("cup__0")

    def test_far_copy_object_never_pairs_by_word(self):
        self.assertIsNone(self.cup(1.0, gates=False))  # 1 m off, no gate frozen: the default gate holds
        self.assertIsNone(self.cup(1.0))  # outside its gate: missing, its units fail
        self.assertEqual(self.cup(0.05), "cup__0")
        self.assertEqual(self.cup(0.05, gates=False), "cup__0")

    def test_a_robot_task_object_reaches_across_its_support(self):
        """The true cup stands on a 1.6 m table and is a robot task's object: a copy cup 1 m along the table still pairs
        (task_reach: the support's diagonal), a cup 2 m off does not; without the task, 1 m off is missing."""
        cup = dict(slot="cup__0", category="cup", role="free", center=[0, 0, 0.8], dims=[0.08, 0.08, 0.1])
        table = dict(slot="table__0", category="table", role="free", center=[0, 0, 0.4], dims=[1.6, 0.8, 0.8])
        task = dict(
            objects=[cup, table],
            support_pairs=[["table__0", "cup__0"]],
            functionality=dict(units=[dict(id="robot_push|cup__0", kind="robot_push", slots=["cup__0"])]),
        )
        K.task_reach(task)
        self.assertAlmostEqual(cup["_task_reach"], float(np.linalg.norm([1.6, 0.8, 0.8])), places=3)
        for at, found in ((1.0, True), (2.0, False)):
            got = K.correspond([cup], boxes({"cup__0": ((at, 0, 0.8), (0.08, 0.08, 0.1))}))
            self.assertEqual(bool(got), found, at)
        plain = dict(cup, _task_reach=None)
        plain.pop("_task_reach")
        self.assertEqual(K.correspond([plain], boxes({"cup__0": ((1.0, 0, 0.8), (0.08, 0.08, 0.1))})), {})

    def test_pairs_across_words_reach_at_most_1_m(self):
        """Steps 2 and 3 of correspond() (moving parts, place) pair different words only within cross_gate(), min(1 m,
        max(0.3 m, half the true box diagonal))."""
        door = dict(slot="door__0", category="door", role="fixed+joints", center=[0, 0, 1.2], dims=[1.0, 0.1, 2.4])
        chair = dict(slot="chair__0", category="chair", role="free", center=[0, 0, 0.38], dims=[0.85, 0.84, 0.76])
        for at, found in ((0.5, True), (2.0, False)):  # the door's own size is 2.6 m: before, 2 m paired too
            sc = boxes({"cabinet__0": ((at, 0, 1.2), (0.8, 0.4, 0.9))})
            sc.joints = {"cabinet__0": ["cabinet__0_hinge"]}
            self.assertEqual(bool(K.correspond([door], sc)), found, at)
        self.assertAlmostEqual(K.cross_gate(chair), 0.5 * float(np.linalg.norm(chair["dims"])), places=6)  # 0.71 m
        for at, found in ((0.5, True), (1.0, False)):  # an armchair at the chair's place; 1 m off passed before
            self.assertEqual(
                bool(K.correspond([chair], boxes({"armchair__0": ((at, 0, 0.38), (0.85, 0.84, 0.76))}))), found, at
            )

    def test_a_same_word_pair_needs_a_like_size(self):
        """The word step pairs objects of one word only when their box diagonals are within 2x: a 1.6 m potted plant
        does not take a 0.4 m one 1 m away; a 1.2 m one there still pairs."""
        plant = dict(slot="plant__0", category="plant", role="free", center=[0, 0, 0.8], dims=[0.9, 0.9, 1.1])
        for size, found in (((0.2, 0.2, 0.3), False), ((0.7, 0.7, 0.8), True)):
            self.assertEqual(bool(K.correspond([plant], boxes({"plant__0": ((1.0, 0, 0.8), size)}))), found, size)

    def test_the_floor_is_0_3_m(self):
        self.assertEqual((FACTOR, FLOOR), (1.0, 0.3))
        self.assertEqual(self.cup(0.25), "cup__0")  # 25 cm off is still the cup
        self.assertIsNone(self.cup(0.35))

    def test_gate_from_own_size(self):
        c = set_gates(
            dict(objects=[dict(dims=[2.0, 1.0, 2.0]), dict(dims=[0.05, 0.05, 0.05])]), pose_factor=0
        )  # gates only
        self.assertEqual([o["gate"] for o in c["objects"]], [3.0, 0.3])
        self.assertEqual(c["pairing"], dict(gate="max(floor, factor x the box diagonal)", factor=1.0, floor=0.3))
        table = set_gates(
            dict(
                objects=[dict(slot="table__0", category="table", role="free", center=[0, 0, 0.4], dims=[0.6, 0.8, 0.0])]
            )
        )["objects"]  # box diagonal 1 m
        for at, found in ((0.9, True), (1.1, False)):
            self.assertEqual(
                bool(K.correspond(table, boxes({"table__0": ((at, 0, 0.4), (0.6, 0.8, 0.02))}))), found, at
            )


if __name__ == "__main__":
    unittest.main()
