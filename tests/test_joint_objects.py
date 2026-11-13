"""Moving parts per object: an object's joint units count as one unit, the share of them that pass; other units count
one each."""

import math
import unittest
from sbv.functionality.aggregate import levels, functionality, level_points, parts_points


def unit(uid, kind, slot):
    return dict(id=uid, kind=kind, slots=[slot], params={})


def res(ok):
    return [
        dict(passed=dict(physics_hint=ok, input_only=ok), passed_reference_bar=dict(physics_hint=ok, input_only=ok))
    ]


class JointObjects(unittest.TestCase):
    def test_a_piano_of_four_keys_is_one_unit(self):
        units = [unit(f"joint|piano__0|k{i}", "joint", "piano__0") for i in range(4)] + [
            unit("joint|cabinet__0|door", "joint", "cabinet__0"),
            unit("push|chair__0", "push", "chair__0"),
        ]
        results = {u["id"]: res(ok) for u, ok in zip(units, (True, True, True, False, False, True))}
        lv = levels(1.0, units, results)["input_only"]
        self.assertEqual(sorted(lv["affordance"]), [0.0, 0.75])
        self.assertEqual(lv["response"], [1.0])
        f = functionality(1.0, units, results)["input_only"]
        self.assertEqual(f["units"]["F3"], 2)
        self.assertEqual(f["layers"]["F3"], 37.5)

    def test_moving_parts_are_worth_one_plus_ln_of_their_number(self):
        self.assertEqual([round(parts_points(n), 2) for n in (1, 2, 4, 88)], [1.0, 1.69, 2.39, 5.48])
        units = [unit(f"joint|piano__0|k{i}", "joint", "piano__0") for i in range(4)] + [
            unit("joint|cabinet__0|door", "joint", "cabinet__0"),
            unit("push|chair__0", "push", "chair__0"),
        ]
        results = {u["id"]: res(ok) for u, ok in zip(units, (True, True, True, False, True, True))}
        pts = level_points(1.0, units, results)["input_only"]
        self.assertEqual(pts["response"], [[1.0, 1.0]])
        piano = round(parts_points(4), 6)
        self.assertEqual(
            sorted(pts["affordance"]), sorted([[round(0.75 * piano, 6), piano], [1.0, 1.0]])
        )  # 3 of 4 keys, the cabinet's one door
        self.assertEqual(
            level_points(0.0, units, results)["input_only"]["response"], [[0.0, 1.0]]
        )  # a copy that fails the gate earns none
        group = dict(id="group|book|shelf__0", kind="group", slots=[f"book__{i}" for i in range(20)])
        self.assertEqual(
            level_points(1.0, [group], {group["id"]: res(True)})["input_only"]["balance"],
            [[round(1 + math.log(20), 6)] * 2],
        )  # 4.0


if __name__ == "__main__":
    unittest.main()
