"""Self-tests of the non-rigid verifiers (sbv/nonrigid). All but one need only requirements/core.txt: calibrate and
evaluate run with a table of readouts in place of Genesis. The slow one runs Genesis, with SBV_SLOW=1 and
SBV_NONRIGID_PYTHON pointing at a Python that has requirements/nonrigid.txt (about 2 minutes). Run from the repository
root:
    python -m unittest tests.test_nonrigid -v
"""

import json, math, os, shutil, subprocess, sys, tempfile, unittest
from unittest import mock
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from sbv.nonrigid import interior, room, schema, verify  # noqa: E402
from tests import rooms  # noqa: E402

COFFEE = dict(
    name="coffee__0",
    kind="fluid",
    on="cup__0",
    shape=dict(type="cylinder", pos=[1.3, 1.35, 0.79], radius=0.03, height=0.04),
    material=dict(viscosity="water", density=1000),
)
CUSHION = dict(
    name="cushion__0",
    kind="soft_body",
    on="chair__0",
    shape=dict(type="box", pos=[1.5, 0.62, 0.52], size=[0.4, 0.35, 0.07]),
    material=dict(youngs_kpa=40, poisson=0.3, density=50),
)
JACKET = dict(
    name="jacket__0",
    kind="thin_shell",
    on="cabinet__0",
    shape=dict(
        type="draped_sheet", rail=[3.4, 1.4, 0.9], rail_radius=0.01, width=0.5, front_length=0.6, back_length=0.6
    ),
    material=dict(fabric="wool_coat"),
)
CUBE_OBJ = "".join(f"v {x} {y} {z}\n" for x in (0, 1) for y in (0, 1) for z in (0, 1)) + "".join(
    f"f {a} {b} {c}\n"
    for a, b, c in [
        (1, 3, 2),
        (2, 3, 4),
        (5, 6, 7),
        (6, 8, 7),
        (1, 2, 5),
        (2, 6, 5),
        (3, 7, 4),
        (4, 7, 8),
        (1, 5, 3),
        (3, 5, 7),
        (2, 4, 6),
        (4, 8, 6),
    ]
)


HOLDERS = "".join(
    f'<body name="{n}" pos="{i} 0 0.5"><geom type="box" size="0.05 0.05 0.05"/></body>'
    for i, n in enumerate(("cup__0", "chair__0", "cabinet__0"))
)


def side(objs, files=None, **top):
    """A folder with nonrigid.json (and any extra files, and top-level keys such as seed), and a scene.xml naming the
    fixtures' holders and supports unless files gives its own; returns it."""
    d = tempfile.mkdtemp(prefix="sbv_soft_")
    with open(os.path.join(d, "nonrigid.json"), "w") as f:
        json.dump(dict(schema="nonrigid/v1", objects=objs, **top), f)
    if "scene.xml" not in (files or {}):
        with open(os.path.join(d, "scene.xml"), "w") as f:
            f.write(f"<mujoco><worldbody>{HOLDERS}</worldbody></mujoco>")
    for name, text in (files or {}).items():
        with open(os.path.join(d, name), "w") as f:
            f.write(text)
    return d


def moved(o, dx=0.0, **material):
    s = dict(o["shape"])
    s["rail" if "rail" in s else "pos"] = list(np.add(s.get("rail", s.get("pos")), [dx, 0, 0]))
    return dict(o, shape=s, material=dict(o["material"], **material))


class SideFile(unittest.TestCase):
    def test_a_good_file_has_no_problems(self):
        self.assertEqual(schema.load(os.path.join(side([COFFEE, CUSHION, JACKET]), "nonrigid.json"))[1], [])

    def test_each_bad_field_is_named(self):
        bad = [
            dict(COFFEE, name="coffee"),
            dict(CUSHION, kind="jelly"),
            dict(JACKET, on=""),
            moved(COFFEE, viscosity="honey") | dict(name="tea__0"),
            moved(JACKET, fabric="leather") | dict(name="coat__0"),
            moved(CUSHION, youngs_kpa=0) | dict(name="pad__0"),
        ]
        problems = schema.load(os.path.join(side(bad), "nonrigid.json"))[1]
        for word in ("'coffee'", "kind", "'on'", "viscosity", "fabric", "youngs_kpa"):
            self.assertTrue(any(word in p for p in problems), (word, problems))

    def test_a_mesh_sits_next_to_the_side_file_and_a_soft_mesh_is_closed(self):
        open_obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
        objs = [
            dict(CUSHION, shape=dict(type="mesh", file="../cube.obj", pos=[0, 0, 0])),
            dict(CUSHION, name="cushion__1", shape=dict(type="mesh", file="sheet.obj", pos=[0, 0, 0])),
            dict(CUSHION, name="cushion__2", shape=dict(type="mesh", file="cube.obj", pos=[0, 0, 0])),
        ]
        problems = schema.load(
            os.path.join(side(objs, {"cube.obj": CUBE_OBJ, "sheet.obj": open_obj}), "nonrigid.json")
        )[1]
        self.assertEqual(sorted(p.split(":")[0] for p in problems), ["cushion__0", "cushion__1"])

    def test_a_file_that_is_not_a_json_object_is_a_problem_not_a_crash(self):
        d = tempfile.mkdtemp()
        for text, word in (
            ('{"objects": [', "not valid JSON"),
            ("[1, 2]", "not an object"),
            ('{"objects": [3]}', "objects[0]"),
        ):
            with open(os.path.join(d, "nonrigid.json"), "w") as f:
                f.write(text)
            objs, problems = schema.load(os.path.join(d, "nonrigid.json"))
            self.assertEqual(objs, [])
            self.assertTrue(any(word in p for p in problems), (text, problems))

    def test_a_bad_entry_does_not_stop_the_package_tasks(self):
        from sbv.package_tasks.input_bundle import files

        objs = [
            dict(CUSHION, shape=dict(type="mesh", file="../cube.obj", pos=[0, 0, 0])),
            dict(CUSHION, name="cushion__2", shape=dict(type="mesh", file="cube.obj", pos=[0, 0, 0])),
        ]
        d = side(objs, {"cube.obj": CUBE_OBJ})
        self.assertEqual(sorted(files(d)), ["cube.obj", "nonrigid.json"])
        with open(os.path.join(d, "nonrigid.json"), "w") as f:
            f.write('{"objects": [')
        self.assertEqual(sorted(files(d)), ["nonrigid.json"])


class Pairing(unittest.TestCase):
    def test_nearest_of_the_same_kind_within_one_metre_each_used_once(self):
        tea = moved(COFFEE, 0.5) | dict(name="tea__0")
        copies = [
            moved(COFFEE, 0.45) | dict(name="cup_b__0"),
            moved(COFFEE, 0.1) | dict(name="cup_a__0"),
            moved(CUSHION) | dict(name="pad__0"),
        ]
        p = verify.pair([COFFEE, tea], copies)
        self.assertEqual((p["coffee__0"]["name"], p["tea__0"]["name"]), ("cup_a__0", "cup_b__0"))

    def test_a_declared_object_far_away_still_pairs(self):
        self.assertEqual(verify.pair([COFFEE], [moved(COFFEE, 3.0)])["coffee__0"]["_distance"], 3.0)

    def test_name_first_then_word_then_kind(self):
        far_same_name, near_same_word, nearest = (
            moved(COFFEE, 2.0),
            moved(COFFEE, 0.5) | dict(name="coffee__7"),
            moved(COFFEE, 0.1) | dict(name="tea__0"),
        )
        self.assertEqual(
            verify.pair([COFFEE], [nearest, near_same_word, far_same_name])["coffee__0"]["name"], "coffee__0"
        )
        self.assertEqual(verify.pair([COFFEE], [nearest, near_same_word])["coffee__0"]["name"], "coffee__7")
        self.assertEqual(verify.pair([COFFEE], [nearest])["coffee__0"]["name"], "tea__0")
        self.assertEqual(verify.pair([COFFEE], [moved(CUSHION) | dict(name="coffee__0")]), {})  # never across kinds,
        self.assertEqual(
            verify.pair([COFFEE], [moved(CUSHION) | dict(name="coffee__0")], same_kind=False)["coffee__0"]["kind"],
            "soft_body",
        )  # but for the kind check

    def test_word_and_kind_steps_follow_the_rigid_size_limits(self):
        big = dict(
            JACKET, name="jacket__1", shape=dict(JACKET["shape"], width=2.0, front_length=2.0, back_length=2.0)
        )  # diagonal 3.6 times
        towel = dict(JACKET, name="towel__0")
        self.assertEqual(verify.pair([towel], [moved(JACKET, 3.0)])["towel__0"]["name"], "jacket__0")  # any distance,
        self.assertEqual(verify.pair([towel], [moved(big, 0.1)]), {})  # within the place step's 3 times
        self.assertEqual(verify.pair([JACKET], [moved(big, 0.1)]), {})  # its own word, but not within 2 times
        self.assertEqual(
            verify.unpaired_why(JACKET, [big], []), "no thin_shell of its word or size left in the copy's nonrigid.json"
        )

    def test_the_rigid_numbers_kept_here_agree(self):
        from sbv import pair_gate
        from sbv.common import case
        from sbv.geometry import g2_shape

        self.assertEqual(
            (verify.POSE_FACTOR, verify.POSE_CAP, verify.SIZE_BARS, verify.RAMP, verify.SAME_SIZE, verify.PLACE_SIZE),
            (
                pair_gate.POSE_FACTOR,
                pair_gate.POSE_CAP,
                g2_shape.SIZE_BARS,
                g2_shape.RAMP,
                case.SAME_SIZE,
                case.PLACE_SIZE,
            ),
        )
        self.assertEqual(
            [verify.ramp(x, 0.1, 0.2) for x in (0.05, 0.13, 0.25)],
            [g2_shape.ramp(x, 0.1, 0.2) for x in (0.05, 0.13, 0.25)],
        )

    def test_why_a_true_object_is_unpaired(self):
        self.assertEqual(verify.unpaired_why(COFFEE, [], None), "no nonrigid.json in the copy")
        self.assertEqual(verify.unpaired_why(COFFEE, [CUSHION], []), "no fluid in the copy's nonrigid.json")
        self.assertEqual(
            verify.unpaired_why(COFFEE, [], [("coffee__0", "fluid")]),
            "no fluid in the copy's nonrigid.json (coffee__0 dropped for problems in the file)",
        )


class Safety(unittest.TestCase):
    def test_each_rule(self):
        bottom = 0.52 - 0.035
        cases = [
            (COFFEE, dict(error="over the 600 s compute cap"), "cap"),
            (COFFEE, dict(finite=False), "blew up"),
            (CUSHION, dict(ball_lowest=bottom + 0.004, object_lowest=bottom), "went through"),
            (CUSHION, dict(ball_lowest=bottom + 0.03, object_lowest=bottom - 0.03), "fell through"),
            (JACKET, dict(top_z=0.8, rail_z=0.9), "rail"),
        ]
        for obj, r, word in cases:
            self.assertTrue(any(word in b for b in verify.safety(obj, r)), (word, verify.safety(obj, r)))
        self.assertEqual(
            verify.safety(COFFEE, dict(held_at_rest=0.9, finite=True)), []
        )  # a leak is the containment verifier's, not a rule

    def test_containment_reads_the_rest_state_before_the_action(self):
        self.assertEqual(verify.contained(dict(held_at_rest=0.96, rest_speed=0.01, finite=True)), [])
        self.assertEqual(
            verify.contained(dict(held_at_rest=0.96, finite=True)), []
        )  # a run from before the speed readout
        for r, word in [
            (dict(held_at_rest=0.9, rest_speed=0.01, finite=True), "leaks"),
            (dict(held_at_rest=1.0, rest_speed=0.3, finite=True), "not at rest"),
            (dict(held_at_rest=1.0, finite=False), "blew up"),
            (dict(error="over the 600 s compute cap"), "cap"),
            (dict(finite=True), "no rest readout"),
        ]:
            self.assertTrue(any(word in b for b in verify.contained(r)), (word, verify.contained(r)))
        self.assertEqual(
            verify.contained(dict(held_at_rest=1.0, rest_speed=0.3, finite=True), rest_mps=0.5), []
        )  # a unit's own bar


READOUTS = {  # object -> run -> readout. The coffee decides; the cushion does not (half and double move it under 8 mm);
    # the jacket's lift does: denim (its class) reads within a factor of 2, cotton (the other) outside
    "coffee__0": {
        "base": 0.50,
        "repeat": 0.50,
        "repeat0": 0.51,
        "repeat1": 0.49,
        "nudge+": 0.56,
        "nudge-": 0.47,
        "half": 0.20,
        "double": 0.63,
    },
    "cushion__0": {
        "base": 0.030,
        "repeat": 0.030,
        "repeat0": 0.031,
        "repeat1": 0.029,
        "nudge+": 0.032,
        "nudge-": 0.029,
        "half": 0.035,
        "double": 0.027,
    },
    "jacket__0": {"base": 0.18, "repeat0": 0.18, "repeat1": 0.17, "same": 0.29, "other": 0.80},
}


class CalibrateAndScore(unittest.TestCase):
    """calibrate and evaluate end to end, with READOUTS in place of the Genesis runs."""

    def run_table(self, copy_readouts=None):
        seen = []

        def fake(scene_xml, obj, action, scale, work, video=None, threads=None, **kwargs):
            seen.append(threads)
            run = os.path.basename(work)
            r = READOUTS[obj["name"]][run] if run in READOUTS[obj["name"]] else copy_readouts[obj["name"]]
            return dict(
                readout=r,
                held_at_rest=1.0,
                ball_lowest=1.0,
                object_lowest=1.0,
                top_z=2.0,
                rail_z=0.9,
                finite=True,
                seconds=1.0,
            )

        return mock.patch.object(verify, "_one", side_effect=fake), seen

    def setUp(self):
        self.truth = side([COFFEE, CUSHION, JACKET])
        self.out = tempfile.mkdtemp(prefix="sbv_soft_case_")
        with self.run_table()[0]:
            verify.calibrate(self.truth, os.path.join(self.out, "nonrigid_case.json"), procs=2)
        with open(os.path.join(self.out, "nonrigid_case.json")) as f:
            self.case = json.load(f)
        self.units = {u["id"]: u for u in self.case["units"]}

    def tearDown(self):
        shutil.rmtree(self.truth, ignore_errors=True)
        shutil.rmtree(self.out, ignore_errors=True)

    def test_a_unit_decides_only_when_half_or_double_falls_outside_a_factor_of_2(self):
        u = self.units["S1.coffee__0"]
        self.assertEqual(
            (u["truth"], u["sensitivity"], u["rule"]), (0.50, dict(half=0.10, double=1.26), "factor2")
        )  # poured, of the true coffee
        self.assertEqual(u["readout"], "poured_of_true_amount")
        self.assertEqual([self.units[f"S1.{n}"]["decidable"] for n in ("coffee__0", "cushion__0")], [True, False])
        self.assertIn(
            "within a factor of 2", self.units["S1.cushion__0"]["not_decidable"]
        )  # 0.035 and 0.027 against 0.030
        u = self.units["S1.jacket__0"]  # the lift
        self.assertEqual(
            (u["decidable"], u["sensitivity"], u["action"]["force_n"]),
            (True, dict(same=0.29, other=0.80), round(0.5 * 9.81 * 0.8 * schema.sheet(JACKET).area, 4)),
        )

    def test_the_case_keeps_its_thread_count_and_names_its_room_relative_to_itself(self):
        self.assertEqual(self.case["threads"], verify.THREADS)
        self.assertEqual(os.path.normpath(os.path.join(self.out, self.case["room"])), os.path.normpath(self.truth))

    def test_a_copy_is_scored_unit_by_unit(self):
        self.case["threads"] = "3"
        with open(os.path.join(self.out, "nonrigid_case.json"), "w") as f:
            json.dump(self.case, f)
        copy = side(
            [
                dict(COFFEE, shape=dict(COFFEE["shape"], height=0.05)),  # 25% more coffee: within 30%
                dict(CUSHION, shape=dict(CUSHION["shape"], size=[0.6, 0.35, 0.07])),
            ]
        )  # 50% too long; no jacket
        patch, seen = self.run_table({"coffee__0": 0.60, "cushion__0": 0.031})
        with patch:
            rep = verify.evaluate(
                os.path.join(self.out, "nonrigid_case.json"), copy, os.path.join(copy, "nonrigid_report.json"), procs=2
            )
        rows = {r["id"]: r for r in rep["units"]}
        self.assertEqual(
            {k: r["score"] for k, r in rows.items()},
            {
                "S1.coffee__0": 1.0,
                "contain.coffee__0": 1.0,
                "S0.coffee__0": 1.0,
                "S1.cushion__0": None,
                "S0.cushion__0": 0.0,
                "N.coffee__0": 1.0,
                "H.coffee__0": 1.0,
                "N.cushion__0": 1.0,
                "N.jacket__0": 0.0,  # the jacket is not built
                "P0.cushion__0": 0.672446,
                "P0.jacket__0": 0.0,
                "S1.jacket__0": 0.0,
            },
        )  # 1.5 times as long: between the bars
        self.assertEqual(
            [x["stage"] for x in rows["N.coffee__0"]["stages"]], ["kind", "material", "behaviour"]
        )  # contain and spill fold into it
        self.assertEqual(
            [x["stage"] for x in rows["N.cushion__0"]["stages"]], ["kind", "material"]
        )  # its press cannot decide
        # 25% more coffee pours 60% of itself, 0.75 of the true amount against 0.50: within a factor of 2
        self.assertEqual(
            {r["id"]: r["kind"] for r in rep["units"]}["contain.coffee__0"], "contain"
        )  # every row says what it is, for the reward's parts
        self.assertEqual(
            (rep["nonrigid"], rep["not_counted"]),
            (66.7, ["S1.coffee__0", "contain.coffee__0", "S1.cushion__0", "S0.cushion__0", "S1.jacket__0"]),
        )
        # N, H, N, N, P0, P0, S0: (1 + 1 + 1 + 0 + 0.67 + 0 + 1) / 7; the cushion's size row folds into its pose
        self.assertEqual(
            (rows["S0.cushion__0"]["folded_into"], rows["P0.cushion__0"]["position_error"]), ("P0.cushion__0", 0.0)
        )
        self.assertEqual(seen, ["3"])  # scoring runs with the case's threads, and not the press that cannot decide

    def test_a_fluid_that_leaks_at_rest_fails_containment_and_a_leaking_true_room_is_refused(self):
        copy = side([COFFEE])
        patch, _ = self.run_table({"coffee__0": 0.50})
        with (
            patch,
            mock.patch.object(
                verify,
                "_one",
                side_effect=lambda *a, **k: dict(
                    readout=0.5, held_at_rest=0.8, rest_speed=0.01, finite=True, seconds=1.0
                ),
            ),
        ):
            rep = verify.evaluate(
                os.path.join(self.out, "nonrigid_case.json"), copy, os.path.join(copy, "nonrigid_report.json"), procs=2
            )
        rows = {r["id"]: r for r in rep["units"]}
        self.assertFalse(rows["contain.coffee__0"]["passed"])
        self.assertIn("leaks", rows["contain.coffee__0"]["why"])
        self.assertTrue(rows["S1.coffee__0"]["passed"])  # the spill is judged on its own number
        with mock.patch.object(
            verify,
            "_one",
            side_effect=lambda *a, **k: dict(readout=0.5, held_at_rest=1.0, rest_speed=0.4, finite=True, seconds=1.0),
        ):
            with self.assertRaises(SystemExit) as e:
                verify.calibrate(self.truth, os.path.join(self.out, "again.json"), procs=2)
        self.assertIn("not at rest", str(e.exception))

    def test_no_side_file_scores_zero(self):
        copy = tempfile.mkdtemp(prefix="sbv_soft_copy_")
        rep = verify.evaluate(
            os.path.join(self.out, "nonrigid_case.json"), copy, os.path.join(copy, "nonrigid_report.json")
        )
        self.assertEqual((rep["nonrigid"], rep["side_file_problems"]), (0.0, ["no nonrigid.json"]))


def inside(spec, v):
    """v is a value RANGES spec can draw."""
    if isinstance(spec, list) and not isinstance(spec[0], tuple):
        return v in spec
    if isinstance(spec, list):
        return len(v) == len(spec) and all(inside(a, b) for a, b in zip(spec, v))
    return spec[0] <= v <= spec[1]


class HiddenActions(unittest.TestCase):
    """Each reference object's action: drawn from the public RANGES by seed and name, or the reference room's own block;
    frozen in the case."""

    def calibrate(self, objs, seed=None, top=None):
        truth, out, runs = side(objs, **(top or {})), tempfile.mkdtemp(prefix="sbv_soft_case_"), {}
        self.addCleanup(shutil.rmtree, truth, True)
        self.addCleanup(shutil.rmtree, out, True)

        def fake(scene_xml, obj, action, scale, work, video=None, threads=None, **kwargs):
            runs[(obj["name"], os.path.basename(work))] = action
            return dict(
                readout=0.5,
                held_at_rest=1.0,
                ball_lowest=1.0,
                object_lowest=1.0,
                top_z=2.0,
                rail_z=0.9,
                finite=True,
                seconds=1.0,
            )

        with mock.patch.object(verify, "_one", side_effect=fake):
            verify.calibrate(truth, os.path.join(out, "nonrigid_case.json"), procs=2, seed=seed)
        with open(os.path.join(out, "nonrigid_case.json")) as f:
            case = json.load(f)
        return case, {u["object"]: u["action"] for u in case["units"] if "action" in u}, runs

    def test_drawn_actions_are_inside_the_public_ranges(self):
        case, acts, _ = self.calibrate([COFFEE, CUSHION, JACKET])  # no seed: a fresh one, kept in the case
        self.assertIsInstance(case["seed"], int)
        draws = [(o["kind"], acts[o["name"]]) for o in (COFFEE, CUSHION)] + [
            ("thin_shell", {k: v for k, v in acts["jacket__0"].items() if k != "force_n"})
        ]
        draws += [(k, verify.default_action(k, "thing__0", s)) for k in verify.RANGES for s in range(200)]
        for kind, a in draws:
            self.assertEqual(set(a), set(verify.RANGES[kind]))
            for k, v in a.items():
                self.assertTrue(inside(verify.RANGES[kind][k], v), (kind, k, v))

    def test_the_same_seed_gives_the_same_actions(self):
        acts = self.calibrate([COFFEE, CUSHION, JACKET], seed=7)[1]
        self.assertEqual(self.calibrate([COFFEE, CUSHION, JACKET], seed=7)[1], acts)
        self.assertEqual(
            self.calibrate([CUSHION, COFFEE], top=dict(seed=7))[1]["cushion__0"], acts["cushion__0"]
        )  # the side file's seed
        self.assertNotEqual(self.calibrate([COFFEE, CUSHION, JACKET], seed=8)[1], acts)

    def test_a_true_rooms_own_action_and_nudge_win_over_the_draw(self):
        own = dict(
            CUSHION,
            action=dict(offset=[0.0, 0.0], tool="block", block=[0.25, 0.25, 0.03], block_kg=10.0, height=0.03),
            nudge={"height": 0.004},
        )
        case, acts, runs = self.calibrate([COFFEE, own, JACKET], seed=7)
        drawn = verify.default_action("soft_body", "cushion__0", 7)
        self.assertEqual(acts["cushion__0"], {**drawn, **own["action"]})  # its keys win; the rest (duration) is drawn
        self.assertEqual(
            sorted(r for o, r in runs if o == "cushion__0"), ["base", "double", "half", "repeat0", "repeat1"]
        )  # no nudged runs; two repeats
        self.assertEqual(
            acts["coffee__0"], verify.default_action("fluid", "coffee__0", 7)
        )  # other objects keep their draw
        self.assertEqual(
            sorted(r for o, r in runs if o == "jacket__0"), ["base", "other", "repeat0", "repeat1", "same"]
        )  # its fabric swapped
        self.assertEqual(
            [
                round(runs[("jacket__0", r)]["t_start"] - acts["jacket__0"]["t_start"], 4)
                for r in ("repeat0", "repeat1")
            ],
            [0.1, 0.0],
        )
        self.assertFalse(any("action" in u["truth_object"] or "nudge" in u["truth_object"] for u in case["units"]))
        with self.assertRaises(SystemExit):
            self.calibrate([dict(CUSHION, nudge={"weight": 1.0})], seed=7)

    def test_an_existing_case_still_evaluates_with_its_own_actions_only(self):
        out = tempfile.mkdtemp(prefix="sbv_soft_old_")
        self.addCleanup(shutil.rmtree, out, True)
        old = dict(
            yaw_deg=0, angle_deg=70, t_start=0.5, t_ramp=0.5, duration=2.2
        )  # a case written before RANGES: no seed
        case = dict(
            schema="nonrigid_case/v0",
            room=out,
            threads="4",
            inputs={},
            units=[
                dict(
                    id="S1.coffee__0",
                    object="coffee__0",
                    kind="fluid",
                    action=old,
                    truth=0.5,
                    tolerance=0.1,
                    decidable=True,
                    sensitivity=dict(half=0.2, double=0.7),
                    seconds=1.0,
                    truth_object=COFFEE,
                ),
                dict(
                    id="S0.coffee__0",
                    object="coffee__0",
                    kind="amount",
                    truth=schema.volume_ml(COFFEE),
                    tolerance=0.3,
                    decidable=True,
                    truth_object=COFFEE,
                ),
            ],
        )
        with open(os.path.join(out, "nonrigid_case.json"), "w") as f:
            json.dump(case, f)
        copy, seen = side([dict(COFFEE, action=dict(angle_deg=5))]), []  # a copy's own "action" is ignored
        self.addCleanup(shutil.rmtree, copy, True)

        def fake(scene_xml, obj, action, scale, work, video=None, threads=None, **kwargs):
            seen.append(action)
            return dict(readout=0.55, held_at_rest=1.0, finite=True, seconds=1.0)

        with mock.patch.object(verify, "_one", side_effect=fake):
            rep = verify.evaluate(os.path.join(out, "nonrigid_case.json"), copy, os.path.join(copy, "r.json"))
        self.assertEqual(seen, [old])
        self.assertEqual(
            (rep["nonrigid"], [r["passed"] for r in rep["units"]]), (100.0, [True, True, True, True])
        )  # N, H, S1 (folded), S0


class RetiredSwing(unittest.TestCase):
    def test_an_older_cases_cloth_unit_does_not_count(self):  # S1.3
        out = tempfile.mkdtemp(prefix="sbv_soft_old_")
        self.addCleanup(shutil.rmtree, out, True)
        case = dict(
            schema="nonrigid_case/v0",
            room=out,
            threads="4",
            inputs={},
            units=[
                dict(
                    id="S1.jacket__0",
                    object="jacket__0",
                    kind="thin_shell",
                    action=dict(t_start=1.0, impulse=0.5, yaw_deg=0, duration=2.0),
                    truth=0.08,
                    tolerance=0.015,
                    decidable=True,
                    sensitivity=dict(half=0.12, double=0.05),
                    seconds=1.0,
                    truth_object=JACKET,
                )
            ],
        )
        with open(os.path.join(out, "nonrigid_case.json"), "w") as f:
            json.dump(case, f)
        copy, seen = side([JACKET]), []
        self.addCleanup(shutil.rmtree, copy, True)
        with mock.patch.object(
            verify,
            "_one",
            side_effect=lambda *a, **k: seen.append(1) or dict(readout=0.08, finite=True, top_z=2.0, rail_z=0.9),
        ):
            rep = verify.evaluate(os.path.join(out, "nonrigid_case.json"), copy, os.path.join(copy, "r.json"))
        rows = {r["id"]: r for r in rep["units"]}
        self.assertEqual((seen, rep["not_counted"]), ([], ["S1.jacket__0"]))  # not run, not counted
        self.assertIn("retired", rows["S1.jacket__0"]["why"])
        self.assertEqual(rows["N.jacket__0"]["score"], 1.0)  # the jacket's kind and material still count


class Lift(unittest.TestCase):
    def test_a_lift_unit_is_run_and_folds_into_the_cloths_unit(self):
        out = tempfile.mkdtemp(prefix="sbv_lift_")
        self.addCleanup(shutil.rmtree, out, True)
        unit = dict(
            id="S1.jacket__0",
            object="jacket__0",
            kind="thin_shell",
            action=dict(t_start=0.5, yaw_deg=0, duration=2.0, force_n=1.0),
            truth=0.18,
            tolerance=0.01,
            rule="factor2",
            decidable=True,
            sensitivity=dict(same=0.29, other=0.8),
            seconds=1.0,
            truth_object=JACKET,
        )
        with open(os.path.join(out, "nonrigid_case.json"), "w") as f:
            json.dump(dict(schema="nonrigid_case/v1", room=out, threads="4", inputs={}, units=[unit]), f)
        for copy_obj, rise, score in (
            (JACKET, 0.3, 1.0),
            (JACKET, 0.8, 2 / 3),
            (dict(JACKET, material=dict(fabric="cotton")), None, 1 / 3),
        ):
            copy, seen = side([copy_obj]), []
            self.addCleanup(shutil.rmtree, copy, True)
            with mock.patch.object(
                verify,
                "_one",
                side_effect=lambda *a, **k: seen.append(a[2])
                or dict(readout=rise, finite=True, top_z=2.0, rail_z=0.9, seconds=1.0),
            ):
                rep = verify.evaluate(os.path.join(out, "nonrigid_case.json"), copy, os.path.join(copy, "r.json"))
            rows = {r["id"]: r for r in rep["units"]}
            self.assertAlmostEqual(
                rows["N.jacket__0"]["score"], score
            )  # a cotton copy fails its material and is not run
            self.assertEqual((seen, rows["S1.jacket__0"]["counted"]), ([unit["action"]] if rise else [], False))
        unsettled = dict(error="cloth is not at rest", start_speed_mps=1.51)  # SX.7 fails the behaviour, not the object
        copy = side([JACKET])
        self.addCleanup(shutil.rmtree, copy, True)
        with mock.patch.object(verify, "_one", return_value=unsettled):
            rows = {
                r["id"]: r
                for r in verify.evaluate(os.path.join(out, "nonrigid_case.json"), copy, os.path.join(copy, "r.json"))[
                    "units"
                ]
            }
        self.assertEqual((rows["N.jacket__0"]["score"], rows["S1.jacket__0"]["why"]), (2 / 3, "cloth is not at rest"))
        self.assertIn(
            "cloth is not at rest", verify.readout_problems(JACKET, unsettled, out)
        )  # a true run like it still cannot calibrate
        settled = dict(
            readout=0.3, finite=True, top_z=2.0, rail_z=0.9, start_speed_mps=0.02
        )  # every cloth run reports its start speed
        self.assertEqual(
            verify.readout_problems(JACKET, settled, out), []
        )  # a run at rest with no error calibrates (was a KeyError)


class Threshold(unittest.TestCase):
    """A hidden action passes when the copy's readout is within a factor of 2 of the truth, or within the floor of
    it."""

    def test_a_factor_of_2_either_way_or_the_floor(self):
        w = lambda r, t=0.20, k="fluid": verify.within(r, t, k)
        self.assertEqual([w(r) for r in (0.20, 0.10, 0.40, 0.09, 0.41)], [True, True, True, False, False])
        self.assertTrue(w(0.07, 0.0))
        self.assertFalse(w(0.09, 0.0))  # near nothing, the 8-point floor decides
        self.assertTrue(w(0.004, 0.010, "soft_body"))  # 2.5 times less, but within 8 mm


class RoomCut(unittest.TestCase):
    """The MuJoCo side of a run: the part of the room kept around an object, and a holder's real inside."""

    @classmethod
    def setUpClass(cls):
        cls.xml = rooms.write(tempfile.mkdtemp(prefix="sbv_soft_room_"))

    def test_the_cut_keeps_what_is_near_and_holds_it_still(self):
        keep, _ = room.neighbourhood(self.xml, COFFEE)
        self.assertIn("table__0", keep)
        self.assertIn("cup__0", keep)
        self.assertNotIn("cabinet__0", keep)
        files = room.cut(self.xml, keep, "cup__0", tempfile.mkdtemp(prefix="sbv_soft_cut_"))
        with open(files["still"]) as f, open(files["actor"]) as g:
            still, actor = f.read(), g.read()
        self.assertNotIn("freejoint", still)
        self.assertNotIn('name="cup__0"', still)
        self.assertIn("freejoint", actor)

    def test_a_cups_real_inside_holds_its_middle_not_its_outside(self):
        held = interior.held_test(*interior.inside(self.xml, "cup__0", [1.3, 1.35, 0.75]))
        self.assertEqual(
            held(np.array([[1.3, 1.35, 0.79], [1.3, 1.45, 0.79], [1.3, 1.35, 0.9]])).tolist(), [True, False, False]
        )


def genesis_ok():
    try:
        return subprocess.run([verify.PYTHON, "-c", "import genesis"], capture_output=True, timeout=120).returncode == 0
    except Exception:
        return False


@unittest.skipUnless(
    os.environ.get("SBV_SLOW") == "1" and genesis_ok(),
    "slow: set SBV_SLOW=1 and SBV_NONRIGID_PYTHON to a Python with requirements/nonrigid.txt",
)
class GenesisRun(unittest.TestCase):
    def test_a_ball_dropped_on_a_soft_block_presses_it_in(self):
        d = tempfile.mkdtemp(prefix="sbv_soft_slow_")
        xml = rooms.write(d)
        block = dict(
            name="cushion__0",
            kind="soft_body",
            on="floor",
            shape=dict(type="box", pos=[2.6, 0.6, 0.05], size=[0.2, 0.2, 0.1]),
            material=dict(youngs_kpa=40, poisson=0.3, density=50),
        )
        r = verify._one(xml, block, verify.default_action("soft_body", block["name"], 0), None, os.path.join(d, "run"))
        self.assertNotIn("error", r, r)
        self.assertEqual(verify.safety(block, r, d), [])
        self.assertTrue(
            0.005 < r["readout"] < 0.1, r["readout"]
        )  # a 1 kg ball into 40 kPa foam: millimetres to centimetres
        shutil.rmtree(d, ignore_errors=True)


class FluidAmount(unittest.TestCase):
    def test_a_factor_of_2_either_way(self):  # before, within 30%
        from sbv.nonrigid.verify import amount_ok

        self.assertEqual(
            [amount_ok(v, 300.0) for v in (150.0, 180.0, 540.0, 600.0, 140.0, 900.0)],
            [True, True, True, True, False, False],
        )


if __name__ == "__main__":
    unittest.main()


class KindAndMaterial(unittest.TestCase):
    def test_stiffness_band_follows_the_true_objects_own_sensitivity(self):
        u = dict(truth=0.0328, sensitivity=dict(half=0.0455, double=0.0214), tolerance=0.0003)  # parcel bag__0
        lo, hi = verify.stiffness_band(u)
        self.assertAlmostEqual(
            lo, min(verify.MATERIAL_MAX, 2 ** (math.log(2) / math.log(0.0455 / 0.0328))), places=6
        )  # each side its own slope
        self.assertAlmostEqual(hi, 2 ** (math.log(2) / math.log(0.0328 / 0.0214)), places=6)
        self.assertEqual(
            verify.stiffness_band(dict(truth=0.0034, sensitivity=dict(half=0.026, double=0.0001), tolerance=0)),
            [verify.BAND, verify.BAND],
        )  # steep: never under BAND
        flat = dict(truth=0.02, sensitivity=dict(half=0.02, double=0.02), tolerance=0.0)
        self.assertEqual(verify.stiffness_band(flat), [verify.MATERIAL_MAX] * 2)
        self.assertEqual(
            verify.stiffness_band(dict(flat, sensitivity=dict(half=0.0200001, double=0.02))), [verify.MATERIAL_MAX] * 2
        )  # nearly flat agree
        self.assertEqual(
            verify.stiffness_band(dict(truth=0.02, sensitivity=dict(half=0.01, double=0.03), tolerance=0)),
            [verify.BAND] * 2,
        )  # backwards: BAND
        self.assertEqual(verify.stiffness_band(None), [verify.BAND, verify.BAND])

    def test_material_per_kind(self):
        ok = lambda t, c, b=verify.BAND: verify.material_check(t, c, b)[0]
        self.assertTrue(ok(CUSHION, moved(CUSHION, youngs_kpa=CUSHION["material"]["youngs_kpa"] * 1.9)))
        self.assertFalse(
            ok(CUSHION, moved(CUSHION, youngs_kpa=CUSHION["material"]["youngs_kpa"] / 4))
        )  # four times too nonrigid
        self.assertTrue(
            ok(CUSHION, moved(CUSHION, youngs_kpa=CUSHION["material"]["youngs_kpa"] / 3), 3.5)
        )  # inside a wider band
        self.assertFalse(ok(CUSHION, moved(CUSHION, density=CUSHION["material"]["density"] * 2.5)))
        self.assertFalse(
            ok(JACKET, dict(JACKET, material=dict(JACKET["material"], fabric="silk")))
        )  # wool coat is heavy, silk light
        self.assertTrue(ok(JACKET, dict(JACKET, material=dict(JACKET["material"], fabric="denim"))))  # both heavy
        self.assertFalse(ok(COFFEE, dict(COFFEE, material=dict(COFFEE["material"], viscosity="honey"))))

    def test_relative_compute_cap(self):
        self.assertEqual(verify.copy_cap(dict(seconds=311.4)), 3 * 311.4)
        self.assertEqual(verify.copy_cap(dict(seconds=50)), verify.CAP_S)

    def test_repeatability_replaces_the_fixed_floor(self):
        steady = dict(base=0.0034, half=0.026, double=0.0001, repeat0=0.0036, repeat1=0.0033)
        noisy = dict(steady, repeat0=0.0015, repeat1=0.048)

        def cal(table):
            truth, out = side([CUSHION]), tempfile.mkdtemp(prefix="sbv_soft_case_")
            self.addCleanup(shutil.rmtree, truth, True)
            self.addCleanup(shutil.rmtree, out, True)
            fake = lambda xml, obj, action, scale, work, *a, **k: dict(
                readout=table[os.path.basename(work)], ball_lowest=1.0, object_lowest=1.0, finite=True, seconds=1.0
            )
            with mock.patch.object(verify, "_one", side_effect=fake):
                verify.calibrate(truth, os.path.join(out, "c.json"), procs=1)
            return next(u for u in json.load(open(os.path.join(out, "c.json")))["units"] if u["id"] == "S1.cushion__0")

        u = cal(steady)
        self.assertTrue(u["decidable"])
        self.assertAlmostEqual(u["tolerance"], 0.0002)
        self.assertFalse(
            verify.within(0.0001, u["truth"], "soft_body", u["tolerance"])
        )  # a stiff copy no longer passes on the floor
        self.assertTrue(verify.within(0.0001, u["truth"], "soft_body"))  # as it did with the 8 mm FLOOR
        u = cal(noisy)
        self.assertFalse(u["decidable"])
        self.assertIn("does not repeat", u["not_decidable"])

    def test_every_run_is_decided(self):
        u = dict(kind="fluid", truth_object=COFFEE)
        d = side([COFFEE])
        self.addCleanup(shutil.rmtree, d, True)
        self.assertIsNone(verify.oversized(u, COFFEE, d, d))
        big = dict(COFFEE, shape=dict(COFFEE["shape"], height=COFFEE["shape"]["height"] * 5))
        self.assertIn("5.0 times the true volume", verify.oversized(u, big, d, d))

    def test_a_declaration_that_cannot_be_built_earns_nothing(self):
        bodies = {"cabinet__0", "cup__0", "floor"}
        self.assertIsNone(verify.unbuildable(JACKET, bodies))
        self.assertIn(
            "cannot be a box",
            verify.unbuildable(dict(JACKET, shape=dict(type="box", pos=[0, 0, 0], size=[1, 1, 1])), bodies),
        )
        self.assertIn("neither the floor", verify.unbuildable(dict(COFFEE, on="cup__999"), bodies))
        self.assertIn(
            "no front_length",
            verify.unbuildable(
                dict(JACKET, shape={k: v for k, v in JACKET["shape"].items() if k != "front_length"}), bodies
            ),
        )  # review round 2
        self.assertIsNone(verify.unbuildable(dict(CUSHION, on="floor"), bodies))  # the builder's floor convention
        d = side([], {"sheet.obj": CUBE_OBJ})
        self.addCleanup(shutil.rmtree, d, True)  # a cloth mesh in room coordinates needs no pos
        sheet = dict(JACKET, shape=dict(type="mesh", file="sheet.obj", rail=[0.5, 0.5, 1.0]))
        self.assertIsNone(verify.unbuildable(sheet, bodies, d))
        self.assertIn(
            "pos is not three finite numbers",
            verify.unbuildable(dict(sheet, shape=dict(sheet["shape"], pos=[0, 0])), bodies, d),
        )
        d = side([])
        self.addCleanup(shutil.rmtree, d, True)
        self.assertEqual(
            verify.scene_supports(os.path.join(d, "scene.xml")), {"floor", "cup__0", "chair__0", "cabinet__0"}
        )

    def test_one_correspondence_for_every_check(self):
        t1, t2 = (
            dict(JACKET, name="towel__1", material=dict(fabric="cotton")),
            dict(JACKET, name="towel__2", material=dict(fabric="cotton")),
        )
        copies = [dict(CUSHION, name="towel__1"), dict(t2), dict(t1, name="towel__3", material=dict(fabric="silk"))]
        one = verify.pair([t1, t2], copies, same_kind=False)
        self.assertEqual(
            one["towel__1"]["name"], "towel__1"
        )  # the exact name, whatever its kind: kind then fails, once
        self.assertEqual(
            verify.pair([CUSHION], [dict(CUSHION, name="pillow__1")], same_kind=False)["cushion__0"]["name"],
            "pillow__1",
        )  # then the same kind

    def test_a_broken_run_scores_the_object_zero_and_a_failed_material_is_not_run(self):
        """review round 2: a run that blew up kept 2/3 for kind and material; a run whose material already failed
        cost simulation for no credit."""
        d = side([CUSHION])
        self.addCleanup(shutil.rmtree, d, True)
        case = dict(units=[dict(id="S1.cushion__0", object="cushion__0", kind="soft_body", decidable=True)])
        rows = [
            dict(
                id="S1.cushion__0",
                object="cushion__0",
                kind="soft_body",
                decidable=True,
                counted=True,
                passed=False,
                safety=["blew up"],
                why="blew up",
            )
        ]
        sup = verify.scene_supports(os.path.join(d, "scene.xml"))
        out = verify.object_units(case, [CUSHION], {"cushion__0": CUSHION}, sup, rows, [CUSHION], [], d, d)
        self.assertEqual((out[0]["score"], out[0]["why"]), (0.0, "broken run rule: blew up"))
        nonrigid = dict(CUSHION, material=dict(CUSHION["material"], youngs_kpa=CUSHION["material"]["youngs_kpa"] / 10))
        out = verify.object_units(case, [CUSHION], {"cushion__0": nonrigid}, sup, [], [nonrigid], [], d, d)
        self.assertEqual(
            ([x["stage"] for x in out[0]["stages"]], out[0]["score"]), (["kind", "material"], 0.5)
        )  # material fails: no behaviour stage counted

    def test_round_three_cases(self):
        """review round 3: an unmeasurable shape crashed pairing; a scalar rail passed; a skipped run zeroed the
        unit."""
        broken = dict(CUSHION)
        broken["shape"] = {k: v for k, v in CUSHION["shape"].items() if k != "size"}
        self.assertEqual(verify.pair([CUSHION], [broken])["cushion__0"]["_distance"], float("inf"))  # pairs, no crash
        self.assertIn("no size", verify.unbuildable(broken, {"chair__0", "floor"}))
        self.assertIn(
            "three finite numbers",
            verify.unbuildable(dict(JACKET, shape=dict(JACKET["shape"], rail=1)), {"cabinet__0"}),
        )
        d = side([CUSHION])
        self.addCleanup(shutil.rmtree, d, True)
        case = dict(units=[dict(id="S1.cushion__0", object="cushion__0", kind="soft_body", decidable=True)])
        rows = [
            dict(
                id="S1.cushion__0",
                object="cushion__0",
                kind="soft_body",
                decidable=True,
                counted=True,
                passed=False,
                skipped=True,
                why="not run",
            )
        ]
        nonrigid = dict(CUSHION, material=dict(CUSHION["material"], youngs_kpa=CUSHION["material"]["youngs_kpa"] / 10))
        out = verify.object_units(
            case,
            [CUSHION],
            {"cushion__0": nonrigid},
            verify.scene_supports(os.path.join(d, "scene.xml")),
            rows,
            [nonrigid],
            [],
            d,
            d,
        )
        self.assertEqual(
            (out[0]["score"], out[0]["evidence"]), (1 / 3, "static")
        )  # kind passes; skipped is not broken; nothing ran

    def test_round_four_cases(self):
        """review round 4: non-text names crashed grading; an infinite length passed; a malformed cloth rail crashed
        load()."""
        self.assertIn("not a name", verify.unbuildable(dict(CUSHION, on=["chair__0"]), {"chair__0"}))
        self.assertIn(
            "not a name", verify.unbuildable(dict(CUSHION, shape=dict(CUSHION["shape"], type=["box"])), {"chair__0"})
        )
        self.assertIn(
            "finite", verify.unbuildable(dict(JACKET, shape=dict(JACKET["shape"], front_length=1e309)), {"cabinet__0"})
        )
        d = side(
            [dict(JACKET, shape=dict(type="mesh", file="cube.obj", pos=[0, 0, 1], rail=1))], {"cube.obj": CUBE_OBJ}
        )
        self.addCleanup(shutil.rmtree, d, True)
        objs, problems = schema.load(os.path.join(d, "nonrigid.json"))
        self.assertTrue(any("rail" in p for p in problems))  # a problem, not a crash

    def test_round_five_cases(self):
        """review round 5: a dropped entry with a numeric name crashed the reason; an infinite yaw passed."""
        self.assertIn("123", verify.unpaired_why(COFFEE, [], [(123, "fluid")]))
        self.assertIn(
            "yaw_deg", verify.unbuildable(dict(CUSHION, shape=dict(CUSHION["shape"], yaw_deg=1e309)), {"chair__0"})
        )

    def test_round_six_cases(self):
        """review round 6: an unbuildable material (poisson 'invalid') earned the unit; non-finite bounds and legacy
        ran."""
        self.assertIn(
            "Poisson",
            verify.unbuildable(dict(CUSHION, material=dict(CUSHION["material"], poisson="invalid")), {"chair__0"}),
        )
        self.assertIn(
            "Poisson", verify.unbuildable(dict(CUSHION, material=dict(CUSHION["material"], poisson=0.5)), {"chair__0"})
        )
        self.assertIn(
            "viscosity",
            verify.unbuildable(dict(COFFEE, material=dict(COFFEE["material"], density=float("nan"))), {"cup__0"}),
        )
        with mock.patch.object(verify, "bounds", return_value=(np.array([0, 0, np.nan]), np.ones(3))):
            self.assertIn("not finite", verify.unbuildable(CUSHION, {"chair__0"}))

    def test_huge_integers_are_rejected_not_crashes(self):
        """review round 7: np.isfinite raised on an integer beyond float range."""
        big = 10**400
        self.assertIn(
            "Poisson", verify.unbuildable(dict(CUSHION, material=dict(CUSHION["material"], poisson=big)), {"chair__0"})
        )
        self.assertIn(
            "Poisson", verify.unbuildable(dict(CUSHION, material=dict(CUSHION["material"], density=big)), {"chair__0"})
        )
        self.assertIn(
            "yaw_deg", verify.unbuildable(dict(CUSHION, shape=dict(CUSHION["shape"], yaw_deg=big)), {"chair__0"})
        )
        self.assertIn(
            "three finite numbers",
            verify.unbuildable(dict(CUSHION, shape=dict(CUSHION["shape"], pos=[big, 0, 0])), {"chair__0"}),
        )

    def test_round_eight_cases(self):
        """review round 8: overflowing numbers, a null material, a list viscosity and a stray file key crashed;
        ring clips on a support without a rod earned credit."""
        d = side(
            [
                dict(COFFEE, shape=dict(COFFEE["shape"], radius=1e200)),
                dict(CUSHION, material=None),
                dict(COFFEE, name="tea__0", material=dict(viscosity=[])),
                dict(CUSHION, name="pad__0", shape=dict(CUSHION["shape"], size=[10**100, 1, 1])),
            ]
        )
        self.addCleanup(shutil.rmtree, d, True)
        objs, problems = schema.load(os.path.join(d, "nonrigid.json"))
        self.assertEqual(objs, [])
        self.assertEqual(len(problems), 4)  # four problems, no crash
        self.assertIsNone(
            verify.unbuildable(dict(CUSHION, shape=dict(CUSHION["shape"], file=123)), {"chair__0"})
        )  # a stray key on a box is ignored
        ring = dict(JACKET, attachments=dict(type="sliding_ring_clips", count=2, axis=[1, 0, 0]))
        self.assertIn("rod", verify.unbuildable(ring, {"cabinet__0"}, d))  # the cabinet is a box: no rod

    def test_round_nine_cases(self):
        """review round 9: a nameless entry crashed pairing; deep nesting crashed the guard; a clip axis across its rod
        passed."""
        deep = []
        for _ in range(500):
            deep = [deep]  # decodes, then the guard must not recurse out
        d = side(
            [
                {k: v for k, v in CUSHION.items() if k != "name"},
                dict(CUSHION, name="pad__0", shape=dict(CUSHION["shape"], extra=deep)),
            ]
        )
        self.addCleanup(shutil.rmtree, d, True)
        objs, problems = schema.load(os.path.join(d, "nonrigid.json"))
        self.assertEqual((objs, len(problems)), ([], 2))
        rodroom = side(
            [],
            {
                "scene.xml": '<mujoco><worldbody><body name="rail__0" pos="0 0 1"><geom type="capsule" fromto="-0.5 0 0 0.5 0 0" size="0.01"/></body></worldbody></mujoco>'
            },
        )
        self.addCleanup(shutil.rmtree, rodroom, True)
        ring = dict(
            JACKET,
            on="rail__0",
            shape=dict(JACKET["shape"], rail=[0, 0, 1]),
            attachments=dict(type="sliding_ring_clips", count=2, axis=[0, 1, 0]),
        )
        self.assertIn("not parallel", verify.unbuildable(ring, {"rail__0"}, rodroom))
        self.assertIsNone(
            verify.unbuildable(dict(ring, attachments=dict(ring["attachments"], axis=[1, 0, 0])), {"rail__0"}, rodroom)
        )
        with open(os.path.join(d, "nonrigid.json"), "w") as f:
            f.write('{"objects": ' + "[" * 100000 + "]" * 100000 + "}")
        self.assertIn(
            "not valid JSON", schema.load(os.path.join(d, "nonrigid.json"))[1][0]
        )  # too deep to decode: a problem

    def test_round_ten_cases(self):
        """review round 10: null attachments, and clips whose header is far from the rod, earned cloth credit."""
        self.assertIn("not an object", verify.unbuildable(dict(JACKET, attachments=None), {"cabinet__0"}))
        rodroom = side(
            [],
            {
                "scene.xml": '<mujoco><worldbody><body name="rail__0" pos="0 0 1"><geom type="capsule" fromto="-0.5 0 0 0.5 0 0" size="0.01"/></body></worldbody></mujoco>'
            },
        )
        self.addCleanup(shutil.rmtree, rodroom, True)
        clips = dict(type="sliding_ring_clips", count=2, axis=[1, 0, 0])
        far = dict(JACKET, on="rail__0", shape=dict(JACKET["shape"], rail=[0, 1, 1]), attachments=clips)
        self.assertIn("8 cm", verify.unbuildable(far, {"rail__0"}, rodroom))
        self.assertIsNone(
            verify.unbuildable(dict(far, shape=dict(JACKET["shape"], rail=[0, 0, 1])), {"rail__0"}, rodroom)
        )
        wide = dict(
            far, shape=dict(JACKET["shape"], rail=[0, 0, 1], width=2.0)
        )  # review round 11: a header wider than the rod
        self.assertIn("8 cm", verify.unbuildable(wide, {"rail__0"}, rodroom))
        high = dict(
            far,
            shape=dict(JACKET["shape"], rail=[0, 0, 1], width=0.5, rail_radius=0.2, front_length=0.4, back_length=0.4),
        )  # review round 12
        self.assertIn(
            "8 cm", verify.unbuildable(high, {"rail__0"}, rodroom)
        )  # the drape's own header sits 0.2 m above the rod
        self.assertIn(
            "no area", verify.unbuildable(dict(JACKET, shape=dict(JACKET["shape"], width=0.01)), {"cabinet__0"})
        )  # review round 13
        self.assertIn(
            "holder body", verify.unbuildable(dict(COFFEE, on="floor"), {"floor", "cup__0"})
        )  # review round 14

    def test_pose_is_the_rigid_rule_on_the_declared_rest_shape(self):
        long_front = dict(JACKET, shape=dict(JACKET["shape"], front_length=0.9, back_length=0.1))
        self.assertAlmostEqual(schema.bounds(long_front)[0][2], 0.0)  # the longer panel hangs lowest
        d = side([])
        pose = lambda c: verify.pose_unit(JACKET, c, d, d)
        self.assertEqual(
            (pose(JACKET)["score"], pose(moved(JACKET, 0.15))["passed"], pose(None)["why"]), (1.0, True, "not declared")
        )
        # bar: a quarter of the 0.79 m diagonal, 0.197 m; credit falls to 0 at twice that
        self.assertEqual((pose(moved(JACKET, 0.3))["score"], pose(moved(JACKET, 0.4))["score"]), (0.467134, 0.0))
        as_box = dict(
            CUSHION, name="jacket__0", shape=dict(type="box", pos=list(schema.center(JACKET)), size=[0.5, 0.02, 0.61])
        )
        self.assertEqual(pose(as_box)["score"], 1.0)  # built as a soft body: its place still counts here
        thick = dict(as_box, shape=dict(as_box["shape"], size=[0.5, 0.2, 0.61]))
        self.assertEqual(pose(thick)["size_ratio"], 1.0)  # 20 cm against 2: a thin shell's third side is not compared
        from sbv.reward import nonrigid_part

        self.assertEqual(nonrigid_part(pose(JACKET)), "geometry")

    def test_loose_world_geometry_is_no_support(self):
        """review round 15: a box straight under <worldbody> came back as "world", and a fluid held there was
        buildable."""
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        with open(os.path.join(d, "scene.xml"), "w") as f:
            f.write(
                '<mujoco><worldbody><geom type="box" size="1 1 0.1"/><body name="cup__0"><geom type="box" size="0.05 0.05 0.05"/></body></worldbody></mujoco>'
            )
        sup = verify.scene_supports(os.path.join(d, "scene.xml"))
        self.assertEqual(sup, {"floor", "cup__0"})
        self.assertIn("neither the floor", verify.unbuildable(dict(COFFEE, on="world"), sup))
