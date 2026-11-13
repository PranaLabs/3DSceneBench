"""End-to-end checks on a primitive test room: the reference scores 100 against itself, and each deliberately broken
copy fails exactly the tests built for its mistake. Plus the doc's own worked examples. Run from this folder:
    python -m unittest tests.test_verifiers -v
"""

import copy, importlib.util, json, os, re, shutil, struct, sys, tempfile, time, unittest, zlib
import numpy as np
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from sbv import calibrate, evaluate, perturb  # noqa: E402
from sbv.common import case as K  # noqa: E402
from sbv.geometry import g1_inventory as Inv  # noqa: E402
from sbv.functionality.aggregate import functionality  # noqa: E402
from tests import rooms  # noqa: E402


class DocExamples(unittest.TestCase):
    def test_inventory_f1(self):  # three apples, three cups against exactly three apples and two cups
        scene = SimpleNamespace(
            objects={f"o{i}": i for i in range(6)},
            cat={**{f"o{i}": "apple" for i in range(3)}, **{f"o{i}": "cup" for i in range(3, 6)}},
        )
        slots = [dict(slot=f"t{i}", category=c) for i, c in enumerate(["apple"] * 3 + ["cup"] * 2)]
        corr = dict(
            t0="o0", t1="o1", t2="o2", t3="o3", t4="o4"
        )  # what correspond() pairs: every reference object found
        r = Inv.inventory(dict(required=dict(apple=3, cup=2)), scene, corr, slots)
        self.assertAlmostEqual(r["f1"], 10 / 11)
        self.assertAlmostEqual(r["precision"], 5 / 6)
        self.assertAlmostEqual(r["recall"], 1.0)
        c = Inv.counts([dict(category="apple", op="exactly", n=3), dict(category="cup", op="exactly", n=2)], scene)
        self.assertAlmostEqual(c["score"], 0.5)

    def test_functionality_aggregation(
        self,
    ):  # 3/4 F2 bundles, 2/3 F3 functions, 1/2 F4 tasks -> 63.89; F1 failure -> 0
        units = [dict(id=f"{k}{i}", kind=k) for k, n in (("push", 4), ("container", 3), ("robot", 2)) for i in range(n)]
        ok = {"push0", "push1", "push2", "container0", "container1", "robot0"}
        res = {u["id"]: [dict(passed={"physics_hint": u["id"] in ok, "input_only": u["id"] in ok})] for u in units}
        self.assertAlmostEqual(functionality(1.0, units, res)["physics_hint"]["functionality"], 63.89, places=2)
        self.assertEqual(functionality(0.0, units, res)["physics_hint"]["functionality"], 0.0)

    def test_functionality_at_the_reference_bar(self):  # its own pass decisions, and its own hint-only marks
        units = [
            dict(id="p0", kind="push", hint_only=False, hint_only_reference_bar=True),
            dict(id="p1", kind="push", hint_only=False),
        ]
        res = dict(
            p0=[
                dict(
                    passed=dict(physics_hint=True, input_only=True),
                    passed_reference_bar=dict(physics_hint=False, input_only=False),
                )
            ],
            p1=[dict(passed=dict(physics_hint=True, input_only=True))],
        )  # no reference-bar entry: same as main
        main, ref = functionality(1.0, units, res), functionality(1.0, units, res, bar=1)
        self.assertEqual([main[s]["functionality"] for s in ("physics_hint", "input_only")], [100.0, 100.0])
        self.assertEqual([ref[s]["functionality"] for s in ("physics_hint", "input_only")], [50.0, 100.0])

    def test_general_and_declared_groups(self):  # declared = named by the reference room (functions.json, tasks.json)
        units = [
            dict(id="push0", kind="push", slots=["book__0"]),
            dict(id="push1", kind="push", slots=["mug__0"]),
            dict(id="opening0", kind="opening", slots=["cup__0"], named=True),
            dict(id="robot0", kind="robot", slots=["mug__0", "shelf__0"], named=True),
        ]
        ok = {"push0", "opening0"}
        res = {u["id"]: [dict(passed={"physics_hint": u["id"] in ok, "input_only": u["id"] in ok})] for u in units}
        f = functionality(1.0, units, res, slots=["book__0", "mug__0", "cup__0", "shelf__0", "lamp__0"])
        self.assertEqual(
            f["physics_hint"]["groups"], dict(general=dict(score=50.0, units=2), declared=dict(score=50.0, units=2))
        )
        self.assertEqual(f["declared_coverage"], dict(objects=5, covered=3, names=["cup__0", "mug__0", "shelf__0"]))
        self.assertAlmostEqual(
            f["physics_hint"]["functionality"], functionality(1.0, units, res)["physics_hint"]["functionality"]
        )  # headline unchanged


def edited_room(edit):
    """The test room written to a temporary folder, with its MJCF text passed through edit(); returns the scene path."""
    path = rooms.write(tempfile.mkdtemp(prefix="sbv_edit_"))
    with open(path) as f:
        xml = f.read()
    with open(path, "w") as f:
        f.write(edit(xml))
    return path


class Robustness(unittest.TestCase):
    """Regression checks for problems found in review: each one used to fail or crash."""

    def test_names_with_double_underscore_give_their_exact_category(self):
        from sbv.common.scene import category

        self.assertEqual(category("cup_holder__0", ["cup"]), "cup_holder")
        self.assertEqual(category("chair_left", ["chair"]), "chair")

    def test_wall_planes_and_geomless_bodies_pass_the_gate(self):
        from sbv.functionality.f1_gate import gate
        from sbv.common.scene import PHYSICS, Scene

        xml = edited_room(
            lambda x: x.replace(
                "</worldbody>",
                '<geom name="wall_x" type="plane" zaxis="1 0 0" pos="0.01 1.5 1" size="2 2 0.1"/>'
                '<body name="camera_rig__0" pos="1 1 2"><camera name="c"/></body></worldbody>',
            )
        )
        ok, rep, _ = gate(xml, dict(PHYSICS))
        self.assertTrue(ok, rep)
        self.assertNotIn("camera_rig__0", Scene(xml).objects)

    def test_scene_options_cannot_change_the_physics(self):
        from sbv.common.scene import Scene

        xml = edited_room(
            lambda x: x.replace(
                "<worldbody>",
                '<option density="1000" noslip_iterations="5"><flag frictionloss="disable"/></option><worldbody>',
                1,
            )
        )
        o = Scene(xml).m.opt
        self.assertEqual((o.density, o.noslip_iterations, o.disableflags), (0.0, 0, 0))

    def test_a_cup_with_see_through_walls_holds_nothing(self):
        from sbv.common.scene import Scene
        from sbv.functionality import f3_functions as Fn

        held = []
        for edit in (lambda x: x, lambda x: re.sub(r'(<geom name="cup_w\d")', r'\1 contype="0" conaffinity="0"', x)):
            s = Scene(edited_room(edit), probes=True)
            s.settle()
            held.append(Fn.container_once(s, "cup__0", None)[0])
        self.assertEqual(held, [True, False])

    def test_placement_sees_fixed_overlaps_and_side_contact(self):
        from sbv.common.scene import Scene
        from sbv.geometry import g3_layout as L

        ref = os.path.dirname(edited_room(lambda x: x))
        spec = dict(
            support_tol=0.01, penetration_tol=0.002, bounds_tol=0.02, bounds=[[-0.1, -0.1, -0.05], [4.1, 3.1, 2.5]]
        )

        def rows(*args):
            s = Scene(
                perturb.main([ref, tempfile.mkdtemp(prefix="sbv_p_"), *args])
                if args
                else os.path.join(ref, "scene.xml")
            )
            s.reset(s.q0)
            return {r["object"]: r for r in L.placement(spec, [], s, {n: n for n in s.objects})["objects"]}

        self.assertTrue(all(r["passed"] for r in rows().values()))
        self.assertFalse(
            rows("--move", "cabinet__0", "0.2", "0")["cabinet__0"]["non_penetration"]
        )  # a fixed cabinet sunk into the wall
        self.assertFalse(
            rows("--move", "book__0", "0.45", "0")["book__0"]["support"]
        )  # a book floating past the table edge

    def test_a_lamp_flush_on_a_mesh_desk_is_not_sunk(self):
        """MuJoCo 3.14 reads a cylinder standing flush on a mesh slab as 76 mm deep at most places, on arm64 and amd64
        alike. The verifiers' distance reads 0 there, and still reads a real 5 mm sink and a 3 mm lift."""
        import mujoco
        from sbv.geometry import g3_layout as L

        slab = " ".join(
            f"{x} {y} {z}" for z in (-0.03, 0.03) for x, y in ((-0.6, -0.4), (0.6, -0.4), (0.6, 0.4), (-0.6, 0.4))
        )
        m = mujoco.MjModel.from_xml_string(
            f'<mujoco><asset><mesh name="slab" vertex="{slab}"/></asset><worldbody>'
            '<geom name="desk" type="mesh" mesh="slab" pos="0 0 0.79"/><body name="lamp__0" pos="0 0 0.828">'
            '<freejoint/><geom name="base" type="cylinder" size="0.068 0.008"/></body></worldbody></mujoco>'
        )
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        s = SimpleNamespace(m=m, d=d)
        raw = []
        for x in np.linspace(-0.4, 0.4, 9):
            for y in np.linspace(-0.25, 0.25, 5):
                for dz in (0.0, -0.005, 0.003):
                    d.geom_xpos[1] = (x, y, 0.828 + dz)
                    raw.append(mujoco.mj_geomDistance(m, d, 1, 0, 0.01, None))
                    self.assertAlmostEqual(L.pair_gap(s, 1, 0, 0.01), dz, delta=1e-6)
        self.assertLess(
            min(raw), -0.05
        )  # MuJoCo's own reading does go wrong on this grid, so the check above means something

    def test_a_lamp_nanometres_above_its_table_is_not_sunk_into_it(self):
        # MuJoCo reads shapes within nanometres of touching as deep as the whole table; the lamp must still count as
        # sunk-free AND supported, and a real 5 mm overlap must still fail
        import mujoco
        from sbv.common.scene import Scene
        from sbv.geometry import g3_layout as L

        spec = dict(
            support_tol=0.01, penetration_tol=0.002, bounds_tol=0.02, bounds=[[-0.1, -0.1, -0.05], [4.1, 3.1, 2.5]]
        )

        def lamp(z):
            s = Scene(
                edited_room(
                    lambda x: x.replace(
                        "</worldbody>",
                        f'<body name="lamp__0" pos="1.6 1.3 {z}"><freejoint/>'
                        '<geom name="lamp_base" type="cylinder" pos="0 0 0.008" size="0.068 0.008"/></body></worldbody>',
                    )
                )
            )
            s.reset(s.q0)
            return s, next(
                r for r in L.placement(spec, [], s, {n: n for n in s.objects})["objects"] if r["object"] == "lamp__0"
            )

        s, row = lamp("0.740000001")  # 1 nm above the 0.74 m table top
        a, b = (mujoco.mj_name2id(s.m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("lamp_base", "table_top"))
        self.assertLess(
            mujoco.mj_geomDistance(s.m, s.d, a, b, 0.01, np.zeros(6)), -0.05
        )  # the misreading this self-test covers
        self.assertTrue(row["passed"])
        self.assertEqual(
            [lamp(z)[1]["non_penetration"] for z in ("0.739", "0.735")], [True, False]
        )  # 1 mm in passes, 5 mm in fails

    def test_tall_furniture_blocks_the_floor_and_sizes_ignore_turning(self):
        from sbv.common.scene import Scene
        from sbv.geometry import g3_layout as L, g2_shape as Geo

        s = Scene(edited_room(lambda x: x))
        s.reset(s.q0)
        xs, ys, free = L.floor_plan(s, [[-0.1, -0.1, -0.05], [4.1, 3.1, 2.5]], 0.3)
        self.assertFalse(free[np.argmin(abs(xs - 3.7)), np.argmin(abs(ys - 1.4))])
        t = Geo.surface_samples(s, "table__0")
        c, sn = np.cos(np.pi / 4), np.sin(np.pi / 4)
        R = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
        self.assertTrue(np.allclose(Geo.oriented_dims(t), Geo.oriented_dims(t @ R.T), atol=0.005))
        self.assertGreater(Geo.shape_scores(t, t @ R.T, 0.03)["fscore"], 0.99)

    def test_pose_size_bars_are_ratios_on_each_side(self):  # G2.3: each side within 1.25x (main) and 2x (reference bar)
        from sbv.geometry import g2_shape as Geo

        box = lambda *d: np.array([[x, y, z] for x in (0, d[0]) for y in (0, d[1]) for z in (0, d[2])], float)
        slots, req = (
            [dict(slot="desk__0", center=[0, 0, 0], dims=[1.0, 0.5, 0.75])],
            [dict(slot="desk__0", position_tol=0.1, size_bars=list(Geo.SIZE_BARS))],
        )
        for dims, want in (
            ((1.2, 0.42, 0.75), (True, True)),
            ((1.6, 0.5, 0.75), (False, True)),
            ((1.0, 0.5, 0.3), (False, False)),
        ):
            row = Geo.pose_size(
                req,
                slots,
                SimpleNamespace(center=lambda g: np.zeros(3)),
                {"desk__0": "d"},
                {"d": box(*dims)},
                {"desk__0": box(1.0, 0.5, 0.75)},
            )["objects"][0]
            self.assertEqual(
                (row["passed"], row["passed_reference_bar"]), want, dims
            )  # 20% too long failed the old 15% bar

    def test_pose_size_credit_is_smooth_between_the_bars(
        self,
    ):  # G2.3: full within the bar, 0 at twice it
        from sbv.geometry import g2_shape as Geo

        box = lambda *d: np.array([[x, y, z] for x in (0, d[0]) for y in (0, d[1]) for z in (0, d[2])], float)
        slots, req = (
            [dict(slot="desk__0", center=[0, 0, 0], dims=[1.0, 0.5, 0.75])],
            [dict(slot="desk__0", position_tol=0.1, size_bars=list(Geo.SIZE_BARS))],
        )

        def row(dims, at=0.0):
            return Geo.pose_size(
                req,
                slots,
                SimpleNamespace(center=lambda g: np.array([at, 0, 0])),
                {"desk__0": "d"},
                {"d": box(*dims)},
                {"desk__0": box(1.0, 0.5, 0.75)},
            )

        size = [row((x, 0.5, 0.75))["objects"][0]["score"] for x in (1.2, 1.3, 2.5**0.5, 1.8, 2.0, 2.5)]
        self.assertEqual(size[0], 1.0)
        self.assertAlmostEqual(size[2], 0.5, places=6)
        self.assertEqual(size[4:], [0.0, 0.0])
        self.assertTrue(1.0 > size[1] > size[2] > size[3] > 0.0, size)  # 30% too long now earns most, not none
        pos = [row((1.0, 0.5, 0.75), at)["objects"][0]["score"] for at in (0.1, 0.15, 0.2)]
        self.assertEqual([round(x, 6) for x in pos], [1.0, 0.5, 0.0])
        both = row((1.3, 0.5, 0.75), 0.15)
        self.assertAlmostEqual(both["objects"][0]["score"], size[1] * 0.5, places=5)  # the checks' credits multiply
        self.assertEqual(
            (both["score_passed"], both["objects"][0]["passed"]), (0.0, False)
        )  # the hard verdict stays beside it

    def test_a_long_object_turned_90_degrees_fails_pose(
        self,
    ):  # G2.3: footprint 1.5 times as long as wide, diagonal 0.5 m or more
        from sbv.geometry import g2_shape as Geo

        box = lambda *d: np.array([[x, y, z] for x in (0, d[0]) for y in (0, d[1]) for z in (0, d[2])], float)

        def turn(p, deg):
            c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
            return p @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T

        def row(dims, deg):  # the copy is the true box turned about z: same sizes, same centre
            slots, req = (
                [dict(slot="s__0", center=[0, 0, 0], dims=list(dims))],
                [dict(slot="s__0", position_tol=0.1, size_bars=list(Geo.SIZE_BARS))],
            )
            return Geo.pose_size(
                req,
                slots,
                SimpleNamespace(center=lambda g: np.zeros(3)),
                {"s__0": "s"},
                {"s": turn(box(*dims), deg)},
                {"s__0": box(*dims)},
            )["objects"][0]

        for dims in (
            (0.9, 0.4, 1.3),
            (1.2, 0.8, 0.74),
        ):
            self.assertEqual(
                [(row(dims, d)["passed"], round(row(dims, d)["long_axis_error"], 6)) for d in (90, 20)],
                [(False, 90.0), (True, 20.0)],
                dims,
            )
        for dims in (
            (0.14, 0.042, 0.1),
            (0.6, 0.5, 0.9),
        ):  # a trinket (diagonal 0.18 m) and a cabinet (1.2 times): not checked
            r = row(dims, 90)
            self.assertTrue(r["passed"])
            self.assertNotIn("long_axis_error", r)
        slots, req = (
            [dict(slot="s__0", center=[0, 0, 0], dims=[0.9, 0.4, 1.3])],
            [dict(slot="s__0", position_tol=0.1, size_bars=list(Geo.SIZE_BARS))],
        )
        r = Geo.pose_size(
            req,
            slots,
            SimpleNamespace(center=lambda g: np.zeros(3)),
            {"s__0": "s"},
            {"s": turn(box(0.62, 0.6, 1.3), 90)},
            {"s__0": box(0.9, 0.4, 1.3)},
        )["objects"][0]  # a near-square copy has no long axis to read: not checked
        self.assertNotIn("long_axis_error", r)
        self.assertEqual((r["passed"], r["passed_reference_bar"]), (False, True))  # sizes still answer

    def test_a_copy_where_nothing_moves_fails_support_removal(self):
        from sbv.common.scene import Scene
        from sbv.functionality import f2_physical as P

        unit = dict(kind="remove", slots=["crate__0", "vase__0"], params=dict(control="book__0"))
        glued = Scene(
            perturb.main(
                [os.path.dirname(edited_room(lambda x: x)), tempfile.mkdtemp(prefix="sbv_g_"), "--fix", "vase__0"]
            ),
            probes=True,
        )
        glued.settle()
        ref = Scene(edited_room(lambda x: x), probes=True)
        ref.settle()
        self.assertEqual(P.trial(ref, unit)[0], [True, False])
        self.assertEqual(P.trial(glued, unit)[0], [False, False])

    def test_input_only_ignores_rollouts_of_hint_only_joint_components(self):
        from sbv.functionality.units import judge
        from sbv.functionality.f3_functions import COMPONENTS

        unit = dict(
            kind="joint", tol=0.01, hint_components=["rest", "actuation", "full_travel"], expect=dict(rest=True)
        )
        obs = dict({k: True for k in COMPONENTS}, rest=True, actuation=[False, True], full_travel=True)
        bad = SimpleNamespace(owners=[1], component="full_travel", ok=lambda o, t: False, report=lambda o: {})
        self.assertEqual(judge(unit, obs, [bad])["passed"], {"physics_hint": False, "input_only": True})

    def test_input_cameras_are_not_the_held_out_views(self):
        from sbv.common.scene import Scene
        from sbv.appearance import views as App

        s = Scene(edited_room(lambda x: x))
        s.reset(s.q0)
        b = [[-0.1, -0.1, -0.05], [4.1, 3.1, 2.5]]
        held = {tuple(c["position"]) for c in App.cameras(s, b)}
        inp = {tuple(c["position"]) for c in App.cameras(s, b, height=1.6, kind="input")}
        self.assertEqual((len(held), len(inp), held & inp), (4, 12, set()))  # twelve input cameras
        self.assertTrue(
            set(App.CONDITIONS["1"])
            <= set(App.CONDITIONS["2"])
            <= set(App.CONDITIONS["4"])
            <= set(App.CONDITIONS["8"])
            <= set(App.CONDITIONS["12"])
        )

    def test_held_out_cameras_repeat_by_seed_and_stay_on_free_floor_in_the_room(self):
        from sbv.common.scene import Scene
        from sbv.appearance import views as App
        from sbv.geometry.g3_layout import floor_plan

        s = Scene(edited_room(lambda x: x))
        s.reset(s.q0)
        b = [[-0.1, -0.1, -0.05], [4.1, 3.1, 2.5]]
        J = App.JITTER
        xs, ys, free = floor_plan(s, b, 0.25)
        floor = {(round(xs[i], 3), round(ys[j], 3)) for i, j in np.argwhere(free)}
        picks = [np.array(c["position"][:2]) for c in App.cameras(s, b)]
        self.assertEqual(App.cameras(s, b, seed=7), App.cameras(s, b, seed=7))
        self.assertNotEqual(App.cameras(s, b, seed=7), App.cameras(s, b, seed=8))
        for seed in range(20):
            for cam, pick in zip(App.cameras(s, b, seed=seed), picks):
                x, y, z = cam["position"]
                dl = np.subtract(cam["look_at"], [2.0, 1.5, 0.8])
                self.assertIn((x, y), floor)
                self.assertTrue(
                    b[0][0] < x < b[1][0] and b[0][1] < y < b[1][1] and J["height"][0] <= z <= J["height"][1]
                )
                self.assertLessEqual(np.linalg.norm([x, y] - pick), J["move"] + 1e-6)
                self.assertTrue(abs(dl[:2]).max() <= J["look_side"] + 1e-3 and abs(dl[2]) <= J["look_up"] + 1e-3)

    def test_furniture_with_joints_is_found_by_its_place_under_another_word(self):
        # true chest drawer__0 built as cabinet__0; the nearer cabinet__1 has no joints; the hinged box__0 is 2 m from
        # drawer__1
        at = dict(cabinet__0=(0.3, 0, 0.4), cabinet__1=(0.05, 0, 0.4), cabinet__2=(0.3, 2, 0.4), box__0=(3, 0, 0.4))
        scene = SimpleNamespace(
            objects=at,
            cat={n: n.split("__")[0] for n in at},
            center=lambda n: np.array(at[n]),
            aabb=lambda n: (np.array(at[n]) - 0.4, np.array(at[n]) + 0.4),
            joints=dict(cabinet__0=[0, 1, 2], cabinet__1=[], cabinet__2=[3], box__0=[4]),
        )
        slots = [
            dict(slot=n, category=n.split("__")[0], role="fixed+joints", center=c, dims=[0.8, 0.5, 0.75])
            for n, c in (("drawer__0", (0, 0, 0.4)), ("drawer__1", (5, 0, 0.4)), ("cabinet__0", (0.3, 2, 0.4)))
        ]
        self.assertEqual(K.correspond(slots, scene), {"cabinet__0": "cabinet__2", "drawer__0": "cabinet__0"})

    def test_an_unnamed_joint_is_still_calibrated(self):
        from sbv.common.scene import Scene
        from sbv.functionality import f3_calibrate as CF

        s = Scene(edited_room(lambda x: x.replace('<joint name="drawer_slide" ', "<joint ")), probes=True)
        s.settle()
        units = [CF.cal_joint(s, "cabinet__0", j)[0] for j in s.joints["cabinet__0"]]
        self.assertTrue(all(units), units)


def with_a_worker_killed(fn):
    """fn() while a thread kills its first new worker process with SIGKILL, as the kernel's out-of-memory killer would.
    An alarm stops fn after 120 s, so a pool that waits forever fails the self-test instead of hanging it."""
    import multiprocessing as mp, signal, threading

    before = {p.pid for p in mp.active_children()}

    def kill():
        while not (new := [p for p in mp.active_children() if p.pid not in before]):
            time.sleep(0.01)
        os.kill(new[0].pid, signal.SIGKILL)

    def hung(*_):
        raise TimeoutError("still running 120 s after a worker was killed")

    threading.Thread(target=kill, daemon=True).start()
    old = signal.signal(signal.SIGALRM, hung)
    signal.alarm(120)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def textured_room(picture):
    """The test room with a picture texture (file tex.png, given as bytes) on the floor and a 2048-pixel builtin
    skybox."""
    path = edited_room(
        lambda x: x.replace(
            "<worldbody>",
            '<asset><texture name="pic" type="2d" file="tex.png"/><material name="floor_mat" texture="pic"/>'
            '<texture name="sky" type="skybox" builtin="gradient" width="2048" height="2048"/></asset><worldbody>',
            1,
        ).replace('<geom name="floor" ', '<geom name="floor" material="floor_mat" ')
    )
    with open(os.path.join(os.path.dirname(path), "tex.png"), "wb") as f:
        f.write(picture)
    return path


def png(w=64, h=32):
    chunk = lambda t, b: struct.pack(">I", len(b)) + t + b + struct.pack(">I", zlib.crc32(t + b))
    rows = b"".join(b"\0" + bytes((x * 4 + y) % 256 for x in range(3 * w)) for y in range(h))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


class Textures(unittest.TestCase):
    """Without Appearance the verifiers load 1-pixel textures (sbv.common.scene.flat_textures)."""

    def test_flat_textures_change_only_the_texture_arrays(self):
        from sbv.common.scene import Scene

        xml = textured_room(png())
        full, flat = Scene(xml).m, Scene(xml, textures=False).m
        self.assertEqual((full.ntex, flat.ntex), (2, 2))
        self.assertGreater(full.tex_data.nbytes, 2048 * 2048 * 3)
        self.assertLess(flat.tex_data.nbytes, 100)
        arrays = [k for k in dir(full) if isinstance(getattr(full, k), np.ndarray) and not k.startswith(("_", "tex_"))]
        self.assertGreater(len(arrays), 100)
        self.assertEqual([k for k in arrays if not np.array_equal(getattr(full, k), getattr(flat, k))], [])

    def test_a_broken_picture_still_fails_the_gate(self):
        from sbv.common.scene import PHYSICS, Scene
        from sbv.functionality.f1_gate import gate

        ok, rep, scene = gate(textured_room(png()), dict(PHYSICS), textures=False)
        self.assertTrue(ok, rep)
        self.assertFalse(scene.textures)
        xml = textured_room(b"not a png")
        Scene(xml, textures=False)  # loads flat, so only the gate's own compile can catch it
        for textures in (True, False):
            ok, rep, scene = gate(xml, dict(PHYSICS), textures=textures)
            self.assertFalse(ok)
            self.assertIsNone(scene)
            self.assertIn("does not compile", rep["definitions"][0])
            self.assertIn("tex.png", rep["definitions"][0])

    def test_calibration_loads_no_pictures(self):
        """calibrate loads every room with 1-pixel pictures (calibrate.LOAD): the workers of its pools, the prepare job
        and the main process, since a room's layered materials can cost a worker many gigabytes. A picture MuJoCo cannot
        decode shows it in the workers' own processes: a flat load never reads the file. The real pictures are still
        compiled once (calibrate.resources, then the F1 gate), so calibrating that room still stops, naming the file."""
        from unittest.mock import patch
        from sbv.common.scene import PHYSICS
        from sbv.common.workers import Workers

        bad, phys = textured_room(b"not a png"), dict(PHYSICS)
        with Workers(
            1, calibrate._init, (bad, phys, [0.0, 0.0], {}), ((bad, phys), calibrate.LOAD)
        ) as pool:  # objects, events, robot
            self.assertIs(pool.submit(_worker_textures, 0).result(), False)
        with Workers(1, calibrate._init_records, (bad, phys)) as pool:  # the init records
            self.assertTrue(pool.submit(calibrate._record_job, 0).result()[0])
        with self.assertRaisesRegex(
            SystemExit, "(fails the F1 gate|reference compile).*tex.png"
        ):  # calibrate.resources checks first
            calibrate.build(
                os.path.dirname(bad),
                procs=1,
                log=lambda *a: None,
                robot_tasks=0,
                push_tasks=0,
                require_tasks=False,
                camera_seed=7,
            )
        seen, real = [], calibrate.Scene

        def spy(*a, **k):
            s = real(*a, **k)
            seen.append(s.textures)
            return s

        with patch.object(calibrate, "Scene", spy):  # the prepare job
            calibrate._prepare(
                dict(
                    ref=os.path.dirname(textured_room(png())),
                    jitter=0,
                    nav_start=None,
                    mounts=None,
                    functions=None,
                    camera_seed=7,
                )
            )
        self.assertEqual(seen, [False])


def _worker_textures(_):
    return calibrate.S.textures


class TestRoom(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="sbv_test_")
        cls.ref = os.path.dirname(rooms.write(os.path.join(cls.tmp, "ref")))
        case = calibrate.build(cls.ref, procs=4, draws=6, need=5, jitter=2, log=lambda *a: None, require_tasks=False)
        case["reference"] = os.path.relpath(K.scene_xml(cls.ref), cls.tmp)
        cls.case_path = os.path.join(cls.tmp, "case.json")
        K.save(case, cls.case_path)
        cls.case = K.load(cls.case_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def score(self, name, *args):
        path = self.ref if not args else os.path.dirname(perturb.main([self.ref, os.path.join(self.tmp, name), *args]))
        r = evaluate.evaluate(self.case, path, procs=4, log=lambda *a: None)
        failed = {k for k, v in r["units"].items() if not all(x["passed"]["physics_hint"] for x in v)}
        return r, failed

    def test_unit_workers_load_no_pictures_when_appearance_is_scored(self):
        """The F2.2-F4.2 workers and their small models never decode the room's pictures, also when Appearance is
        scored: physics never reads them, and on a room with large layered textures every small model held a copy."""
        from unittest.mock import patch

        seen, real = [], evaluate._unit_jobs

        def spy(jobs, procs, initargs):
            seen.append(initargs[5])
            return real(jobs, procs, initargs)

        with patch.object(evaluate, "_unit_jobs", spy), patch.object(evaluate.App, "evaluate", lambda *a, **k: {}):
            r = evaluate.evaluate(
                self.case, self.ref, procs=4, log=lambda *a: None, renders=os.path.join(self.tmp, "pictures_renders")
            )
        self.assertEqual(seen, [False])
        self.assertFalse(r.get("incomplete"))

    def test_a_killed_worker_ends_the_run_with_the_cause(self):
        from sbv import pipeline

        r = with_a_worker_killed(lambda: evaluate.evaluate(self.case, self.ref, procs=4, log=lambda *a: None))
        lost = {k for k, v in r["units"].items() if any(x.get("error", "").startswith("WorkerDied: ") for x in v)}
        self.assertTrue(lost)
        self.assertEqual(lost, set(r["incomplete"]["units"]))
        self.assertIn("SIGKILL", r["incomplete"]["reason"])
        self.assertTrue(
            all("out of memory" in x["error"] and not x["passed"]["input_only"] for k in lost for x in r["units"][k])
        )
        self.assertLess(r["seconds_by_part"]["F2.2-F4.2"], 15)
        self.assertLess(r["functionality"]["physics_hint"]["functionality"], 100)
        self.assertTrue(evaluate.summary(r).startswith("GRADER FAILED (no reward): INCOMPLETE: "))
        self.assertEqual(r["status"], "grader_failed")
        rep, out = os.path.join(self.tmp, "killed.json"), os.path.join(self.tmp, "killed_reward.json")
        pipeline.write(rep, r)
        self.assertEqual(pipeline.reward_file(rep, out), 1)
        self.assertNotIn("reward", json.load(open(out)))

    def test_a_killed_worker_stops_calibration_with_the_cause(self):  # an error, not a partial case
        from sbv.common.workers import WorkerDied

        t = time.time()
        with self.assertRaisesRegex(WorkerDied, "SIGKILL"):
            with_a_worker_killed(
                lambda: calibrate.build(
                    self.ref, procs=4, draws=6, need=5, jitter=2, log=lambda *a: None, require_tasks=False
                )
            )
        self.assertLess(time.time() - t, 15)

    def appearance(self, path, case, name):  # Blender-free machines get a render status; the warning still runs
        return evaluate.evaluate(
            dict(case, appearance=dict(case["appearance"], samples=4)),
            path,
            procs=4,
            log=lambda *a: None,
            renders=os.path.join(self.tmp, name),
        )

    def test_an_old_case_keeps_its_cameras_and_still_evaluates(self):
        from sbv.common.scene import Scene
        from sbv.appearance import views as App

        s = Scene(K.scene_xml(self.ref))
        s.reset(s.q0)
        app = {k: v for k, v in self.case["appearance"].items() if not k.startswith("camera_")}
        old = dict(
            self.case, appearance=dict(app, cameras=App.cameras(s, self.case["room"]["bounds"]))
        )  # as calibrated before the jitter
        before = json.dumps(old["appearance"]["cameras"])
        r = self.appearance(self.ref, old, "old_renders")
        self.assertEqual(json.dumps(old["appearance"]["cameras"]), before)
        self.assertEqual(r["appearance"]["near_camera"], [])
        self.assertEqual(r["functionality"]["input_only"]["functionality"], 100.0)
        moved = copy.deepcopy(old)
        moved["appearance"]["cameras"][0]["position"][0] += 0.01
        self.assertNotEqual(App.ref_key(old), App.ref_key(moved))  # a new pose never reuses an old reference render

    def test_a_case_from_before_reference_files_still_evaluates(self):
        """A case calibrated before calibrate recorded reference_files (every reference room and released task before
        this) is still graded: its scene file's hash is checked as before, the file manifest only when the case has
        one."""
        old = copy.deepcopy(self.case)
        old.pop("reference_files", None)
        self.assertIn("reference_files", self.case)
        self.assertEqual(K.reference_path(old), K.reference_path(self.case))
        r = evaluate.evaluate(old, self.ref, procs=4, log=lambda *a: None)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["functionality"]["physics_hint"]["functionality"], 100.0)

    def test_a_board_in_front_of_a_held_out_camera_is_flagged(self):
        cam = self.case["appearance"]["cameras"][0]
        fw = -np.array(cam["cam_to_world"])[:3, 2]
        p = np.array(cam["position"]) + 0.1 * fw
        board = (
            f'<geom name="board" type="box" size="0.3 0.3 0.005" pos="{p[0]} {p[1]} {p[2]}" zaxis="{fw[0]} {fw[1]} {fw[2]}" '
            'contype="0" conaffinity="0" rgba="1 1 1 1"/>'
        )  # picture-only, 10 cm in front of view_same_0
        path = os.path.join(self.tmp, "board")
        shutil.copytree(self.ref, path)
        xml = K.scene_xml(path)
        with open(xml) as f:
            text = f.read()
        with open(xml, "w") as f:
            f.write(text.replace("</worldbody>", board + "</worldbody>", 1))
        r = self.appearance(path, self.case, "board_renders")
        near = r["appearance"]["near_camera"]
        from sbv.common.scene import Scene
        from sbv.appearance import views as App

        s = Scene(K.scene_xml(self.ref))
        s.reset(s.q0)
        cams = self.case["appearance"]["cameras"]
        self.assertEqual(App.near_cameras(cams, s, true=s), [])  # a faithful copy is never warned
        above = next(c for c in cams if c["source_input"] == cam["source_input"] and c is not cam)  # view_high_0
        self.assertEqual(above["position"][:2], cam["position"][:2])  # stands 0.55 m right above view_same_0
        # The board, for the camera it faces and the one above, and nothing the reference room has there itself.
        self.assertEqual(
            sorted((w["camera"], w["geom"]) for w in near), sorted([(cam["name"], "board"), (above["name"], "board")])
        )
        self.assertAlmostEqual(next(w["distance"] for w in near if w["camera"] == cam["name"]), 0.095, places=2)
        self.assertEqual(r["functionality"]["input_only"]["functionality"], 100.0)
        self.assertIn("Appearance WARNING", evaluate.summary(r))

    def test_calibration(self):
        kinds = {u["kind"] for u in self.case["functionality"]["units"]}
        self.assertTrue(
            {"init", "push", "tilt", "slide", "edge", "joint", "opening"} <= kinds
            and not {"load", "remove", "chain"} & kinds
            and "container" not in kinds,
            kinds,
        )

    def test_reference_scores_full(self):
        r, failed = self.score("ref")
        self.assertTrue(r["F1"]["passed"])
        self.assertEqual(failed, set())
        for s in ("physics_hint", "input_only"):
            self.assertEqual(r["functionality"][s]["functionality"], 100.0)
        g = r["geometry"]
        for v in (
            g["G1"]["inventory"]["f1"],
            g["G2"]["fscore"],
            g["G2"]["pose_size"]["score"],
            g["G3"]["relations"]["score"],
            g["G3"]["placement"]["score"],
            g["G3"]["navigation"]["score"],
            g["G2"]["pose_size"]["score_reference_bar"],
        ):
            self.assertAlmostEqual(v, 1.0, places=6)
        from sbv.functionality.units import NUMBERS  # every unit carries its verifier number and timing

        self.assertEqual(
            {r["verifiers"][u["id"]] for u in self.case["functionality"]["units"]},
            {NUMBERS[u["kind"]] for u in self.case["functionality"]["units"]},
        )
        self.assertTrue(all("seconds" in x for v in r["units"].values() for x in v if "record" not in x))
        self.assertTrue(
            {"F1", "G", "F2.1", "F2.2-F4.2"} <= set(r["seconds_by_part"])
            and {"G1", "G2.3", "G3.3"} <= set(g["seconds"])
        )

    def test_reference_bars_score_full_and_are_reported(self):
        from sbv import pipeline

        s = pipeline.scores(self.score("ref")[0])
        self.assertEqual([s[k + "_reference_bar"] for k in ("reward_functionality", "pose_size")], [100.0] * 2)
        bars = {
            u["kind"]: u["params"]["bars"] for u in self.case["functionality"]["units"] if "bars" in u.get("params", {})
        }
        self.assertTrue(
            {"push", "tilt", "slide", "joint"} <= set(bars) and all(b == [2.0, 1.25] for b in bars.values()), bars
        )

    def test_grippier_book_and_stiffer_door_pass_at_factor_2_and_fail_at_factor_1_25(self):
        r, _ = self.score(
            "grippy", "--friction", "book__0", "1.0", "--damping", "door_hinge", "3"
        )  # friction 1.67x; opening force 1.6x
        self.assertTrue({"push|book__0", "tilt|book__0"} <= set(r["units"]))
        for k in ("push", "tilt", "slide"):
            for x in r["units"].get(f"{k}|book__0", []):
                self.assertEqual(
                    (x["passed"]["physics_hint"], x["passed_reference_bar"]["physics_hint"]), (True, False), k
                )
        door = r["units"]["joint|cabinet__0|door_hinge"][0]
        self.assertEqual(
            (door["components"]["actuation"], door["components_reference_bar"]["actuation"]), (True, False)
        )

    def test_missing_cup(self):
        r, failed = self.score("nocup", "--remove", "cup__0")
        self.assertIn("init|cup__0", failed)
        self.assertIn("opening|cup__0|0", failed)
        self.assertAlmostEqual(r["geometry"]["G1"]["inventory"]["f1"], 12 / 13)
        self.assertLess(r["geometry"]["G3"]["relations"]["score"], 1)

    def test_heavy_book(self):
        r, failed = self.score("heavybook", "--mass", "book__0", "3")
        self.assertIn("push|book__0", failed)
        self.assertAlmostEqual(r["parameter_errors"]["book__0"]["mass_ratio"], 3.0, places=2)

    def test_welded_drawer(self):
        r, failed = self.score("weld", "--weld", "drawer_slide")
        self.assertEqual(failed, {"joint|cabinet__0|drawer_slide"})
        self.assertFalse(r["units"]["joint|cabinet__0|drawer_slide"][0]["components"]["presence"])

    def test_flipped_door(self):
        r, failed = self.score("flip", "--flip", "door_hinge")
        self.assertIn("joint|cabinet__0|door_hinge", failed)
        self.assertFalse(r["units"]["joint|cabinet__0|door_hinge"][0]["components"]["axis"])

    def test_door_damping_is_hint_only(self):
        r, failed = self.score("damper", "--damping", "door_hinge", "5")
        res = r["units"]["joint|cabinet__0|door_hinge"][0]
        self.assertFalse(res["passed"]["physics_hint"])
        self.assertTrue(res["passed"]["input_only"])

    def test_solid_cup(self):
        r, failed = self.score("solidcup", "--solid", "cup__0")
        self.assertIn("opening|cup__0|0", failed)
        self.assertLess(r["geometry"]["G2"]["fscore"], 1)

    def test_a_cup_that_falls_is_tested_no_further(self):
        """an object that ends 5 cm or 30 degrees from where it was put (here a cup let go 25 cm above the table) fails
        its other units unrun, "not at rest", and the report gives the share of objects not still."""
        path = os.path.join(self.tmp, "dropcup")
        shutil.copytree(self.ref, path)
        xml = K.scene_xml(path)
        with open(xml) as f:
            text = f.read()
        lifted = text.replace('<body name="cup__0" pos="1.3 1.35 0.74"', '<body name="cup__0" pos="1.3 1.35 0.99"')
        self.assertNotEqual(lifted, text)
        with open(xml, "w") as f:
            f.write(lifted)
        r = evaluate.evaluate(self.case, path, procs=4, log=lambda *a: None)
        cup = [k for k in r["units"] if "cup__0" in k.split("|")[1:] and not k.startswith("init|")]
        self.assertTrue(cup)
        self.assertIn("cup__0", r["at_rest"]["away"])
        self.assertGreater(r["at_rest"]["share"], 0)
        self.assertTrue(
            all(
                "cup__0" in x.get("not_at_rest", []) and not x["passed"]["physics_hint"]
                for k in cup
                for x in r["units"][k]
            )
        )
        self.assertFalse(r["units"]["init|cup__0"][0]["passed"]["input_only"])

    def test_moved_chair(self):
        r, failed = self.score("movechair", "--move", "chair__0", "0.3", "0")
        row = next(o for o in r["geometry"]["G2"]["pose_size"]["objects"] if o["slot"] == "chair__0")
        self.assertFalse(row["passed"])

    def test_a_cabinet_under_another_word_keeps_its_door_and_drawer(self):
        path = os.path.join(self.tmp, "dresser")
        shutil.copytree(self.ref, path)
        xml = K.scene_xml(path)
        with open(xml) as f:
            text = f.read()
        with open(xml, "w") as f:
            f.write(text.replace('"cabinet__0"', '"dresser__0"'))
        r = evaluate.evaluate(self.case, path, procs=4, log=lambda *a: None)
        joints = [k for k in r["units"] if k.startswith("joint|")]
        self.assertEqual(r["correspondence"]["cabinet__0"], "dresser__0")
        self.assertEqual(len(joints), 2)
        self.assertTrue(
            all(x["passed"][s] for k in joints for x in r["units"][k] for s in ("physics_hint", "input_only"))
        )
        g1 = r["geometry"]["G1"]
        self.assertEqual(g1["inventory"]["f1"], 1)  # the inventory counts what is found;
        self.assertEqual(
            (g1["words"]["accuracy"] < 1, [w["copy"] for w in g1["words"]["wrong"]]), (True, ["dresser__0"])
        )  # the word apart

    def test_a_taller_cabinet_keeps_its_door_and_drawer(self):
        """moving parts are compared from the box's bottom centre. A crown on the copy's cabinet makes its box 48 cm
        taller; from the box centre every part looked 24 cm low, from the bottom nothing moved."""
        path = os.path.join(self.tmp, "crown")
        shutil.copytree(self.ref, path)
        xml = K.scene_xml(path)
        with open(xml) as f:
            text = f.read()
        tall = text.replace(
            '<body name="cabinet__0" pos="3.7 1.4 0">',
            '<body name="cabinet__0" pos="3.7 1.4 0"><geom name="cab_crown" type="box" pos="0 0 1.04" size="0.25 0.4 0.24"/>',
        )
        self.assertNotEqual(tall, text)
        with open(xml, "w") as f:
            f.write(tall)
        r = evaluate.evaluate(self.case, path, procs=4, log=lambda *a: None)
        joints = [k for k in r["units"] if k.startswith("joint|cabinet__0")]
        self.assertTrue(joints)
        self.assertTrue(
            all(x["components"]["connectivity"] for k in joints for x in r["units"][k]),
            [r["units"][k][0].get("details", {}).get("margins") for k in joints],
        )

    def test_on_needs_contact_and_retired_units_are_gone(self):
        """no load, push chain or support removal units are made or scored; an "on" relation needs the item to touch its
        support once settled. The reference room keeps every relation; a cup welded 8 mm above the table (inside the
        geometric tolerance, touching nothing) loses its "on"."""
        r = evaluate.evaluate(self.case, self.ref, procs=4, log=lambda *a: None)
        self.assertFalse([k for k in r["units"] if k.split("|")[0] in ("load", "chain", "remove")])
        on = [x for x in r["geometry"]["G3"]["relations"]["relations"] if x["type"] == "on"]
        self.assertTrue(on)
        self.assertTrue(all(x["touching"] and x["passed"] for x in on))
        path = os.path.join(self.tmp, "hovercup")
        shutil.copytree(self.ref, path)
        xml = K.scene_xml(path)
        with open(xml) as f:
            text = f.read()
        hover = text.replace(
            '<body name="cup__0" pos="1.3 1.35 0.74"><freejoint/>', '<body name="cup__0" pos="1.3 1.35 0.748">'
        )
        self.assertNotEqual(hover, text)
        with open(xml, "w") as f:
            f.write(hover)
        r = evaluate.evaluate(self.case, path, procs=4, log=lambda *a: None)
        cup = [x for x in r["geometry"]["G3"]["relations"]["relations"] if x["a"] == "cup__0" and x["type"] == "on"]
        self.assertTrue(cup)
        self.assertFalse(any(x["touching"] or x["passed"] for x in cup))

    def test_an_object_under_another_word_is_found_by_its_place(self):
        path = os.path.join(self.tmp, "bowl")
        shutil.copytree(self.ref, path)
        xml = K.scene_xml(path)
        with open(xml) as f:
            text = f.read()
        with open(xml, "w") as f:
            f.write(text.replace('"cup__0"', '"bowl__0"'))
        r = evaluate.evaluate(self.case, path, procs=4, log=lambda *a: None)
        cup = [k for k in r["units"] if "cup__0" in k.split("|")]
        self.assertEqual(r["correspondence"]["cup__0"], "bowl__0")
        self.assertTrue(cup)
        self.assertTrue(all(x["passed"]["physics_hint"] for k in cup for x in r["units"][k]))
        self.assertEqual(r["geometry"]["G1"]["inventory"]["f1"], 1)
        self.assertLess(r["geometry"]["G1"]["words"]["accuracy"], 1)

    def test_centimetres_fail_the_gate(self):
        r, _ = self.score("cm", "--scale", "100")
        self.assertFalse(r["F1"]["passed"])
        self.assertTrue(r["F1"]["units_frames"])
        self.assertEqual(r["functionality"]["physics_hint"]["functionality"], 0.0)

    def test_missing_asset_fails_the_gate(self):
        r, _ = self.score("missing", "--missing-asset")
        self.assertFalse(r["F1"]["passed"])
        self.assertTrue(r["F1"]["package"])


from sbv.functionality.f4_robot import plan as robot_plan  # noqa: E402
from sbv.appearance import views as App  # noqa: E402

HAS_ROBOT = robot_plan.installed()


@unittest.skipUnless(HAS_ROBOT, "RB-Y1 model not installed (scripts/get_robot.sh)")
class RobotTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="sbv_robot_")
        cls.ref = os.path.dirname(rooms.write(os.path.join(cls.tmp, "ref"), rooms.robot_room_xml()))
        case = calibrate.build(
            cls.ref, procs=4, draws=4, need=3, jitter=1, log=lambda *a: None, robot_tasks=2, push_tasks=1
        )
        case["reference"] = os.path.relpath(K.scene_xml(cls.ref), cls.tmp)
        path = os.path.join(cls.tmp, "case.json")
        K.save(case, path)
        cls.case = K.load(path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def robot(self, name, *args):
        path = self.ref if not args else os.path.dirname(perturb.main([self.ref, os.path.join(self.tmp, name), *args]))
        r = evaluate.evaluate(self.case, path, procs=4, log=lambda *a: None)
        self.pushes = {k: v[0] for k, v in r["units"].items() if k.startswith("robot_push|")}
        return r, {k: v[0] for k, v in r["units"].items() if k.startswith("robot|")}

    def test_reference_task_succeeds(self):
        r, tasks = self.robot("ref")
        self.assertTrue(tasks and self.pushes)
        self.assertEqual(r["functionality"]["physics_hint"]["layers"]["F4"], 100.0)
        self.assertTrue(all(t["details"]["miss_cm"] <= 1.0 for t in self.pushes.values()))

    def test_missing_goal_fails(self):
        _, tasks = self.robot("nogoal", "--remove", "cabinet__0")
        self.assertTrue(
            all(not t["passed"]["physics_hint"] and "missing" in t["details"]["reason"] for t in tasks.values())
        )

    def test_controller_adapts_to_a_moved_cup(self):
        _, tasks = self.robot("movecup", "--move", "cup__0", "0", "0.2")
        self.assertTrue(all(t["passed"]["physics_hint"] for t in tasks.values()))

    def test_robot_world_keeps_the_case_physics_and_grips_a_turned_cup(self):
        from sbv.functionality.f4_robot import plan as RP, pick_place as robot
        from sbv.common.scene import Scene, PHYSICS

        odd = edited_room(lambda x: x)  # any room: the robot's world must not take its <option>
        with open(odd) as f:
            xml = f.read()
        with open(odd, "w") as f:
            f.write(xml.replace("<worldbody>", '<option gravity="0 0 -3" timestep="0.005"/><worldbody>', 1))
        w = RP.World(odd, (2.0, 0.5, 0.0))
        self.assertEqual((list(w.m.opt.gravity), w.m.opt.timestep), (PHYSICS["gravity"], PHYSICS["timestep"]))
        turned = os.path.join(self.tmp, "turned")
        shutil.copytree(self.ref, turned, dirs_exist_ok=True)
        path = K.scene_xml(turned)
        with open(path) as f:
            xml = f.read()
        with open(path, "w") as f:
            f.write(
                xml.replace(
                    '<body name="cup__0" pos="1.85 2.0 0.74">',
                    '<body name="cup__0" pos="1.85 2.0 0.74" euler="0 0 0.7854">',
                )
            )
        unit = [u for u in self.case["functionality"]["units"] if u["kind"] == "robot"][0]
        s = Scene(path)
        self.assertTrue(robot.trial(s, unit, {n: n for n in s.objects})[0] == [True])

    def test_the_hint_check_keeps_the_robot_where_it_stands(self):
        from sbv.functionality.f4_robot import plan as RP

        w = RP.World(K.scene_xml(self.ref), (2.5, 1.0, 0.0))
        q = w.d.qpos.copy()
        RP.edit_physics(w, w.objects["cup__0"], dict(mass=1.3, friction=0.15))
        self.assertTrue(np.allclose(q, w.d.qpos))

    def test_glued_cup_fails(self):
        _, tasks = self.robot("gluedcup", "--fix", "cup__0")
        self.assertTrue(all(not t["passed"]["physics_hint"] for t in tasks.values()))
        self.assertTrue(
            self.pushes
            and all(not t["passed"]["physics_hint"] and t["details"]["miss_cm"] > 15 for t in self.pushes.values())
        )

    def test_named_tasks_are_run_and_kept_or_refused_with_a_reason(self):
        import json

        room = os.path.dirname(perturb.main([self.ref, os.path.join(self.tmp, "named")]))
        with open(os.path.join(room, "tasks.json"), "w") as f:
            json.dump(
                {
                    "push": [{"object": "cup__0", "direction": [0, 1]}],
                    "pick_place": [{"object": "vase__9", "goal": "cabinet__0"}],
                    "shelve": [
                        {
                            "object": "book__9",
                            "goal": "cabinet__0",
                            "bay": [[0, 0, 0], [1, 1, 1]],
                            "spine": [1, 0],
                            "front": [0, 1],
                        }
                    ],
                },
                f,
            )
        case = calibrate.build(
            room, procs=4, draws=4, need=3, jitter=1, log=lambda *a: None, robot_tasks=0, push_tasks=0
        )
        push = [u for u in case["functionality"]["units"] if u["id"] == "robot_push|cup__0"]
        self.assertTrue(push and push[0]["params"]["dir"] == [0.0, 1.0] and push[0]["named"])
        why = {e["id"]: e["reason"] for e in case["functionality"]["excluded"]}
        self.assertIn("no object named vase__9", why["robot|vase__9|cabinet__0"])
        self.assertIn("no object named book__9", why["robot_shelve|book__9|cabinet__0"])

    def test_room_without_task_objects_stops_with_the_reasons(self):
        room = os.path.dirname(perturb.main([self.ref, os.path.join(self.tmp, "nocup"), "--remove", "cup__0"]))
        with self.assertRaises(SystemExit) as e:
            calibrate.build(room, procs=4, draws=4, need=3, jitter=1, log=lambda *a: None, robot_tasks=1, push_tasks=1)
        self.assertIn("no working pick-and-place task", str(e.exception))
        self.assertIn("no working push task", str(e.exception))

    def test_push_force_follows_weight_and_a_too_grippy_cup_fails(self):
        self.robot("ref")
        f0 = [t["details"]["push_force_N"] for t in self.pushes.values()]
        self.robot("heavycup", "--mass", "cup__0", "3")  # still pushable: passes, with about three times the force
        self.assertTrue(
            all(
                t["passed"]["physics_hint"] and 2.2 * f < t["details"]["push_force_N"] < 4 * f
                for t, f in zip(self.pushes.values(), f0)
            )
        )
        self.robot("grippycup", "--friction", "cup__0", "3")  # too grippy: it tips or jams between the fingertips
        self.assertTrue(all(not t["passed"]["physics_hint"] for t in self.pushes.values()))


@unittest.skipUnless(HAS_ROBOT and importlib.util.find_spec("bpy"), "needs the RB-Y1 model and Blender's module")
class Pipeline(unittest.TestCase):
    def test_one_room_and_one_copy_end_to_end(self):
        import json
        from unittest import mock
        from sbv import pipeline

        work = tempfile.mkdtemp(prefix="sbv_work_")
        rooms.write(os.path.join(work, "rooms", "robotroom"), rooms.robot_room_xml())
        cushion = dict(
            name="cushion__0",
            kind="soft_body",
            on="table__0",
            shape=dict(type="box", pos=[1.3, 2.0, 0.77], size=[0.2, 0.2, 0.06]),
            material=dict(youngs_kpa=40, poisson=0.3, density=50),
        )  # non-rigid verifiers too, with a fixed readout for Genesis
        with open(os.path.join(work, "rooms", "robotroom", "nonrigid.json"), "w") as f:
            json.dump(dict(objects=[cushion]), f)
        perturb.main(
            [
                os.path.join(work, "rooms", "robotroom"),
                os.path.join(work, "submissions", "robotroom", "glued"),
                "--fix",
                "cup__0",
            ]
        )
        with mock.patch.object(
            pipeline.Nonrigid,
            "_one",
            return_value=dict(readout=0.02, ball_lowest=9.0, object_lowest=9.0, finite=True, seconds=0.1),
        ):
            rows = pipeline.run(work, procs=4, log=lambda *a: None, calib=dict(draws=4, need=3, jitter=1))
        self.assertEqual(
            (rows[0]["reward_geometry"], rows[0]["reward_response"]), (100.0, 100.0)
        )  # the cushion's size and press count in their parts
        self.assertTrue(
            os.path.exists(
                os.path.join(work, "results", "robotroom", "entrants", "glued_nonrigid", "nonrigid_report.json")
            )
        )
        self.assertEqual([(r["entrant"], r.get("note")) for r in rows][0], ("(the room itself)", "ready"))
        self.assertEqual(
            (rows[0]["reward_functionality_hint"], rows[0]["reward_interaction"], rows[0]["sim_ready"]),
            (100.0, 100.0, 1.0),
        )
        self.assertLess(rows[1]["reward_interaction"], 100.0)
        with open(os.path.join(work, "results", "summary.md")) as f:
            self.assertTrue(
                f.readline().startswith("| Room | Entrant | Overall | Overall (visible) | Parse | Sim-ready |")
            )
        from scenebench_harbor import make_tasks  # Harbor tasks: the public one never carries the answer
        make_tasks.main(["--work", work, "--out", os.path.join(work, "tasks"), "--pictures", "1"])
        with open(os.path.join(work, "tasks", "private", "room_ids.json")) as f:
            rid = json.load(f)["robotroom"]  # public tasks name the room by id only
        pub, priv = (os.path.join(work, "tasks", x, f"3dscenebench-{rid}") for x in ("public", "private"))
        self.assertTrue(
            os.path.exists(os.path.join(pub, "environment", "pictures", "input_0.png"))
            and os.path.exists(os.path.join(priv, "tests", "case.json"))
        )
        self.assertFalse(
            any(f in ("case.json", "scene.xml") or "room" in d.split(os.sep) for d, _, fs in os.walk(pub) for f in fs)
        )
        shutil.rmtree(work, ignore_errors=True)


@unittest.skipUnless(
    os.path.exists(App.TORCH_PY) and importlib.util.find_spec("bpy"),
    "needs Blender's module and the PyTorch environment (.venv-metrics)",
)
class Appearance(unittest.TestCase):
    def test_renders_repeat_and_metrics_see_a_moved_chair(self):
        import json, subprocess
        from sbv.common.scene import Scene

        tmp = tempfile.mkdtemp(prefix="sbv_app_")
        ref = rooms.write(os.path.join(tmp, "ref"))
        moved = perturb.main([os.path.dirname(ref), os.path.join(tmp, "moved"), "--move", "chair__0", "0.3", "0"])
        s = Scene(ref)
        s.reset(s.q0)
        bounds = [[-0.1, -0.1, -0.05], [4.1, 3.1, 2.5]]
        spec = dict(bounds=bounds, resolution=512, samples=16, cameras=App.cameras(s, bounds)[:2])
        for xml, out in ((ref, "r1"), (ref, "r2"), (moved, "m")):
            App.render(xml, spec, os.path.join(tmp, out))
        res = {}
        for out in ("r2", "m"):
            subprocess.run(
                [
                    App.TORCH_PY,
                    os.path.join(os.path.dirname(HERE), "sbv", "appearance", "metrics.py"),
                    os.path.join(tmp, "r1"),
                    os.path.join(tmp, out),
                    "--out",
                    os.path.join(tmp, out + ".json"),
                ],
                check=True,
                capture_output=True,
            )
            with open(os.path.join(tmp, out + ".json")) as f:
                report = json.load(f)
            if report.get("unavailable"):
                self.skipTest(f"image model weights are not on this machine: {sorted(report['unavailable'])}")
            res[out] = report["mean"]
        self.assertGreater(res["r2"]["siglip2"], 0.999)
        self.assertLess(res["r2"]["lpips_vgg"], 0.01)
        self.assertGreater(res["m"]["lpips_vgg"], 5 * max(res["r2"]["lpips_vgg"], 1e-3))
        if res["r2"]["dinov3"] is not None:  # None without the gated checkpoint
            self.assertGreater(res["r2"]["dinov3"], 0.999)
            self.assertLess(res["m"]["dinov3"], res["r2"]["dinov3"])
        shutil.rmtree(tmp, ignore_errors=True)

    def test_true_room_renders_once_into_the_cache(self):
        from sbv.common.scene import Scene, PHYSICS  # what a grader image does while it builds

        tmp = tempfile.mkdtemp(prefix="sbv_ref_")
        ref = rooms.write(os.path.join(tmp, "ref"))
        s = Scene(ref)
        s.reset(s.q0)
        bounds = [[-0.1, -0.1, -0.05], [4.1, 3.1, 2.5]]
        case = dict(
            _dir=tmp,
            reference="ref/scene.xml",
            room=dict(bounds=bounds),
            physics=dict(PHYSICS),
            appearance=dict(resolution=512, samples=4, cameras=App.cameras(s, bounds)[:1], input_cameras=[]),
        )
        os.environ["SBV_REF_CACHE"] = os.path.join(tmp, "cache")
        try:
            _, d = App.reference_renders(case)
            png = os.path.join(d, case["appearance"]["cameras"][0]["name"] + ".png")
            self.assertTrue(d.startswith(os.path.join(tmp, "cache")) and os.path.exists(png))
            t = os.path.getmtime(png)
            self.assertEqual(App.reference_renders(case)[1], d)
            self.assertEqual(os.path.getmtime(png), t)  # not rendered again
        finally:
            del os.environ["SBV_REF_CACHE"]
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
