"""Appearance axis orchestration: the case's frozen held-out cameras and twelve input cameras (the pictures an entrant
receives: 1, 2, 4, 8 or 12 of them, by condition), Blender renders of the reference (cached next to the case) and of the
submission, then the three image metrics in the PyTorch environment (sbv/appearance/metrics.py). Held-out and
input-camera scores are reported separately; camera_groups additionally reports each hidden group independently. New
cases use heldout.py's two groups of eight; legacy cases keep their frozen cameras. The held-out scores are the
Appearance part of the reward (sbv/reward.py); they never enter Geometry or Functionality."""

import hashlib, json, os, subprocess, sys, time
import numpy as np, mujoco
from sbv.geometry.g3_layout import floor_plan, world_geoms

from sbv import ROOT as HERE

TORCH_PY = os.environ.get("SBV_TORCH_PYTHON", os.path.join(HERE, ".venv-metrics", "bin", "python"))


CONDITIONS = {
    "1": ["input_0"],
    "2": ["input_0", "input_2"],
    "4": ["input_0", "input_1", "input_2", "input_3"],  # pictures given
    "8": [f"input_{k}" for k in range(8)],
    "12": [f"input_{k}" for k in range(12)],
}  # 12: big or long rooms
JITTER = dict(
    move=0.4, height=[1.2, 1.6], look_side=0.3, look_up=0.2
)  # published ranges of the held-out cameras, metres
NEAR = 0.5  # copy geoms this close in front of a held-out camera: warning
NEAR_SLACK = 0.1  # ... when this much nearer than the reference room along the same ray
SPACING = 0.75  # input_4-11 stand this far from held-out and earlier inputs


def sees(scene, p, z, look):
    """A camera at floor point p, z metres up, sees the room: its line of sight to the look-at point crosses no wall,
    floor or ceiling (any geom of the world body or of an architectural body, visual or not); furniture may stand in
    between. Free floor behind a wall fails: the room
    bounds can reach past it, and a camera there would see only the wall."""
    m, d, g = scene.m, scene.d, np.zeros(1, np.int32)
    own = lambda k: int(scene.owner[m.geom_bodyid[k]])
    arch = {
        k
        for k in range(m.ngeom)
        if own(k) == 0 or (scene.name(own(k)) not in scene.objects and not scene.name(own(k)).startswith("probe_"))
    }
    pos = np.array([p[0], p[1], z], float)
    v = np.asarray(look, float) - pos
    L = float(np.linalg.norm(v))
    u, s = v / L, 0.0
    for _ in range(256):  # step past each object on the way, stop at architecture
        x = mujoco.mj_ray(m, d, pos + s * u, u, None, 1, -1, g)  # every group: a visual-only wall blocks the view too
        if x < 0 or s + x >= L:
            return True
        if int(g[0]) in arch:
            return False
        s += x + 1e-4
    return True


def cameras(scene, bounds, height=1.4, hfov=60.0, res=512, kind="held_out", seed=None):
    """Four interior views from the free floor nearest each corner of the room (held-out, named view_k, at 1.4 m), or
    twelve input views (named input_k, at 1.6 m): nearest the middle of each wall (input_0-3, as before eight existed),
    then nearest the point halfway from each wall's middle to its next corner (input_4-7), each at least SPACING from
    the held-out picks and from earlier inputs, all looking at the room centre 0.8 m up; then input_8-11 for what those
    eight miss (cover(): each aimed at the objects no earlier input shows, a big room's far zones and corners);
    intrinsics and camera-to-world (OpenGL axes: -z forward, +y up) saved with each. The two sets never share a
    position. With a seed (calibration draws one and keeps it in the private case), each camera moves to a random
    free-floor point within JITTER["move"] of its pick, at a random height in JITTER["height"], looking at a point moved
    up to JITTER["look_side"] sideways and JITTER["look_up"] up or down. These seeded corner views are no longer the
    scoring cameras: sbv/appearance/heldout.py builds those from the input cameras by a fixed rule (or the room brings
    its own), so the scoring views follow from this code and a task's cameras.json. They are held out, not secret. Every
    pick, jittered or not, is the nearest free floor from which the camera sees the room (sees())."""
    xs, ys, free = floor_plan(scene, bounds, 0.25)
    cells = np.argwhere(free)
    pts = np.c_[xs[cells[:, 0]], ys[cells[:, 1]]]
    (x0, y0, _), (x1, y1, _) = bounds
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2, 0.8])
    out = []
    anchors = (
        [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        if kind == "held_out"
        else [
            (c[0], y0),
            (x1, c[1]),
            (c[0], y1),
            (x0, c[1]),  # wall middles: the 1, 2 and 4 conditions
            ((c[0] + x1) / 2, y0),
            (x1, (c[1] + y1) / 2),
            ((x0 + c[0]) / 2, y1),
            (x0, (y0 + c[1]) / 2),  # 8: halfway on
            ((x0 + c[0]) / 2, y0),
            (x1, (y0 + c[1]) / 2),
            ((c[0] + x1) / 2, y1),
            (x0, (c[1] + y1) / 2),
        ]
    )  # 12: halfway back
    rng = None if seed is None else np.random.default_rng(seed)
    J = JITTER
    taken = [h["position"][:2] for h in cameras(scene, bounds, kind="held_out")] if kind == "input" else []
    apart = (
        lambda q, k: k < 4 or min(np.linalg.norm(np.subtract(taken, q), axis=1)) >= SPACING
    )  # input_4-11: clear of the rest
    for k, corner in enumerate(anchors):
        if k == 8:  # the 12-picture condition: aimed at what the first eight miss
            for k2, shot in enumerate(cover(scene, pts, out, taken, height, hfov), start=8):
                if shot is None:
                    break
                taken.append(shot[0][:2])
                out.append(camera(f"input_{k2}", shot[0], shot[1], hfov, res))
            if len(out) == 12:
                break
        if k < len(out):
            continue  # input_8-11 already aimed by cover(): its anchor is not needed
        order = np.argsort(np.linalg.norm(pts - np.add(corner, 0.0), axis=1)) if len(pts) else []
        p = next((pts[i] for i in order if apart(pts[i], k) and sees(scene, pts[i], height, c)), None)
        if p is None:
            p = next(
                (pts[i] for i in order if sees(scene, pts[i], height, c)),
                pts[order[0]] if len(pts) else np.array(corner) * 0.8 + c[:2] * 0.2,
            )
        taken.append(p)
        z, look = height, c
        if rng is not None:
            near = (
                pts[np.linalg.norm(pts - p, axis=1) <= J["move"]] if len(pts) else p[None]
            )  # the same draws as before:
            q, z = near[rng.integers(len(near))], rng.uniform(*J["height"])
            look = c + rng.uniform(-1, 1, 3) * [J["look_side"], J["look_side"], J["look_up"]]
            p = (
                q
                if sees(scene, q, z, look)
                else next((x for x in near[np.argsort(np.linalg.norm(near - q, axis=1))] if sees(scene, x, z, look)), p)
            )
        out.append(
            camera(
                f"{'view' if kind == 'held_out' else 'input'}_{k}",
                np.array([p[0], p[1], z]),
                np.asarray(look, float),
                hfov,
                res,
            )
        )
    return out


def camera(name, pos, look, hfov, res):
    """One camera record: intrinsics and camera-to-world (OpenGL axes: -z forward, +y up)."""
    f = look - pos
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    fx = res / 2 / np.tan(np.radians(hfov) / 2)
    M = np.vstack([np.c_[r, u, -f, pos], [0, 0, 0, 1]])
    return dict(
        name=name,
        position=pos.round(3).tolist(),
        look_at=look.round(3).tolist(),
        hfov_deg=hfov,
        intrinsics=[[round(fx, 3), 0, res / 2], [0, round(fx, 3), res / 2], [0, 0, 1]],
        cam_to_world=M.round(5).tolist(),
        clip=[0.05, 50.0],
        axes="OpenGL: -z forward, +y up",
    )


def shown(scene, pos, look, hfov, names):
    """The objects among names a camera at pos looking at look shows: the centre of each one's box, or the middle of its
    top (a table's centre is the empty space under it), inside the picture and the first thing a ray to it meets."""
    m, d, g = scene.m, scene.d, np.zeros(1, np.int32)
    f = look - pos
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    u, t, out = np.cross(r, f), np.tan(np.radians(hfov) / 2), set()
    for n in names:
        lo, hi = scene.aabb(n)
        c = (lo + hi) / 2
        for q in (c, np.r_[c[:2], hi[2] - 0.01]):
            v = q - pos
            depth = float(v @ f)
            if depth <= 0.1 or abs(v @ r) > t * depth or abs(v @ u) > t * depth:
                continue
            if (
                mujoco.mj_ray(m, d, pos, v / np.linalg.norm(v), None, 1, -1, g) >= 0
                and int(scene.owner[m.geom_bodyid[g[0]]]) == scene.objects[n]
            ):
                out.add(n)
                break
    return out


def cover(scene, pts, cams, taken, height, hfov, n=4, ok=None):
    """Up to n more cameras, one at a time, each showing the most objects no earlier camera shows, the objects and goals
    of the room's robot tasks (tasks.json, activities included) first: aimed at one such object from free floor 1.5, 2.5
    or 4 m away in one of eight directions, at least SPACING from every earlier camera and seeing it past the
    architecture (sees()); ok(position, look-at), if given, must accept the spot too (the extra pictures' placement
    rule, extra.py). Yields (position, look-at), or None once nothing new can be shown."""
    unseen = set(scene.objects) - set().union(
        *(shown(scene, np.array(c["position"]), np.array(c["look_at"]), hfov, scene.objects) for c in cams)
    )
    first = task_objects(scene) & unseen
    taken = [np.asarray(t, float) for t in taken]
    for _ in range(n):
        best = None
        for o in sorted(unseen):
            lo, hi = scene.aabb(o)
            look = (lo + hi) / 2
            for dist in (1.5, 2.5, 4.0):
                for a in np.arange(8) * np.pi / 4:
                    q = look[:2] + dist * np.array([np.cos(a), np.sin(a)])
                    i = int(np.argmin(np.linalg.norm(pts - q, axis=1))) if len(pts) else -1
                    if (
                        i < 0
                        or np.linalg.norm(pts[i] - q) > 0.3
                        or (taken and min(np.linalg.norm(np.subtract(taken, pts[i]), axis=1)) < SPACING)
                    ):
                        continue
                    if not sees(scene, pts[i], height, look):
                        continue
                    pos = np.array([pts[i][0], pts[i][1], height])
                    if ok is not None and not ok(pos, look):
                        continue
                    got = shown(scene, pos, look, hfov, unseen)
                    if best is None or (len(got & first), len(got)) > (len(best[2] & first), len(best[2])):
                        best = (pos, look, got)
        if best is None or not best[2]:
            yield None
            return
        taken.append(best[0][:2])
        unseen -= best[2]
        first -= best[2]
        yield best[0], best[1]


def task_objects(scene):
    """The objects and goals the room's named robot tasks use (tasks.json next to its scene, activity steps
    included)."""
    from sbv.functionality.f4_robot import tasks as T

    spec = T.load(getattr(scene, "xml", "")) if getattr(scene, "xml", None) else {}
    ts = [t for k in T.KINDS for t in spec.get(k, [])] + [
        st for a in spec.get("activity", []) for st in a.get("steps", [])
    ]
    pkg = {
        o for t in spec.get("package", []) if isinstance(t, dict) for o in t.get("objects", []) if isinstance(o, str)
    }  # F4.7
    return {t[f] for t in ts for f in ("object", "goal") if t.get(f) in scene.objects} | (pkg & set(scene.objects))


def _rays(cam, n):
    """Unit directions of an n x n grid of rays through a camera's field of view."""
    t = np.linspace(-1, 1, n)
    gx, gy = (a.reshape(-1, 1) for a in np.meshgrid(t, t))
    M = np.array(cam["cam_to_world"])
    s = np.tan(np.radians(cam["hfov_deg"]) / 2)
    v = -M[:3, 2] + s * (gx * M[:3, 0] + gy * M[:3, 1])
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _cast(scene, cam, v, cutoff):
    gid, dist = np.full(len(v), -1, np.int32), np.full(len(v), -1.0)
    mujoco.mj_multiRay(
        scene.m, scene.d, np.array(cam["position"], float), v.ravel(), None, 1, -1, gid, dist, None, len(v), cutoff
    )
    return gid, dist


def ray_depths(cams, scene, n=32, cutoff=NEAR + NEAR_SLACK):
    """{camera: distance to the first thing along each of near_cameras()'s rays, inf past cutoff}: the reference room's,
    for near_cameras(true=...). The scene must be at its initial state."""
    out = {}
    for cam in cams:
        gid, dist = _cast(scene, cam, _rays(cam, n), cutoff)
        out[cam["name"]] = np.where((gid >= 0) & (dist >= 0), dist, np.inf)
    return out


def near_cameras(cams, scene, reach=NEAR, n=32, true=None):
    """Diagnostic, never scored: every geom of the copy (picture-only ones included) that a held-out camera sees within
    `reach` metres, found with an n x n grid of rays through its field of view, nearest hit per (camera, geom). With the
    reference room (`true`: its Scene, or its ray_depths()), a hit counts only when it is more than NEAR_SLACK nearer
    than what the reference room shows along the same ray: the reference room's own furniture close to a camera, copied
    faithfully, is no warning (with sixteen held-out cameras, the low ones often stand within reach of a table). A board
    carrying a painted corner view just past the 5 cm near clip shows up either way, also in front of furniture the
    reference room has there. Scenes at their initial state."""
    m = scene.m
    hit = {}
    if true is not None and not isinstance(true, dict):
        true = ray_depths(cams, true, n)
    for cam in cams:
        gid, dist = _cast(scene, cam, _rays(cam, n), reach)
        there = true[cam["name"]] if true is not None else np.full(len(gid), np.inf)
        for g, x, y in zip(gid.tolist(), dist.tolist(), there.tolist()):
            if g >= 0 and 0 <= x <= reach and x < y - NEAR_SLACK and x < hit.get((cam["name"], g), (np.inf,))[0]:
                hit[cam["name"], g] = (x, y)
    return [
        dict(
            camera=c,
            geom=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}",
            body=scene.name(scene.owner[m.geom_bodyid[g]]),
            distance=round(x, 3),
            **({} if true is None else dict(true_distance=None if y == np.inf else round(y, 3))),
        )
        for (c, g), (x, y) in sorted(hit.items())
    ]


def light_height(scene, bounds):
    """Where the ceiling light hangs: 5 cm under the lowest ceiling found straight above the room (rays up from 1.5 m
    through anything that is not a ceiling, as ceiling_light says), else at 2.6 m or the room's top. Frozen in the case
    as light_z; grading lowers it for a copy whose own ceiling is lower (light_for)."""
    z = ceiling_light(scene, bounds)
    (_, _, z1) = bounds[1]
    return z if z is not None else round(min(z1, 2.6), 3)


LIGHT_RULE = "light-under-lower-ceiling-1"  # in the reference cache key: the rule that sets the light height per copy


def light_for(case, scene):
    """The light height for grading one copy, used for its renders and the reference room's alike: the case's light_z (5
    cm under the true ceiling), or 5 cm under the copy's own ceiling where that is lower."""
    (_, _, z1) = case["room"]["bounds"][1]
    z0 = case["appearance"].get("light_z", round(min(z1, 2.6), 3))
    zc = ceiling_light(scene, case["room"]["bounds"])
    return dict(z=z0 if zc is None else min(z0, zc), case=z0, copy=zc)


CEILING_SHARE = (
    0.25  # a drawn, non-colliding architecture geom is a ceiling when seen from above it covers this share of the room
)


def ceiling_light(scene, bounds):
    """5 cm under the lowest ceiling straight above the room (see light_height), or None when there is none. A ceiling
    is an architecture geom that collides, or one the renders draw (render.py's rule, Scene.visible) that covers at
    least CEILING_SHARE of the room's floor area from above. Visual-only ceilings were missed before, the light stayed
    above them and those copies rendered black."""
    m, d = scene.m, scene.d
    (x0, y0, _), (x1, y1, z1) = bounds
    gid = np.zeros(1, np.int32)
    hits = []
    own = lambda g: scene.name(scene.owner[m.geom_bodyid[g]])
    area = CEILING_SHARE * (x1 - x0) * (y1 - y0)
    drawn = set(
        scene.visible([g for g in range(m.ngeom) if own(g) not in scene.objects and not own(g).startswith("probe_")])
    )
    arch = set(world_geoms(scene))
    wide = {}

    def ceiling(g):
        if g in arch or g not in drawn:
            return g in arch
        if g not in wide:
            p = scene.geom_points(g)
            e = p.max(0) - p.min(0)
            wide[g] = e[0] * e[1] >= area
        return wide[g]

    for fx, fy in [(0.5, 0.5), (0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]:
        p = np.array([x0 + fx * (x1 - x0), y0 + fy * (y1 - y0), 1.5])
        for _ in range(8):
            x = mujoco.mj_ray(m, d, p, np.array([0.0, 0.0, 1.0]), None, 1, -1, gid)
            if x < 0:
                break
            if ceiling(int(gid[0])):
                hits.append(p[2] + x)
                break
            p = p + np.array([0.0, 0.0, x + 1e-4])
    return round(min(hits) - 0.05, 3) if hits else None


def input_pictures(case, out, condition="4"):
    """The pictures an entrant receives for a condition: the reference room rendered at those input cameras, plus their
    calibration (cameras.json)."""
    from sbv.common.case import reference_path

    names = set(CONDITIONS[str(condition)])
    cams = [c for c in case["appearance"]["input_cameras"] if c["name"] in names]
    spec = dict(case["appearance"], bounds=case["room"]["bounds"], physics=case["physics"], cameras=cams)
    render(reference_path(case), spec, out)
    with open(os.path.join(out, "cameras.json"), "w") as f:
        json.dump(dict(resolution=spec.get("resolution", 512), cameras=cams), f, indent=1)
    return [c["name"] for c in cams]


def extra_pictures(case, out):
    """The extra pictures (sbv/appearance/extra.py) an entrant receives beside the main ones: the reference room
    rendered at the case's extra cameras, plus their calibration (cameras.json). A case without extra cameras renders
    nothing."""
    from sbv.common.case import reference_path

    cams = case.get("appearance", {}).get("extra_cameras") or []
    if not cams:
        return []
    spec = dict(case["appearance"], bounds=case["room"]["bounds"], physics=case["physics"], cameras=cams)
    render(reference_path(case), spec, out)
    with open(os.path.join(out, "cameras.json"), "w") as f:
        json.dump(dict(resolution=spec.get("resolution", 512), cameras=cams), f, indent=1)
    return [c["name"] for c in cams]


RENDER_TIMEOUT_S = float(
    os.environ.get("SBV_RENDER_TIMEOUT_S", 3600)
)  # one Blender run (a room's views); a room's views take seconds to minutes


def render(xml, spec, out):
    """Render in a separate Blender process so its state never touches the scorer. The spec (the case's cameras and
    physics) is written beside the pictures only while Blender runs, and removed whether it finishes or fails: a
    picture folder is later given to entrants (input_pictures), and used to keep the spec as cameras.json when a render
    stopped half way."""
    os.makedirs(out, exist_ok=True)
    cams = os.path.join(out, "render_spec.json")
    with open(cams, "w") as f:
        json.dump(spec, f)
    try:
        r = subprocess.run(
            [sys.executable, "-m", "sbv.appearance.render", xml, cams, out],
            cwd=HERE,
            capture_output=True,
            text=True,
            timeout=RENDER_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"render failed: stopped after {RENDER_TIMEOUT_S:g} s (SBV_RENDER_TIMEOUT_S)"
        ) from None  # a copy's render: appearance 0, as any failed copy render
    finally:
        if os.path.exists(cams):
            os.remove(cams)
    if r.returncode:
        raise RuntimeError(f"render failed: {r.stderr[-800:]}")
    with open(os.path.join(out, "render.json")) as f:
        return json.load(f)


def input_key(case):
    """What the input pictures of a case depend on: the reference room's files, the input cameras (and the extra ones,
    when the case has them), the render settings, the room bounds and the render overrides. Not the case hash or the
    held-out cameras: they change at every calibration, the pictures do not. make_tasks renders a room's pictures again
    when this changes."""
    from sbv.common.case import reference_path, fingerprint

    a = case.get("appearance", {})
    spec = json.dumps(
        [
            a.get("input_cameras"),
            a.get("resolution"),
            a.get("samples"),
            a.get("light_z"),
            case.get("room", {}).get("bounds"),
        ]
        + ([a["extra_cameras"]] if a.get("extra_cameras") else []),
        sort_keys=True,
        default=float,
    )
    from sbv.common.dependencies import model_files, render_identity

    deps = dict(model=model_files(reference_path(case)), renderer=render_identity(), physics=case.get("physics"))
    return hashlib.sha256(json.dumps([spec, deps], sort_keys=True).encode()).hexdigest()[:16]


def reference_renders(case, condition=None, light_z=None):
    """The reference room rendered at the held-out cameras, plus the input cameras of a picture condition ("1", "2",
    "4", "8" or "12") if given; rendered once and cached in SBV_REF_CACHE, or next to the case. Every file of the
    reference room, every camera pose, the render settings and the light height (light_z: a copy's, from light_for) key
    the cache, so a changed room, camera or light renders again. Returns (spec, folder)."""
    inputs = set(CONDITIONS.get(str(condition), [])) if condition else set()
    spec = dict(
        case["appearance"],
        bounds=case["room"]["bounds"],
        physics=case["physics"],
        cameras=case["appearance"]["cameras"]
        + [c for c in case["appearance"].get("input_cameras", []) if c["name"] in inputs],
    )
    if light_z is not None:
        spec["light_z"] = light_z
    ref_dir = os.path.join(os.environ.get("SBV_REF_CACHE") or case["_dir"], f"appearance_ref_{ref_key(case, light_z)}")
    if not all(os.path.exists(os.path.join(ref_dir, c["name"] + ".png")) for c in spec["cameras"]):
        from sbv.common.case import reference_path

        render(reference_path(case), spec, ref_dir)
    return spec, ref_dir


NOT_RENDERED = ("floor", "extra_cameras", "extra_protocol")  # appearance keys the reference renders do not depend on


def ref_key(case, light_z=None):
    """The reference-render cache key: the reference room's files, the appearance block with every camera pose (without
    the floor and the extra pictures' cameras, which the renders do not depend on), the room bounds, the light rule and
    height (light_z: a copy's, from light_for; default the case's), the render device and light overrides, and the
    renderer's own code (a fixed renderer never reuses renders made before the fix)."""
    from sbv.common.case import reference_path, fingerprint

    z = case["appearance"].get("light_z") if light_z is None else light_z
    a = json.dumps(
        [{k: v for k, v in case["appearance"].items() if k not in NOT_RENDERED}, case["room"]["bounds"], LIGHT_RULE, z],
        sort_keys=True,
        default=float,
    )
    from sbv.common.dependencies import model_files, render_identity

    deps = dict(model=model_files(reference_path(case)), renderer=render_identity(), physics=case.get("physics"))
    return hashlib.sha256(json.dumps([a, deps], sort_keys=True).encode()).hexdigest()[:16]


METRICS = ("siglip2", "dinov3", "lpips_vgg")


def floor(case, log=print):
    """The appearance floor of a case: the reference room's architecture alone (walls, floor, ceiling: no objects),
    rendered from the held-out cameras and scored against the reference room's own renders with the three metrics.
    pipeline.check writes it into the case as appearance.floor, and sbv/reward.py places every copy between it (0) and
    the reference room (100), so an empty room no longer collects most of the appearance part. None when Blender or the
    PyTorch environment is missing here (the reward then keeps the fixed mapping); the reason is logged."""
    from sbv.common.case import reference_path

    try:
        spec, ref_dir = reference_renders(case)
    except RuntimeError as e:
        log(f"appearance floor: no reference renders: {str(e)[:300]}")
        return None
    out = ref_dir + "_floor"
    spec = dict(spec, cameras=case["appearance"]["cameras"], architecture_only=True)
    try:
        if not all(os.path.exists(os.path.join(out, c["name"] + ".png")) for c in spec["cameras"]):
            render(reference_path(case), spec, out)
    except RuntimeError as e:
        log(f"appearance floor: {str(e)[:300]}")
        return None
    if not os.path.exists(TORCH_PY):
        log(f"appearance floor: no PyTorch environment at {TORCH_PY}")
        return None
    r = subprocess.run(
        [
            TORCH_PY,
            os.path.join(HERE, "sbv", "appearance", "metrics.py"),
            ref_dir,
            out,
            "--out",
            os.path.join(out, "metrics.json"),
            "--device",
            os.environ.get("SBV_METRIC_DEVICE", "cpu"),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode:
        log(f"appearance floor: metrics failed: {r.stderr[-300:]}")
        return None
    with open(os.path.join(out, "metrics.json")) as f:
        res = json.load(f)
    mean = res.get("mean") or {}
    if any(mean.get(k) is None for k in METRICS):
        log(f"appearance floor: a metric is unavailable: {res.get('unavailable') or res.get('status')}")
        return None
    return {k: round(float(mean[k]), 5) for k in METRICS}


def evaluate(case, xml, out, condition=None, scene=None):
    """Render and score the held-out views, plus the input cameras of a picture condition ("1", "2", "4", "8" or "12")
    if given. scene: the copy as a Scene (to find its ceiling); loaded here when not given."""
    inputs = set(CONDITIONS.get(str(condition), [])) if condition else set()
    t0 = time.time()
    if scene is None:
        from sbv.common.scene import Scene

        scene = Scene(xml, case["physics"], textures=False)
    scene.reset(scene.q0)
    light = light_for(case, scene)
    try:
        spec, ref_dir = reference_renders(case, condition, light["z"])
    except RuntimeError as e:
        return dict(status=f"reference {e}"[:1000])  # the grader's own render: recorded, left out of the reward
    try:
        t1 = time.time()
        info = render(xml, spec, out)
        t2 = time.time()
    except RuntimeError as e:
        return dict(
            status=str(e)[:1000], copy_render_failed=True
        )  # the copy's own files: appearance scores 0 (sbv/reward.py)
    if not os.path.exists(TORCH_PY):
        return dict(status=f"no PyTorch environment at {TORCH_PY}", render=info)
    r = subprocess.run(
        [
            TORCH_PY,
            os.path.join(HERE, "sbv", "appearance", "metrics.py"),
            ref_dir,
            out,
            "--out",
            os.path.join(out, "metrics.json"),
            "--device",
            os.environ.get("SBV_METRIC_DEVICE", "cpu"),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode:
        return dict(status=f"metrics failed: {r.stderr[-800:]}", render=info)
    with open(os.path.join(out, "metrics.json")) as f:
        res = json.load(f)
    if "status" in res:
        return dict(status=f"metrics failed: {res['status']}", render=info)
    gone = sorted(
        k for k in ("siglip2", "dinov3") if k in (res.get("unavailable") or {})
    )  # the two scored models: never the mean of the one that loaded
    if gone:
        return dict(
            status=("metrics unavailable: " + ", ".join(k + ": " + str(res["unavailable"][k]) for k in gone))[:1000],
            render=info,
        )
    secs = dict(
        reference_render=round(t1 - t0, 3), render=round(t2 - t1, 3), metrics=round(time.time() - t2, 3)
    )  # reference: 0 when cached

    def part(prefix):  # the views of one camera set, with their own means
        idx = [i for i, v in enumerate(res["views"]) if v.startswith(prefix)]
        per = {k: [vals[i] for i in idx] for k, vals in res["per_view"].items()}
        return dict(
            views=[res["views"][i] for i in idx],
            per_view=per,
            mean={k: round(float(np.mean(v)), 5) if v else None for k, v in per.items()},
        )

    held = part("view_")
    held.update(
        unavailable=res["unavailable"],
        render=info,
        light=light,
        reference_renders=ref_dir,
        submission_renders=out,
        seconds=secs,
    )
    if inputs:
        held["input"] = dict(part("input_"), condition=str(condition))
    groups = case["appearance"].get("camera_groups", {})
    if groups:
        held["groups"] = {}
        for group, names in groups.items():
            idx = [res["views"].index(n) for n in names]  # missing views are an error, not a smaller denominator
            per = {k: [v[i] for i in idx] for k, v in res["per_view"].items()}
            held["groups"][group] = dict(
                views=names, per_view=per, mean={k: round(float(np.mean(v)), 5) if v else None for k, v in per.items()}
            )
        held["protocol"] = case["appearance"].get("heldout_protocol", {}).get("protocol")
    return held


if __name__ == "__main__":  # pre-render a case's reference room, e.g. while building a grader image
    import argparse
    from sbv.common.case import load, save

    ap = argparse.ArgumentParser(
        description="python -m sbv.appearance.views CASE [--pictures 1|2|4|8|12] [--floor]: render the reference room once into the cache"
    )
    ap.add_argument("case")
    ap.add_argument("--pictures", choices=list(CONDITIONS))
    ap.add_argument(
        "--floor",
        action="store_true",
        help="also compute the appearance floor and write it into CASE (pipeline.check does this itself)",
    )
    a = ap.parse_args()
    case = load(a.case)
    print(reference_renders(case, a.pictures)[1])
    if a.floor:
        f = floor(case)
        if f is None:
            raise SystemExit("no appearance floor: Blender or the PyTorch environment is missing")
        case["appearance"]["floor"] = f
        save({k: v for k, v in case.items() if not k.startswith("_")}, a.case)
        print("floor", f)
