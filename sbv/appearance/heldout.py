"""Two held-out appearance view sets, generated without physics calibration. Entrants do not get these pictures, but
the views follow from the input cameras by the fixed rule below, so they are held out, not secret.

The first eight input camera azimuths are shifted by half a 45 degree interval.
Set one retains each input's height. Set two alternates four higher and four
lower views. Every one stands 0.5 m, or 15 degrees about the room centre, from
every input camera (separated()), the rule a room's own cameras follow too.
Positions may move radially to find a clear spot that is that far apart; a
camera turns about the room centre only when its own ray has none (build()).
Only the reference room selects the cameras: a copy's
obstructing geometry must not cause the evaluator to choose easier views.

A room may instead bring its own cameras (SUPPLIED): input_cameras.json and
heldout_cameras.json beside scene.xml, chosen per room by the harness for
coverage. supplied() reads them and validate() checks them on the reference
room; there is no fallback when they are present but invalid.
"""

import argparse
import copy
import json
import math
import os
import re
from pathlib import Path

import mujoco
import numpy as np

from sbv.appearance.views import camera, sees
from sbv.common.scene import is_arch
from sbv.geometry.g3_layout import floor_plan

PROTOCOL = "shifted-eight-and-height-eight-v3"  # v2: every camera stands apart from every input (separated());
# 3: the room's own visual-only meshes block by their surfaces (clearance)
SUPPLIED = "per-room-supplied-v1"  # the room's own input_cameras.json and heldout_cameras.json
INPUTS_FILE, HELDOUT_FILE = "input_cameras.json", "heldout_cameras.json"
OBJECT_NAMED = re.compile(
    r"[a-z0-9_]+__\d+"
)  # an object's name (<category>__<n>) starts a world-body geom drawn for it
SEPARATION_M, SEPARATION_DEG = (
    0.5,
    15.0,
)  # a held-out camera stands this far, or this many degrees about the room centre, from every input
CLEARANCE_M, FLOOR_RADIUS_M = 0.15, 0.125
GROUPS = ("same_height", "high_low")
TURN_STEP_DEG = 2.5  # a shifted camera with no clear spot on its own ray turns about the room centre in steps this big
RECORD = ("name", "position", "look_at", "hfov_deg", "intrinsics", "cam_to_world", "clip", "axes")


def visual_mesh_near(scene, geom, local_point, radius):
    """Actual visual surface/solid occupancy after the AABB broad phase.

    A material may combine many separate wall pieces into one mesh whose box encloses the whole empty room.
    MuJoCo's mesh collision hull cannot answer this visual-only query. Welding duplicated face vertices does
    not move surfaces; closed components still reject interior points, and open surfaces retain the radius guard.
    """
    import trimesh

    m = scene.m
    mesh_id = int(m.geom_dataid[geom])
    cache = getattr(scene, "_visual_clearance_meshes", None)
    if cache is None:
        scene._visual_clearance_meshes = cache = {}
    if mesh_id not in cache:
        v = m.mesh_vert[m.mesh_vertadr[mesh_id] : m.mesh_vertadr[mesh_id] + m.mesh_vertnum[mesh_id]]
        f = m.mesh_face[m.mesh_faceadr[mesh_id] : m.mesh_faceadr[mesh_id] + m.mesh_facenum[mesh_id]]
        mesh = trimesh.Trimesh(vertices=v, faces=f, process=True)
        closed = [part for part in mesh.split(only_watertight=False, repair=False) if part.is_watertight]
        cache[mesh_id] = mesh, closed
    mesh, closed = cache[mesh_id]
    point = np.asarray(local_point, float)[None]
    try:
        _, distance, _ = trimesh.proximity.closest_point(mesh, point)
    except ModuleNotFoundError:
        _, distance, _ = trimesh.proximity.closest_point_naive(mesh, point)
    if distance[0] <= radius:
        return True
    for part in closed:
        if np.any(point[0] < part.bounds[0]) or np.any(point[0] > part.bounds[1]):
            continue
        # Solid-angle winding is an exact triangle query and needs no optional R-tree dependency.
        a, b, c = (part.triangles - point[0]).transpose(1, 0, 2)
        la, lb, lc = (np.linalg.norm(x, axis=1) for x in (a, b, c))
        numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
        denominator = (
            la * lb * lc
            + np.einsum("ij,ij->i", a, b) * lc
            + np.einsum("ij,ij->i", b, c) * la
            + np.einsum("ij,ij->i", c, a) * lb
        )
        if abs(float((2 * np.arctan2(numerator, denominator)).sum())) > 2 * np.pi:
            return True
    return False


def clearance(scene, position, radius=0.15):
    """Clearance from every geom, including visual-only surfaces and closed interiors.

    Primitives, collision meshes and every object's visual meshes retain conservative oriented local AABBs, so no
    camera stands inside an object's box, even an open shell's (a wardrobe drawn without a back), nor an object's
    look drawn on the world body (jacket__0_look). The room's own visual-only meshes (the other world-body geoms and
    architecture bodies, scene.is_arch) use those boxes as a broad phase,
    then their actual triangles and closed components: walls merged into one mesh by material must not turn the
    empty room into occupied space (protocol v3). Planes use point/plane distance.
    """
    m, d = scene.m, scene.d
    p = np.asarray(position, float)
    rot = d.geom_xmat.reshape(-1, 3, 3)
    local = np.einsum("nji,nj->ni", rot, p - d.geom_xpos)
    boxes = m.geom_aabb
    inside = np.all(np.abs(local - boxes[:, :3]) <= boxes[:, 3:] + radius, axis=1)
    plane = m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
    inside[plane] = np.abs(local[plane, 2]) <= radius

    def shell(i):  # the room's own visual-only mesh: its surfaces decide
        r = int(m.body_rootid[m.geom_bodyid[i]])
        if int(m.geom_type[i]) != int(mujoco.mjtGeom.mjGEOM_MESH) or m.geom_contype[i] or m.geom_conaffinity[i]:
            return False
        if r:
            return is_arch(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, r) or "")
        o = OBJECT_NAMED.match((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").lower())
        return o is None or is_arch(o.group(0))  # wall__0_visual is the room's; jacket__0_look is the jacket's

    ids = [
        int(i) for i in np.flatnonzero(inside) if not shell(int(i)) or visual_mesh_near(scene, int(i), local[i], radius)
    ]
    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(i)) or f"geom{i}" for i in ids]


def build(scene, bounds, inputs, shift_deg=22.5, height_delta=0.55, clearance_m=0.15, resolution=512):
    """Return cameras, named groups and per-camera reference validity evidence.

    Each camera first searches its own ray (its input's azimuth about the room centre, shifted by shift_deg) for a clear
    spot that stands apart (separated()) from every input camera, input_8-11 included. Where that ray has none
    (furniture filling that side of a small room), it turns about the room centre in TURN_STEP_DEG steps, alternately
    away from its input and back toward it, to the nearest ray with such a spot that also stands apart from the cameras
    already placed in its group; the record and evidence carry the turn. Fails explicitly when a camera has no such
    spot in any direction. No invalid fallback pose and no silent reduction in the number of test views.
    """
    if len(inputs) < 8:
        raise ValueError("two held-out groups require eight input cameras")
    if not 0 < shift_deg < 45 or not height_delta > 0:
        raise ValueError("shift must be between 0 and 45 degrees; height_delta must be positive")
    xs, ys, free = floor_plan(scene, bounds, 0.125)
    center = np.mean(np.asarray(bounds, float), axis=0)[:2]
    floor_z, top_z = float(bounds[0][2]), float(bounds[1][2])

    def spot(a, height, look, near, apart=None):
        """The clear point on the ray from the room centre at azimuth a (radians) nearest `near` metres out, or None."""
        direction = np.array([math.cos(a), math.sin(a)])
        # Search the whole ray from just inside the reference bounds.
        limits = [
            (bounds[1][j] - center[j] - clearance_m) / direction[j]
            if direction[j] > 1e-8
            else (bounds[0][j] - center[j] + clearance_m) / direction[j]
            for j in range(2)
            if abs(direction[j]) > 1e-8
        ]
        for r in sorted(np.arange(0.6, min(limits), 0.025), key=lambda r: abs(r - near)):
            q = center + r * direction
            ix, iy = np.argmin(abs(xs - q[0])), np.argmin(abs(ys - q[1]))
            if not free[ix, iy]:
                continue
            pos = np.r_[q, height]
            if (
                clearance(scene, pos, clearance_m)
                or not sees(scene, q, height, look)
                or (apart and not apart(pos.round(3)))
            ):  # as serialized
                continue
            return pos
        return None

    slots = []
    for group in GROUPS:
        for i, src in enumerate(inputs[:8]):
            p0 = np.asarray(src["position"], float)
            # A shared room centre, independent of object annotations or copy.
            v = p0[:2] - center
            height = p0[2] + (0 if group == "same_height" else height_delta if i % 2 == 0 else -height_delta)
            if not floor_z + clearance_m < height < top_z - clearance_m:
                raise ValueError(f"{group}[{i}]: requested height {height:.3f} is outside room bounds")
            name = f"view_{'same' if group == 'same_height' else 'high' if i % 2 == 0 else 'low'}_{i}"
            slots.append(
                dict(
                    group=group,
                    i=i,
                    src=src,
                    name=name,
                    height=height,
                    near=float(np.linalg.norm(v)),
                    look=np.asarray(src["look_at"], float),
                    a=math.atan2(v[1], v[0]) - math.radians(shift_deg),
                )
            )
    apart_from_inputs = lambda pos: not separated(pos, inputs, center)
    for s in slots:  # the shifted rule first, for every camera: a turn never moves another's spot
        s["pos"], s["turn"] = spot(s["a"], s["height"], s["look"], s["near"], apart_from_inputs), 0.0
    for s in slots:
        if s["pos"] is not None:
            continue
        mates = [
            dict(name=t["name"], position=t["pos"]) for t in slots if t["group"] == s["group"] and t["pos"] is not None
        ]
        apart = lambda pos: not separated(pos, inputs, center) and not separated(pos, mates, center)
        turns = (
            sign * k * TURN_STEP_DEG for k in range(1, int(180 / TURN_STEP_DEG) + 1) for sign in (-1, 1)
        )  # away from its input first
        for turn in turns:
            s["pos"] = spot(s["a"] + math.radians(turn), s["height"], s["look"], s["near"], apart)
            if s["pos"] is not None:
                s["turn"] = turn
                break
        if s["pos"] is None:
            raise ValueError(
                f"no clear, interior spot {SEPARATION_M} m or {SEPARATION_DEG} degrees from every input camera for "
                f"held-out camera {s['name']} (input {s['i']} shifted {shift_deg} degrees, then turned up to 180 "
                f"degrees either way): the room is too small or too full for the shifted rule; give it its own cameras "
                f"({INPUTS_FILE} and {HELDOUT_FILE})"
            )
    out, evidence = [], []
    groups = {g: [] for g in GROUPS}
    for s in slots:
        group, i, src, name, height = s["group"], s["i"], s["src"], s["name"], s["height"]
        shot = camera(name, s["pos"], s["look"], src["hfov_deg"], resolution)
        # Validate the *serialized*, rounded position the renderer will use.
        hits = clearance(scene, shot["position"], clearance_m)
        if hits or not sees(scene, shot["position"][:2], height, shot["look_at"]):
            raise ValueError(f"rounded camera {name} is invalid: {hits}")
        shift = shift_deg if not s["turn"] else round(shift_deg - s["turn"], 3)
        shot.update(
            source_input=src["name"],
            azimuth_shift_deg=shift,
            height_group=group if group == "same_height" else "high" if i % 2 == 0 else "low",
        )
        groups[group].append(name)
        out.append(shot)
        evidence.append(
            dict(
                camera=name,
                valid=True,
                clearance_m=clearance_m,
                occupied_geoms=hits,
                architecture_line_of_sight=True,
                separation=dict(m=SEPARATION_M, deg=SEPARATION_DEG),
                position=shot["position"],
                input_height=float(s["src"]["position"][2]),
                height_delta=round(height - s["src"]["position"][2], 5),
                **({"turned_deg": s["turn"]} if s["turn"] else {}),
            )
        )
    return dict(
        protocol=PROTOCOL,
        shift_deg=shift_deg,
        height_delta=height_delta,
        cameras=out,
        camera_groups=groups,
        validation=evidence,
    )


def update_case(case, scene, supplied_cameras=None, **options):
    """Copy only the appearance view configuration; preserve all physics evidence. supplied_cameras: the room's own
    (inputs, held-out) documents from supplied(), validated here and used in place of the shifted rule."""
    result = copy.deepcopy(case)
    a = result["appearance"]
    if supplied_cameras:
        made = validate(scene, case["room"]["bounds"], *supplied_cameras, resolution=a.get("resolution", 512))
        a["input_cameras"] = made.pop("input_cameras")
    else:
        made = build(scene, case["room"]["bounds"], a["input_cameras"], resolution=a.get("resolution", 512), **options)
    a.update(cameras=made["cameras"], camera_groups=made["camera_groups"], heldout_protocol=made)
    # A floor measured from the former cameras cannot normalize the new ones.
    # Raw metrics are available without measuring a replacement baseline.
    a.pop("floor", None)
    # Extra pictures were chosen apart from the former held-out cameras: choose them again (sbv.add_extra_cameras).
    a.pop("extra_cameras", None)
    a.pop("extra_protocol", None)
    a.pop("camera_seed", None)
    a.pop("camera_jitter", None)
    return result


def supplied(room):
    """The room's own (input_cameras.json, heldout_cameras.json) documents beside its scene.xml, or None when it has
    neither. One without the other is an error: a room either brings both or leaves both to the shifted rule."""
    folder = room if os.path.isdir(room) else os.path.dirname(os.path.abspath(room))
    paths = [os.path.join(folder, n) for n in (INPUTS_FILE, HELDOUT_FILE)]
    have = [os.path.isfile(p) for p in paths]
    if not any(have):
        return None
    if not all(have):
        raise ValueError(
            f"{folder}: {INPUTS_FILE} and {HELDOUT_FILE} come together; found only {os.path.basename(paths[have.index(True)])}"
        )
    docs = []
    for p in paths:
        with open(p) as f:
            docs.append(json.load(f))
    return tuple(docs)


def azimuth(point, center):
    return math.degrees(math.atan2(point[1] - center[1], point[0] - center[0]))


def separated(position, inputs, center):
    """Each input camera this held-out position stands too close to: under SEPARATION_M apart on the floor and under
    SEPARATION_DEG apart in azimuth about the room centre. Empty means the position is separated from every input."""
    close = []
    for c in inputs:
        q = c["position"]
        gap = math.hypot(position[0] - q[0], position[1] - q[1])
        turn = abs((azimuth(position, center) - azimuth(q, center) + 180) % 360 - 180)
        if gap < SEPARATION_M and turn < SEPARATION_DEG:
            close.append(c["name"])
    return close


def placement(scene, bounds, position, look, plan=None):
    """Why a camera at position looking at look cannot be used in the reference room, or None: it must stand inside the
    bounds, above free floor (floor_plan, FLOOR_RADIUS_M), CLEARANCE_M from every geom (visual ones too), between the
    floor and the top of the bounds with that clearance, and see the room past the architecture (views.sees)."""
    xs, ys, free = plan if plan is not None else floor_plan(scene, bounds, FLOOR_RADIUS_M)
    p = np.asarray(position, float)
    if not (bounds[0][0] < p[0] < bounds[1][0] and bounds[0][1] < p[1] < bounds[1][1]):
        return "outside the room bounds"
    if not bounds[0][2] + CLEARANCE_M < p[2] < bounds[1][2] - CLEARANCE_M:
        return f"height {p[2]:.3f} outside the room bounds"
    if not free[np.argmin(abs(xs - p[0])), np.argmin(abs(ys - p[1]))]:
        return "not above free floor"
    hits = clearance(scene, p, CLEARANCE_M)
    if hits:
        return f"within {CLEARANCE_M} m of {', '.join(hits[:4])}"
    if not sees(scene, p[:2], p[2], look):
        return "its line of sight to the look-at point crosses architecture"
    return None


def record_problem(c, resolution):
    """Why c is not a camera record as views.camera() writes it at this resolution, or None."""
    missing = [k for k in RECORD if k not in c]
    if missing:
        return f"missing {', '.join(missing)}"
    try:
        made = camera(
            c["name"],
            np.asarray(c["position"], float),
            np.asarray(c["look_at"], float),
            float(c["hfov_deg"]),
            resolution,
        )
    except Exception as e:
        return f"not a camera: {e}"
    for k in ("position", "intrinsics", "cam_to_world"):
        if np.asarray(c[k], float).shape != np.asarray(made[k]).shape or not np.allclose(
            c[k], made[k], atol=2e-3 if k == "intrinsics" else 1e-3
        ):
            return f"{k} disagrees with its position, look_at, hfov_deg and resolution {resolution}"
    return None


def validate(scene, bounds, inputs_doc, heldout_doc, resolution=512):
    """Check a room's own cameras on the reference room and return them in build()'s form, plus input_cameras.
    Inputs: 8 or 12 camera records named input_0 .. input_{n-1}, each placeable (placement()). Held-out: 16 records
    named view_*, in two groups of eight (GROUPS), each group linking every one of input_0-7 once by source_input, each
    placeable and separated from every input camera (separated()). Raises ValueError naming every problem."""
    problems = []
    raw = inputs_doc.get("cameras") if isinstance(inputs_doc, dict) else None
    held = heldout_doc.get("cameras") if isinstance(heldout_doc, dict) else None
    if not isinstance(raw, list) or not isinstance(held, list):
        raise ValueError(f'{INPUTS_FILE} and {HELDOUT_FILE} need a list under "cameras"')
    names = [c.get("name") if isinstance(c, dict) else None for c in raw]
    if len(raw) not in (8, 12) or sorted(names, key=str) != sorted((f"input_{k}" for k in range(len(raw))), key=str):
        raise ValueError(f"{INPUTS_FILE}: 8 or 12 cameras named input_0 .. input_n-1, found {names}")
    inputs = sorted(({k: c[k] for k in RECORD if k in c} for c in raw), key=lambda c: int(c["name"].split("_")[1]))
    plan = floor_plan(scene, bounds, FLOOR_RADIUS_M)
    center = np.mean(np.asarray(bounds, float), axis=0)[:2]
    for c in inputs:
        why = record_problem(c, resolution) or placement(scene, bounds, c["position"], c["look_at"], plan)
        if why:
            problems.append(f"{c['name']}: {why}")
    groups = heldout_doc.get("camera_groups")
    first = {f"input_{k}" for k in range(8)}
    if not isinstance(groups, dict) or set(groups) != set(GROUPS):
        raise ValueError(f"{HELDOUT_FILE}: camera_groups must be exactly {list(GROUPS)}")
    by_name = {c.get("name"): c for c in held if isinstance(c, dict)}
    listed = [n for g in GROUPS for n in groups[g]]
    if (
        len(held) != 16
        or len(by_name) != 16
        or sorted(listed, key=str) != sorted(by_name, key=str)
        or not all(str(n).startswith("view_") for n in by_name)
    ):
        raise ValueError(
            f"{HELDOUT_FILE}: 16 cameras named view_*, each in one group, found {sorted(by_name, key=str)}"
        )
    evidence, out = [], []
    for g in GROUPS:
        sources = [by_name[n].get("source_input") for n in groups[g]]
        if len(groups[g]) != 8 or set(sources) != first:
            problems.append(f"{g}: eight cameras linking input_0-7 once each by source_input, found {sources}")
        for n in groups[g]:
            c = by_name[n]
            why = record_problem(c, resolution) or placement(scene, bounds, c["position"], c["look_at"], plan)
            close = separated(c["position"], inputs, center)
            if close:
                why = (
                    why + "; " if why else ""
                ) + f"under {SEPARATION_M} m and {SEPARATION_DEG} degrees from {', '.join(close)}"
            if c.get("group", g) != g:
                why = (why + "; " if why else "") + f"says group {c.get('group')} but is listed in {g}"
            if why:
                problems.append(f"{n}: {why}")
                continue
            out.append(dict(c, group=g))
            evidence.append(
                dict(
                    camera=n,
                    valid=True,
                    clearance_m=CLEARANCE_M,
                    occupied_geoms=[],
                    architecture_line_of_sight=True,
                    separation=dict(m=SEPARATION_M, deg=SEPARATION_DEG),
                    position=c["position"],
                    source_input=c["source_input"],
                    rule=c.get("rule"),
                )
            )
    if problems:
        raise ValueError(f"the room's own cameras are not valid in the reference room: {'; '.join(problems)}")
    return dict(
        protocol=SUPPLIED,
        cameras=out,
        camera_groups={g: list(groups[g]) for g in GROUPS},
        validation=evidence,
        input_cameras=inputs,
        selection=inputs_doc.get("protocol"),
        rules=heldout_doc.get("protocol"),
    )


def main():
    from sbv.common import case as K
    from sbv.common.scene import Scene

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case")
    ap.add_argument("--out", required=True, help="new private case; the original is preserved")
    ap.add_argument("--shift-deg", type=float, default=22.5)
    ap.add_argument("--height-delta", type=float, default=0.55)
    args = ap.parse_args()
    case = K.load(args.case)
    xml = str(Path(K.reference_path(case)).resolve())
    scene = Scene(xml, case["physics"], textures=False)
    scene.reset(scene.q0)
    own = supplied(xml)  # a room's own cameras win over the shifted rule, as at calibration
    result = update_case(case, scene, own, shift_deg=args.shift_deg, height_delta=args.height_delta)
    result["reference"] = xml
    K.save({k: v for k, v in result.items() if not k.startswith("_")}, args.out)
    print(json.dumps({"out": args.out, "groups": result["appearance"]["camera_groups"]}))


if __name__ == "__main__":
    main()
