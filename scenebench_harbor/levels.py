"""Instruction levels for the Harbor track:
  l0  the pictures and cameras, the public rules and the room's robot tasks in words;
  l1  + what the verifiers do in this room, in category words (from the case's kept units only): which parts move and
      how, which containers keep things in, what is carried along; and inventory.json, measurements.json and depth/
      (counts and roles, sizes, joint travel, metric depth).
No level's text gives an instance id, a position, a force, an angle, an amount or a count (L1 names kinds of things
only). L1's files hold measured values from the reference room: counts and roles, sizes and joint travel keyed by the
reference room's object names, and metric depth per picture. audit() lists the functions the room declares
(functions.json, tasks.json) that calibration did not keep; make_tasks warns about them, since no level announces or
scores a dropped function (the room should be fixed)."""

import collections, json, os, re
import mujoco, numpy as np
from sbv.common.scene import Scene
from sbv.geometry import g2_shape as G

LEVELS = ("l0", "l1")
PLURAL = {"box": "boxes", "fries": "fries", "food": "food", "glass": "glasses", "bench": "benches", "shelf": "shelves"}
PARTS = (
    "lever",
    "drawer",
    "tray",
    "door",
    "lid",
    "flap",
    "knob",
    "button",
    "sash",
    "wheel",
    "handle",
    "latch",
)  # part words a joint or body name may carry
HOW = {"hinge": "hinged", "slide": "sliding"}
NONE = (
    "Beyond the checks every room gets, the verifiers test no moving parts, containers or carried items in this room."
)
FILES = {
    "inventory.json": "inventory.json lists every object you must build: how many of each category, and each one's role. The roles "
    "are fixed, free (movable), free+joints and fixed+joints (with moving parts).",
    "measurements.json": "measurements.json gives each object's size in metres: the length and width of its smallest footprint rectangle, "
    "longer side first, then its height. Each side is compared with the true one and passes within a factor of "
    f"{G.SIZE_BARS[0]}, larger or smaller. It also gives each moving part's travel from its closed or rest end, in "
    'radians for a hinge and metres for a slide. A part with no stops, such as a wheel, has "no_stops": true '
    "instead of a travel. The values come from the reference room, so size your objects to match them.",
    "depth/": "depth/ gives metric depth for each picture, the extra pictures too (depth/extra_0.json, ...). Each file is a 16 x 16 grid "
    "of z-depth in metres, with null where the ray meets nothing (as above a room without a ceiling). The values come "
    "from the reference room, so place your objects to match them.",
}


def w(c, k=1):
    t = c.replace("_", " ")
    if k == 1 or c in PLURAL:
        return t if k == 1 else PLURAL[c]
    if t.endswith("s") and not t.endswith(("ss", "us")):
        return t  # scissors, pants, chopsticks: plural as they are
    if t.endswith("y") and t[-2:-1] not in "aeiou":
        return t[:-1] + "ies"  # cherry, battery
    if t.endswith(("fe", "lf")):
        return t[:-2] + ("ves" if t.endswith("fe") else "lves")  # knife, shelf
    return t + ("es" if t.endswith(("s", "x", "ch", "sh")) else "s")


def a(xs, sep="or"):
    xs = list(xs)
    return ("an " if xs[0][:1] in "aeiou" else "a ") + listing(xs, sep)  # "an oven or cup"


def listing(xs, sep="and"):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + f" {sep} " + xs[-1]


def part_word(name, kind="hinge"):
    """The part a joint moves, from a whole word of its name (drawer, door, lid, ...); else by how it moves: hinged
    part, sliding part. Whole words only: "sliding" holds no lid. The last such word wins, since names run from the
    object down to the part: bed__0_drawer0_wheel1_axle turns a wheel, not the drawer it hangs under."""
    words = re.split(r"[^a-z]+", name.lower())[::-1]
    return next((w for w in words if w in PARTS), f"{HOW.get(kind, 'moving')} part")


def part_entry(j):
    """One moving part in measurements.json: its travel from the closed or rest end, or no_stops for a joint without
    limits (a wheel, a rotor), whose MuJoCo range is 0 and would read as a part that does not move."""
    e = dict(part=part_word(j["name"], j["type"]), type=j["type"])
    return (
        dict(e, no_stops=True)
        if j["travel"] is None
        else dict(e, travel=j["travel"], unit="rad" if j["type"] == "hinge" else "m")
    )


def objects(case, scene):
    """Per reference object: category, role, oriented size and joints (name, type, travel), from the reference at
    rest."""
    s = Scene(scene, textures=False)
    s.reset(s.q0)
    m = s.m
    out = {}  # no pictures: sizes and joints only
    for o in case["objects"]:
        n = o["slot"]
        dims = np.round(G.oriented_dims(G.surface_samples(s, n)), 4).tolist()
        js = [
            dict(
                name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "",
                type="hinge" if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE else "slide",
                travel=round(float(m.jnt_range[j][1] - m.jnt_range[j][0]), 4) if m.jnt_limited[j] else None,
            )
            for j in s.joints.get(n, [])
        ]
        out[n] = dict(category=o["category"], role=s.role[n], dims=dims, joints=js)
    return out


def l1_text(case, objs):
    """What the verifiers do in this room, named only where a kept unit tests it. Category words stay singular and part
    words plural, so no line says how many of a thing there are. Empty when no kept unit is of these kinds."""
    units = case["functionality"]["units"]
    kinds = collections.defaultdict(list)
    for u in units:
        kinds[u["kind"]].append(u)
    cat = lambda n: objs[n]["category"]
    tested = {u.get("params", {}).get("ref", {}).get("name") for u in kinds["joint"]}
    moving = collections.defaultdict(set)  # category -> {(part word, joint type)}
    for n in dict.fromkeys(u["slots"][0] for u in kinds["joint"]):
        for j in objs[n]["joints"]:
            if j["name"] in tested:
                moving[cat(n)].add((part_word(j["name"], j["type"]), j["type"]))
    lines = []
    if moving:
        part = lambda p, t: w(p, 2) + ("" if p.endswith(" part") else f" ({HOW[t]})")
        lines.append(
            "- Moving parts: "
            + "; ".join(f"{w(c)} {listing(sorted(part(p, t) for p, t in ps))}" for c, ps in sorted(moving.items()))
            + ". The verifiers push or pull each through its whole travel and let go."
        )
    into = kinds["opening"] + kinds["container"]
    if into:
        cs = sorted({cat(u["slots"][0]) for u in into})
        mv = sorted({cat(u["slots"][0]) for u in into if objs[u["slots"][0]]["role"].startswith("free")})
        lines.append(
            f"- Containers: a small ball dropped into {a(w(c) for c in cs)} must stay inside"
            + (f", and {a(w(c) for c in mv)} that is lifted must take it along" if mv else "")
            + ". Build them hollow, with walls and a floor."
        )
    for c, items in sorted(
        {(cat(u["slots"][0]), tuple(sorted({cat(s) for s in u["slots"][1:]}))) for u in kinds["holds"]}
    ):
        lines.append(
            f"- Carrying: {listing(w(i, 2) for i in items)} in {a([w(c)])} must come along when the {w(c)} is lifted. Build them as separate "
            f"free bodies, not merged into the {w(c)}."
        )
    turned = sorted({cat(u["slots"][0]) for u in kinds["turnover"]})
    if turned:
        lines.append(
            f"- Turning over: {a(w(c) for c in turned)} turned upside down must let what is in it or on it fall out, "
            "as the real one does. Build what it holds as separate free bodies, not wedged into its walls."
        )
    if not lines:
        return ""
    return "\n".join(
        [
            "Here is what the verifiers do in this room, on top of the checks every room gets. Which objects they pick, and how",
            "hard and how far they push or pull, stays hidden.",
            "",
        ]
        + lines
    )


def l2_files(objs, scene, cameras):
    """{relative path: JSON}: inventory (counts and roles), measurements (size, tolerance, travel), depth per
    picture."""
    inv = collections.defaultdict(collections.Counter)
    for o in objs.values():
        inv[o["category"]][o["role"]] += 1
    files = {
        "inventory.json": {c: dict(count=sum(r.values()), roles=dict(r)) for c, r in sorted(inv.items())},
        "measurements.json": {
            n: dict(
                size_m=o["dims"],
                size_factor=G.SIZE_BARS[0],
                **({"parts": [part_entry(j) for j in o["joints"]]} if o["joints"] else {}),
            )
            for n, o in objs.items()
        },
    }
    s = Scene(scene, textures=False)
    s.reset(s.q0)
    mujoco.mj_forward(s.m, s.d)
    vis = np.array([1, 1, 1, 0, 0, 0], np.uint8)
    g = np.array([-1], np.int32)
    for c in cameras:
        M = np.array(c["cam_to_world"])
        R, p = M[:3, :3], M[:3, 3]
        K = np.array(c["intrinsics"])
        res = 2 * K[0, 2]
        rows = []
        for i in range(16):
            row = []
            for j in range(16):
                d = R @ np.array(
                    [((j + 0.5) * res / 16 - K[0, 2]) / K[0, 0], -((i + 0.5) * res / 16 - K[1, 2]) / K[1, 1], -1.0]
                )
                u = d / np.linalg.norm(d)
                t = mujoco.mj_ray(s.m, s.d, p, u, vis, 1, -1, g)
                row.append(None if t < 0 else round(float(t * np.dot(u, -R[:, 2])), 2))
            rows.append(row)
        files[f"depth/{c['name']}.json"] = dict(
            note="16 x 16 grid of z-depth in metres along the camera axis at cell centres, row 0 at the top of "
            "the picture; null where the ray meets nothing",
            depth=rows,
        )
    return files


def level(name, case, scene, cameras):
    """(the "What the verifiers do in this room" text, files) of one level: nothing at L0; at L1 l1_text() (or NONE) and
    the three files, which the instruction lists under Input files with FILES' words (make_tasks.instruction())."""
    if name == "l0":
        return "", {}
    objs = objects(case, scene)
    return l1_text(case, objs) or NONE, l2_files(objs, scene, cameras)


def audit(room_dir, case):
    """Declared functions the case did not keep: openings in functions.json, tasks in tasks.json. Empty means every
    declared function is announced and scored."""
    kept = {(u["kind"], u["slots"][0]) for u in case["functionality"]["units"]}
    gaps = []

    def read(name):
        p = os.path.join(room_dir, name)
        if not os.path.exists(p):
            return {}
        with open(p) as f:
            return json.load(f)

    for n, spec in read("functions.json").get("containers", {}).items():
        if spec.get("openings") and ("opening", n) not in kept and ("container", n) not in kept:
            gaps.append(f"opening of {n}")
    kept_tasks = {(u["kind"], *u["slots"][:2]) for u in case["functionality"]["units"]}
    for kind, key in (("pick_place", "robot"), ("push", "robot_push"), ("shelve", "robot_shelve")):
        for x in read("tasks.json").get(kind, []):
            k = (key, x["object"], x["goal"]) if x.get("goal") else (key, x["object"])
            if k not in kept_tasks:
                gaps.append(f"{kind} {x['object']}" + (f" onto {x['goal']}" if x.get("goal") else ""))
    kept_ids = {u["id"] for u in case["functionality"]["units"]}
    gaps += [
        f"activity {i}: {x.get('words', '')[:40]}"
        for i, x in enumerate(read("tasks.json").get("activity", []))
        if f"robot_activity|{i}" not in kept_ids
    ]
    return gaps
