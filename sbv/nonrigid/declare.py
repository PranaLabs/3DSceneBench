"""Draft nonrigid.json entries for a reference room's non-rigid objects, as sbv.declare drafts functions.json.

    python -m sbv.nonrigid.declare ROOM_DIR [--out nonrigid.draft.json]

The non-rigid verifiers can only judge what the reference room declares, so this lists every body whose category word is
in DEFORMABLE and has no entry, with a draft entry (kind, a box at the body's measured size and centre, the kind's
default material) for the room's builder to review: fabric, stiffness and density are guesses to replace with real
values. It never writes nonrigid.json itself. Calibration prints the same list as a warning."""

import argparse, json, os, re
import numpy as np

from .schema import side_file

# category word -> kind; fluids inside closed containers (bottled drinks, reagent bottles) stay rigid: the container
# decides their physics
DEFORMABLE = {
    **{
        w: "thin_shell"
        for w in (
            "towel",
            "blanket",
            "curtain",
            "shower_curtain",
            "cloth",
            "jacket",
            "coat",
            "shirt",
            "sock",
            "rug",
            "bath_mat",
            "apron",
            "napkin",
            "tablecloth",
            "bedsheet",
            "duvet",
            "quilt",
            "strap",
        )
    },
    **{
        w: "soft_body"
        for w in (
            "pillow",
            "cushion",
            "mattress",
            "teddy_bear",
            "plush_toy",
            "sponge",
            "bread",
            "bun",
            "burger",
            "sandwich",
            "ottoman",
            "bean_bag",
        )
    },
}
DEFAULT = dict(thin_shell=dict(fabric="cotton"), soft_body=dict(youngs_kpa=30, poisson=0.3, density=200))
word = lambda n: re.sub(r"__\d+$", "", str(n))


def missing(room):
    """[(body name, kind)] for the room's bodies with a deformable word and no nonrigid.json entry."""
    import xml.etree.ElementTree as ET

    side = side_file(room)
    have = {o.get("name") for o in json.load(open(side)).get("objects", [])} if os.path.exists(side) else set()
    try:
        names = [b.get("name") for b in ET.parse(os.path.join(room, "scene.xml")).getroot().iter("body")]
    except (OSError, ET.ParseError):
        return []  # no scene to read: nothing to warn about
    return [
        (n, DEFORMABLE[word(n)])
        for n in names
        if n and re.search(r"__\d+$", n) and word(n) in DEFORMABLE and n not in have
    ]


def draft(room):
    """Draft entries for missing(room). A soft body gets a box at the body's world bounding box (from each geom's own
    box, turned into the world); a cloth's draped shape and rail cannot be read off a rigid body, so its shape is left
    for the builder (null, with a note); "on" is the parent body, left null when the body sits in the world."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(os.path.join(room, "scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    out = []
    for n, kind in missing(room):
        b = m.body(n).id
        pts = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b:
                continue
            c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        pts.append(d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ (c + h * np.array([sx, sy, sz])))
        parent = m.body(m.body_parentid[b]).name
        entry = dict(name=n, kind=kind, on=None if parent in ("", "world") else parent, material=DEFAULT[kind])
        if kind == "thin_shell" or not pts:
            entry.update(
                shape=None,
                review="to author: the draped shape (draped_sheet or mesh with its rail) and what it hangs on; then the fabric",
            )
        else:
            lo, hi = np.min(pts, axis=0), np.max(pts, axis=0)
            entry.update(
                shape=dict(
                    type="box",
                    pos=[round(float(x), 4) for x in (lo + hi) / 2],
                    size=[round(float(x), 4) for x in hi - lo],
                    yaw_deg=0,
                ),
                review="draft: check the box against the object and replace the material with measured values"
                + ("" if entry["on"] else "; name its support"),
            )
        out.append(entry)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("room")
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = draft(a.room)
    print(f"{len(rows)} undeclared non-rigid bodies:", ", ".join(f"{r['name']} ({r['kind']})" for r in rows))
    if a.out:
        json.dump(dict(objects=rows), open(a.out, "w"), indent=1)
