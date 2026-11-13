"""The part of a copy's room around one non-rigid object, as MJCF Genesis can load: outlines of every body, the
bodies near the object, and the still and moving pieces written out (contact excludes between kept bodies stay).
"""

import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .schema import bounds


def _lean(scene_xml, keep=None, looks=False, hide=()):
    """The scene's XML tree without picture-only geoms (unless looks), assets nothing uses, and (if keep is given)
    other bodies."""
    r = ET.parse(scene_xml).getroot()
    wb = r.find("worldbody")
    for el in list(wb):  # other bodies go, including any inside <replicate> or <frame> blocks (copy B1, v2)
        if (
            keep is not None
            and el.tag in ("body", "replicate", "frame")
            and not any(b.get("name") in keep for b in el.iter("body"))
        ):
            wb.remove(el)
    for parent in list(wb.iter()):  # picture-only geoms cost load time and memory and do no physics
        for g in list(parent.findall("geom")):
            if g.get("contype") == "0" and g.get("conaffinity") == "0" and (not looks or g.get("name") in hide):
                parent.remove(g)  # with looks, only the moving object's own still picture goes
    used = {g.get(k) for g in r.iter("geom") for k in ("mesh", "material")}
    used |= {
        t
        for m in r.iter("material")
        if m.get("name") in used  # a material names its textures directly or in <layer>s
        for t in [m.get("texture")] + [layer.get("texture") for layer in m.findall("layer")]
    }
    for a in r.findall("asset"):
        for el in list(a):
            if el.tag in ("mesh", "material", "texture") and el.get("name") not in used and el.get("type") != "skybox":
                a.remove(el)
    _absolute_files(r, os.path.dirname(os.path.abspath(scene_xml)))
    return r


CUBE_FILES = ("fileright", "fileleft", "fileup", "filedown", "filefront", "fileback")


def _absolute_files(r, base):
    """Asset paths made absolute the way MuJoCo resolves them, so the written still/actor files load from any folder:
    mesh, hfield and skin files under the compiler's meshdir, textures (and a cube texture's six faces) under its
    texturedir, either falling back to assetdir, all relative to the scene's folder; the directories then go. A copy
    that keeps its meshes in assets/ with <compiler meshdir="assets"> is valid MJCF; joining the bare file to the scene
    folder made its non-rigid tasks and non-rigid units fail with "Error opening file"."""
    comp = r.find("compiler")
    get = (lambda k: comp.get(k)) if comp is not None else (lambda k: None)
    sub = dict(mesh=get("meshdir") or get("assetdir") or "", texture=get("texturedir") or get("assetdir") or "")
    for el in r.iter():
        d = sub["texture"] if el.tag == "texture" else sub["mesh"] if el.tag in ("mesh", "hfield", "skin") else ""
        for k in ("file",) + (CUBE_FILES if el.tag == "texture" else ()):
            if el.get(k):
                el.set(k, os.path.join(base, d, el.get(k)))
    if comp is not None:
        for k in ("assetdir", "meshdir", "texturedir"):
            comp.attrib.pop(k, None)


def _model(scene_xml, keep=None):
    """The room (or just the `keep` bodies) loaded in MuJoCo for measuring: no pictures (the room's textures alone took
    3.7 GB), no contact lists or actuators that name dropped bodies."""
    r = _lean(scene_xml, keep)
    for a in r.findall("asset"):
        for t in a.findall("texture"):
            a.remove(t)
    for m in r.iter("material"):
        m.attrib.pop("texture", None)
        for layer in list(m.findall("layer")):
            m.remove(layer)
    for tag in ("contact", "equality", "tendon", "actuator", "sensor", "keyframe") if keep is not None else ():
        for el in r.findall(tag):
            r.remove(el)
    m = mujoco.MjModel.from_xml_string(ET.tostring(r, encoding="unicode"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def _aabbs(scene_xml, with_corners=False):
    """Each top-level body's box in the room; with with_corners=True also the corner points of all its collision
    pieces."""
    m, d = _model(scene_xml)
    box, pts = {}, {}
    for g in range(m.ngeom):
        if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            continue
        b = m.body_rootid[m.geom_bodyid[g]]
        name = m.body(b).name if b else "world"
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        # The piece's own box (geom_aabb, in its frame) turned into the room's frame; bounding spheres put a tall
        # holder's bottom 5 cm under its real base, so the tilt hinge sat inside its support
        corners = (
            np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * m.geom_aabb[g][3:]
            + m.geom_aabb[g][:3]
        )
        world = corners @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        lo, hi = world.min(axis=0), world.max(axis=0)
        pts.setdefault(name, []).append(world)
        a = box.setdefault(name, [lo.copy(), hi.copy()])
        a[0], a[1] = np.minimum(a[0], lo), np.maximum(a[1], hi)
    return (box, {k: np.concatenate(v) for k, v in pts.items()}) if with_corners else box


def cut(scene_xml, keep, actor, out_dir, looks=False, hide=()):
    """Write two MJCF files: the held-still neighbourhood, and the actor (or None)."""
    paths = {}
    for which in ("still", "actor"):
        if which == "actor" and not actor:
            continue
        want = (keep - {actor}) if which == "still" else {actor}
        r = _lean(scene_xml, want, looks, hide)
        wb = r.find("worldbody")
        if which == "still":  # everything near the object is held still
            for b in wb.findall("body"):
                for j in list(b.findall("freejoint")) + [j for j in b.findall("joint") if j.get("type") == "free"]:
                    b.remove(j)
        if which == "actor":
            for g in list(wb.findall("geom")):
                wb.remove(g)
        for tag in ("equality", "tendon", "actuator", "sensor", "keyframe"):
            for el in r.findall(tag):
                r.remove(el)
        for c in r.findall("contact"):
            for e in list(c):
                if e.get("body1") not in want or e.get("body2") not in want:
                    c.remove(e)
        paths[which] = os.path.join(out_dir, f"{which}.xml")
        ET.ElementTree(r).write(paths[which])
    return paths


def neighbourhood(scene_xml, obj, margin=0.25):
    boxes = _aabbs(scene_xml)
    lo, hi = bounds(obj, os.path.dirname(os.path.abspath(scene_xml)))
    lo, hi = np.asarray(lo) - margin, np.asarray(hi) + margin
    keep = {n for n, (a, b) in boxes.items() if np.all(a <= hi) and np.all(b >= lo) and n != "world"}
    keep |= {n for n in boxes if n.startswith(("room_geometry", "wall", "floor"))}
    if obj["on"] != "floor":
        keep.add(obj["on"])
    return keep, boxes


def prune(still_xml, lo, hi, margin=0.02):
    """Drop the held-still collision pieces lying wholly outside the box lo..hi grown by margin. A fluid is kept inside
    its box, so it cannot touch them, and Genesis checks every particle against every piece."""
    m, d = _model(still_xml)
    far = set()
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE or not m.geom(g).name:
            continue
        corners = (
            np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * m.geom_aabb[g][3:]
            + m.geom_aabb[g][:3]
        )
        world = corners @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        if np.any(world.max(axis=0) < np.asarray(lo) - margin) or np.any(world.min(axis=0) > np.asarray(hi) + margin):
            far.add(m.geom(g).name)
    tree = ET.parse(still_xml)
    for parent in list(tree.getroot().iter()):
        for el in list(parent.findall("geom")):
            if el.get("name") in far:
                parent.remove(el)
    tree.write(still_xml)
    return len(far)
