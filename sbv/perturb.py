#!/usr/bin/env python
"""Make a deliberately broken copy of a scene package, to check that each test catches the mistake it is meant to.
Usage: python -m sbv.perturb SOURCE OUT_DIR [--mass BODY K] [--com BODY DZ] [--remove BODY] [--fix BODY] [--weld JOINT]
       [--flip JOINT] [--damping JOINT K] [--range JOINT LO HI] [--solid BODY] [--friction BODY MU] [--move BODY DX DY]
       [--scale K] [--missing-asset]
Edits are applied to scene.xml in this order; other package files are copied unchanged.
"""

import argparse, os, shutil, sys
import xml.etree.ElementTree as ET
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repository root
from sbv.common.case import scene_xml  # noqa: E402
from sbv.common.scene import Scene  # noqa: E402


def find(root, tag, name):
    for el in root.iter(tag):
        if el.get("name") == name:
            return el
    raise SystemExit(f"no <{tag} name='{name}'> in the scene")


def parent_of(root, child):
    return next(p for p in root.iter() if child in list(p))


def nums(s):
    return [float(x) for x in s.split()]


def fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def apply(src_xml, out_xml, a):
    tree = ET.parse(src_xml)
    root = tree.getroot()
    if a.mass or a.com:  # write each link's compiled inertia as an explicit <inertial>: mass scaled by K,
        s = Scene(src_xml)
        s.reset(s.q0)
        ids = {el: i + 1 for i, el in enumerate(root.find("worldbody").iter("body"))}  # or balance
        for n, k, dz in [(n, float(k), 0.0) for n, k in a.mass or []] + [
            (n, 1.0, float(dz)) for n, dz in a.com or []
        ]:  # point raised
            for el in find(root, "body", n).iter("body"):  # DZ metres (MJCF body order = body id)
                bid = ids[el]
                old = el.find("inertial")
                ipos = s.m.body_ipos[bid] + s.d.xmat[bid].reshape(3, 3).T @ [0, 0, dz]
                if old is not None:
                    el.remove(old)
                el.insert(
                    0,
                    ET.Element(
                        "inertial",
                        pos=fmt(ipos),
                        quat=fmt(s.m.body_iquat[bid]),
                        mass=f"{s.m.body_mass[bid] * k:.6g}",
                        diaginertia=fmt(s.m.body_inertia[bid] * k),
                    ),
                )
    if a.remove:  # after the inertia edits, whose body ids come from the original file
        for n in a.remove:
            b = find(root, "body", n)
            parent_of(root, b).remove(b)
    for n in a.fix or []:  # glue an object in place: its free joint goes
        for j in [
            e
            for e in list(find(root, "body", n))
            if e.tag == "freejoint" or (e.tag == "joint" and e.get("type") == "free")
        ]:
            find(root, "body", n).remove(j)
    for n in a.weld or []:
        j = find(root, "joint", n)
        parent_of(root, j).remove(j)
    for n in a.flip or []:
        j = find(root, "joint", n)
        j.set("axis", fmt(-np.array(nums(j.get("axis", "0 0 1")))))
    for n, k in a.damping or []:
        j = find(root, "joint", n)
        j.set("damping", str(float(j.get("damping", 0)) * float(k)))
    for n, lo, hi in a.range or []:  # a joint that travels LO..HI instead (a lift that rises 5 cm, not 23)
        j = find(root, "joint", n)
        j.set("range", f"{float(lo):g} {float(hi):g}")
        j.set("limited", "true")
    if a.solid:  # before friction, so a solid copy keeps a friction edit
        s = Scene(src_xml)
        s.reset(s.q0)
        for n in a.solid:
            body = find(root, "body", n)
            lo, hi = s.aabb(n)
            b = s.objects[n]
            R = s.d.xmat[b].reshape(3, 3)
            p = s.d.xpos[b]
            mass = float(s.m.body_subtreemass[b])
            c = R.T @ ((lo + hi) / 2 - p)
            for g in [e for e in list(body) if e.tag == "geom"]:
                body.remove(g)
            ET.SubElement(
                body, "geom", name=f"{n}_solid", type="box", pos=fmt(c), size=fmt((hi - lo) / 2), mass=str(mass)
            )
    for n, mu in a.friction or []:
        for el in find(root, "body", n).iter("geom"):
            el.set("friction", f"{mu} 0.005 0.0001")
    for n, dx, dy in a.move or []:
        b = find(root, "body", n)
        p = np.array(nums(b.get("pos", "0 0 0")))
        p[:2] += [float(dx), float(dy)]
        b.set("pos", fmt(p))
    if a.scale:
        k = float(a.scale)
        for el in root.iter():
            for key in ("pos", "size", "fromto"):
                if el.get(key) and el.tag in ("body", "geom", "site", "joint", "camera", "light", "inertial"):
                    if not (el.tag == "geom" and el.get("type") == "plane" and key == "size"):
                        el.set(key, fmt(np.array(nums(el.get(key))) * k))
            if el.tag == "joint" and el.get("type") == "slide" and el.get("range"):
                el.set("range", fmt(np.array(nums(el.get("range"))) * k))
            if el.tag == "mesh":
                el.set("scale", fmt(np.array(nums(el.get("scale", "1 1 1"))) * k))
    if a.missing_asset:
        asset = root.find("asset") if root.find("asset") is not None else ET.SubElement(root, "asset")
        ET.SubElement(asset, "mesh", name="sbv_missing", file="sbv_missing_mesh.obj")
    tree.write(out_xml)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("out")
    ap.add_argument("--remove", nargs="*")
    ap.add_argument("--mass", nargs=2, action="append")
    ap.add_argument("--weld", nargs="*")
    ap.add_argument("--flip", nargs="*")
    ap.add_argument("--damping", nargs=2, action="append")
    ap.add_argument("--solid", nargs="*")
    ap.add_argument("--move", nargs=3, action="append")
    ap.add_argument("--scale")
    ap.add_argument("--missing-asset", action="store_true")
    ap.add_argument("--friction", nargs=2, action="append")
    ap.add_argument("--fix", nargs="*")
    ap.add_argument("--com", nargs=2, action="append")
    ap.add_argument("--range", nargs=3, action="append")
    a = ap.parse_args(argv)
    src = scene_xml(a.source)
    if os.path.isdir(a.source):
        shutil.copytree(a.source, a.out, dirs_exist_ok=True)
    else:
        os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "scene.xml")
    apply(src, out, a)
    return out


if __name__ == "__main__":
    print(main())
