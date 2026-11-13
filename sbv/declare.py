#!/usr/bin/env python
"""Draft a reference room's functions.json from the room itself, for a person to review.

Every object named by a container word (declared.CONTAINER_WORDS) is tried in the settled reference room:
  opening   the top of its box, pointing down, with the radius of the clear space 1 cm under the rim. Declared when
            verifier F3.2 passes there now: the 2 cm ball goes in, stays, and comes along when the container is lifted.
  holds     the free objects whose centre lies inside its box. Those that come along when it is lifted 10 cm (F3.5).
Objects that fail go under "not_containers" with the reason, so calibration's warning stays quiet and the reviewer sees
why (a closed box, a lidded kettle, a vase whose neck is narrower than the ball). Entries already in the file are kept
as they are; the draft only adds. Calibration re-checks every declaration with jitter and drops what the reference room
fails.

Usage: python -m sbv.declare REFERENCE (scene dir or .xml) [--write] [--out PATH]
       without --write the draft is printed; --write saves it as functions.json next to scene.xml (or PATH)
       python -m sbv.declare REFERENCE --check
       exits 1 when calibrate would stop: container-like objects and no functions.json (reads names only, no physics)
"""

import argparse, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repository root
from sbv.common import case as K  # noqa: E402
from sbv.common.scene import Scene, BALL_R, Unstable  # noqa: E402
from sbv.functionality import declared as D, f3_functions as Fn, f3_calibrate as CF  # noqa: E402
from sbv.functionality.units import LOCAL_M  # noqa: E402

BELOW_RIM = 0.01  # where the opening's radius is measured, under the top of the box (m)
MIN_R = BALL_R + 0.002  # declared.check's smallest opening


GRID, STEP = 25, 0.005  # at most GRID rays per side over the footprint, at least STEP apart (m)


def hollow(scene, n):
    """The centre (x, y) of n's hollow seen from above: rays straight down on a grid over its footprint; a ray counts
    when it first meets n itself at least 1 cm under n's top, and n walls in the point 1.5 cm above that hit on all 8
    horizontal sides (the walled-in rule of F3.2). None when no ray does (a solid block, a closed box, a flat tray)."""
    lo, hi = scene.aabb(n)
    b = scene.objects[n]
    reach = float(np.linalg.norm(hi[:2] - lo[:2]))
    pts = []
    k = lambda a, z: np.linspace(a, z, int(min(GRID, max(2, (z - a) / STEP))) + 2)[1:-1]
    dirs = [np.array([math.cos(t), math.sin(t), 0.0]) for t in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    for x in k(lo[0], hi[0]):
        for y in k(lo[1], hi[1]):
            depth, owner = Fn.solid_hit(scene, np.array([x, y, hi[2] + 0.01]), np.array([0, 0, -1.0]), -1)
            if depth is None or owner != b or depth < 0.02:
                continue
            p = np.array([x, y, hi[2] + 0.01 - depth + 0.015])
            if all((h := Fn.solid_hit(scene, p, u, -1))[1] == b and h[0] <= reach for u in dirs):
                pts.append((x, y))
    return np.mean(pts, axis=0) if pts else None


def opening(scene, n, at="hollow"):
    """The top opening of n in its own frame, or (None, why): over its deepest hollow (a long handle pulls the centre of
    its box off a cup), or (at="box") over the centre of its box."""
    scene.reset(scene.start)
    d = scene.d
    b = scene.objects[n]
    R, o = d.xmat[b].reshape(3, 3), d.xpos[b]
    lo, hi = scene.aabb(n)
    ext = hi - lo
    c = hollow(scene, n) if at == "hollow" else (lo + hi)[:2] / 2
    if c is None:
        return None, "seen from above it has no hollow that walls a ball in"
    top = np.array([c[0], c[1], hi[2]])
    q = top - [0, 0, min(BELOW_RIM, ext[2] / 2)]
    reach = []
    for t in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        x, owner = Fn.solid_hit(scene, q, np.array([math.cos(t), math.sin(t), 0.0]), -1)
        reach.append(x if x is not None and owner == b else math.inf)
    r = min(min(reach), min(ext[:2]) / 2) - 0.002
    if r < MIN_R:
        return None, f"the clear space under its rim is {200 * max(r, 0):.1f} cm across, too narrow for the 2 cm ball"
    clean = lambda v: [float(x) + 0.0 for x in np.round(v, 4)]  # + 0.0: no -0.0 in the file
    return dict(pos=clean(R.T @ (top - o)), dir=clean(R.T @ [0, 0, -1.0]), radius=round(float(r), 4)), None


def inside(scene, n):
    """Free objects other than n that reach into n at the settled start: the centre of mass over n's footprint, the
    bottom below n's top (a pencil half out of its cup reaches in; a cup on a tray's rim does too, and the lift
    decides)."""
    scene.reset(scene.start)
    lo, hi = scene.aabb(n)
    return [
        i
        for i, b in scene.objects.items()
        if i != n
        and scene.free_dof(i) is not None
        and np.all(scene.d.xipos[b][:2] >= lo[:2])
        and np.all(scene.d.xipos[b][:2] <= hi[:2])
        and scene.aabb(i)[0][2] < hi[2] - 0.005
    ]


def propose(scene, n):
    """(entry, None) for a container, or (None, why not)."""
    entry = {}
    full = scene
    scene = scene.subset([n], LOCAL_M) or scene  # a small model of what lies near n
    with scene.focus([n], LOCAL_M):
        why = None
        for at in ("hollow", "box"):
            op, w = opening(scene, n, at)
            if op:
                try:
                    u, w = CF.cal_opening(scene, n, 0, op)
                except Unstable:
                    u, w = None, "the reference room's run is numerically unstable"
                if u:
                    entry["openings"] = [op]
                    break
            why = why or w  # the first attempt's reason says most
        items = inside(scene, n) if scene.free_dof(n) is not None else []
    if items:
        scene = full.subset([n] + items, LOCAL_M) or full
        with scene.focus([n] + items, LOCAL_M):
            try:
                came = Fn.holds_once(scene, n, items, scene.start)[0]
            except Unstable:
                came = [False] * len(items)
            held = [i for i, c in zip(items, came) if c]
            if held:
                try:
                    u, _ = CF.cal_holds(scene, n, held)
                except Unstable:
                    u = None
                if u:
                    entry["holds"] = held
    if entry:
        return entry, None
    if why is None or why.startswith("declared, but the reference room fails it: reference outcome"):
        why = "the 2 cm ball does not go in and stay (verifier F3.2 fails in the reference room)"
    return None, why.replace("declared, but the reference room fails it: ", "")


def draft(ref, log=print):
    xml = K.scene_xml(os.path.abspath(ref))
    spec = D.load(xml)
    out = dict(spec, containers=dict(spec.get("containers", {})), not_containers=dict(spec.get("not_containers", {})))
    scene = Scene(xml, probes=True, textures=False)
    scene.settle()
    for n in D.container_like(scene.objects):
        if n in out["containers"] or n in out["not_containers"]:
            log(f"{n}: already in the file, kept")
            continue
        entry, why = propose(scene, n)
        if entry:
            out["containers"][n] = entry
            log(
                f"{n}: declared"
                + (f", opening {200 * entry['openings'][0]['radius']:.1f} cm across" if "openings" in entry else "")
                + (f", holds {', '.join(entry['holds'])}" if "holds" in entry else "")
            )
        else:
            out["not_containers"][n] = why
            log(f"{n}: not a container: {why}")
    if not out["not_containers"]:
        out.pop("not_containers")
    return xml, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out")
    ap.add_argument(
        "--check", action="store_true", help="only say whether calibrate would stop for a missing functions.json"
    )
    a = ap.parse_args()
    if a.check:
        xml = K.scene_xml(os.path.abspath(a.reference))
        names = D.object_names(xml)
        why = D.missing_file(xml, names)
        if why:
            sys.exit("functions.json: " + why)
        left = [
            n
            for n in D.container_like(names)
            if n not in set(D.load(xml).get("containers", {})) | set(D.load(xml).get("not_containers", {}))
        ]
        print(
            "functions.json: "
            + ("present" if os.path.exists(D.path(xml)) else "not needed, no container-like objects")
            + (f"; neither declared nor under not_containers: {', '.join(left)}" if left else "")
        )
        sys.exit(0)
    xml, spec = draft(a.reference, log=lambda s: print(s, file=sys.stderr))
    text = json.dumps(spec, indent=1) + "\n"
    if a.write or a.out:
        with open(a.out or D.path(xml), "w") as f:
            f.write(text)
        print("wrote", a.out or D.path(xml), file=sys.stderr)
    else:
        print(text, end="")
