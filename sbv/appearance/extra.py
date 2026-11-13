"""Extra pictures: cameras aimed at the reference objects the main pictures do not show.

Every room has 8 main pictures (input_0-7) and up to MAX extra ones (extra_0 ..), at most 24 in all with the 8 main
ones. The extras come in their own folder of the task, for an entrant who finds the main ones not enough. They are
chosen after the held-out cameras are fixed, so the held-out cameras never move for them: each extra camera is aimed at
the objects no earlier picture shows, the robot tasks' objects first (views.cover), from free floor where a camera may
stand (heldout.placement), apart from every held-out camera by the held-out rule (heldout.separated(): 0.5 m, or 15
degrees about the room centre), so every held-out camera still stands apart from every picture an entrant gets, and
views.SPACING from the earlier extras. Extras stop when nothing new can be shown; objects no camera can show (inside a
closed drawer) are listed as unseen.

A room may bring its own extras instead: extra_cameras.json beside scene.xml (FILE), cameras named extra_0 .. in the
record format of views.camera(), each placeable and separated from every held-out camera. Appearance never scores the
extra pictures; the objects they show join the visible scope (sbv/scope.py)."""

import json, os
import numpy as np
from sbv.appearance import views as V
from sbv.appearance import heldout as H
from sbv.geometry.g3_layout import floor_plan

PROTOCOL, SUPPLIED = "extra-cover-v1", "extra-supplied-v1"
FILE = "extra_cameras.json"
MAIN = V.CONDITIONS["8"]  # the main pictures
MAX = 16  # extra pictures at most: 24 pictures in all
HEIGHT = 1.6  # as the input cameras


def main_cameras(appearance):
    """The case's main input cameras, input_0-7; ValueError for a case with fewer."""
    cams = [c for c in appearance.get("input_cameras", []) if c["name"] in MAIN]
    if len(cams) != len(MAIN):
        raise ValueError(
            f"the main pictures need {', '.join(MAIN)}; the case has {len(cams)} of them (sbv.add_input_cameras)"
        )
    return cams


def make(scene, bounds, appearance, n=MAX):
    """Up to n extra cameras for the room's main pictures and held-out cameras."""
    main, held = main_cameras(appearance), appearance["cameras"]
    hfov, res = main[0]["hfov_deg"], appearance.get("resolution", 512)
    xs, ys, free = floor_plan(scene, bounds, 0.25)
    cells = np.argwhere(free)
    pts = np.c_[xs[cells[:, 0]], ys[cells[:, 1]]]
    plan = floor_plan(scene, bounds, H.FLOOR_RADIUS_M)
    center = np.mean(np.asarray(bounds, float), axis=0)[:2]

    def ok(pos, look):  # as serialized
        p = pos.round(3)
        return not H.separated(p, held, center) and H.placement(scene, bounds, p, look, plan) is None

    out = []
    for shot in V.cover(scene, pts, main, [], HEIGHT, hfov, n=n, ok=ok):
        if shot is None:
            break
        out.append(V.camera(f"extra_{len(out)}", shot[0], shot[1], hfov, res))
    return out


def validate(scene, bounds, doc, held, resolution=512):
    """The room's own extra cameras (FILE), checked on the reference room; ValueError naming every problem."""
    raw = doc.get("cameras") if isinstance(doc, dict) else None
    if not isinstance(raw, list):
        raise ValueError(f'{FILE} needs a list under "cameras"')
    names = [c.get("name") if isinstance(c, dict) else None for c in raw]
    if len(raw) > MAX or sorted(names, key=str) != sorted((f"extra_{k}" for k in range(len(raw))), key=str):
        raise ValueError(f"{FILE}: at most {MAX} cameras named extra_0 .. extra_n-1, found {names}")
    cams = sorted(({k: c[k] for k in H.RECORD if k in c} for c in raw), key=lambda c: int(c["name"].split("_")[1]))
    plan = floor_plan(scene, bounds, H.FLOOR_RADIUS_M)
    center = np.mean(np.asarray(bounds, float), axis=0)[:2]
    problems = []
    for c in cams:
        why = H.record_problem(c, resolution) or H.placement(scene, bounds, c["position"], c["look_at"], plan)
        close = H.separated(c["position"], held, center)
        if close:
            why = (
                why + "; " if why else ""
            ) + f"under {H.SEPARATION_M} m and {H.SEPARATION_DEG} degrees from held-out {', '.join(close)}"
        if why:
            problems.append(f"{c['name']}: {why}")
    if problems:
        raise ValueError(f"the room's own extra cameras are not valid in the reference room: {'; '.join(problems)}")
    return cams


def supplied(room):
    """The room's own extra cameras document (FILE beside its scene.xml), or None."""
    folder = room if os.path.isdir(room) else os.path.dirname(os.path.abspath(room))
    p = os.path.join(folder, FILE)
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


def add(appearance, scene, bounds, room):
    """The appearance block with the room's extra cameras (extra_cameras) and what they add (extra_protocol: what each
    shows, and the objects no picture shows). room: the reference room's scene.xml."""
    a = dict(appearance)
    doc = supplied(room)
    extras = validate(scene, bounds, doc, a["cameras"], a.get("resolution", 512)) if doc else make(scene, bounds, a)
    look = lambda c: V.shown(
        scene, np.array(c["position"], float), np.array(c["look_at"], float), c["hfov_deg"], scene.objects
    )
    seen = set().union(*(look(c) for c in main_cameras(a)))
    shows = {c["name"]: sorted(look(c)) for c in extras}
    more = set().union(*map(set, shows.values()))
    a["extra_cameras"] = extras
    a["extra_protocol"] = dict(
        protocol=SUPPLIED if doc else PROTOCOL,
        max=MAX,
        shows=shows,
        adds=sorted(more - seen),
        unseen=sorted(set(scene.objects) - seen - more),
    )
    return a


def visible(case, vis):
    """A visible scope list with the objects the extra pictures show added (sbv/scope.py)."""
    shows = (case.get("appearance", {}).get("extra_protocol") or {}).get("shows", {})
    slots = {o["slot"] for o in case.get("objects", [])}
    return sorted(set(vis) | (set().union(*map(set, shows.values())) & slots) if shows else set(vis))
