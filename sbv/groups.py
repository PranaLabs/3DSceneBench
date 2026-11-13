"""Groups: many small things of one kind in one container (fries in a tray, cups in a stack, test tubes in a rack) are
scored as one object, not one by one. A copy should show the right kind of thing, about as many, in the right place, and
staying there; which fry lies where is not a property of the room worth scoring. Per group:
  present   at least one copy object of the word stands in the group's region
  count     min(n, N) / max(n, N) of copy members n and true members N (passes at >= COUNT_PASS)
  placed    the copy members' centroid lies within max(FLOOR, half the region's diagonal) of the true centroid
  at_rest   after the scene settles, at least REST_PASS of the copy members are still in the region
Region: the box around the true members' centres, grown by GROW on every side. A copy object counts for at most one
group, and only when its nearest reference object of its word is a member (assign). Members leave the per-object
readouts.
Usage: python -m sbv.groups CASE SCENE.xml [--no-settle]"""

import argparse, json, math
from collections import defaultdict
import numpy as np
from sbv.common import case as K

MIN_N, MAX_DIAG, GROW, FLOOR = (
    5,
    0.3,
    0.15,
    0.3,
)  # GROW 0.15 m: about a container wall plus a cup
COUNT_PASS, REST_PASS = 0.7, 0.9


def word(slot):
    """The word a slot is scored as: mug__3 is a cup, as its copy object is (scene.category), so six mugs in a rack are
    one group of cups and a copy's cups can stand for them."""
    from sbv.common.scene import main_word

    return main_word(slot.lower().split("__")[0])


def groups(case, min_n=MIN_N, max_diag=MAX_DIAG):
    """Group = reference objects of one word, each under max_diag m, under one container (support_pairs; a stack climbs
    to its first other-word support), at least min_n of them. Floor and world never group."""
    par = {}
    for a, b in case.get("support_pairs", []):
        par.setdefault(b, a)

    def root(s):
        for _ in range(500):
            if s in par and word(par[s]) == word(s):
                s = par[s]
            else:
                break
        return par.get(s, "floor")

    O = {o["slot"]: o for o in case["objects"]}
    G = defaultdict(list)
    for s, o in O.items():
        r = root(s)
        if math.hypot(*o["dims"]) < max_diag and r not in ("floor", "world"):
            G[(word(s), r)].append(s)
    out = []
    for (w, r), m in sorted(G.items()):
        if len(m) < min_n:
            continue
        c = np.array([O[s]["center"] for s in m])
        h = np.array([O[s]["dims"] for s in m]).max(0) / 2 + GROW
        lo, hi = c.min(0) - h, c.max(0) + h
        out.append(
            dict(
                id=f"group|{w}|{r}",
                word=w,
                container=r,
                members=sorted(m),
                n=len(m),
                centroid=c.mean(0).round(4).tolist(),
                lo=lo.round(4).tolist(),
                hi=hi.round(4).tolist(),
            )
        )
    return out


def inside(p, g):
    return bool((np.asarray(p) >= g["lo"]).all() and (np.asarray(p) <= g["hi"]).all())


def assign(G, scene, case_objects):
    """Copy object -> group: a copy object of a group's word whose nearest reference object of that word, members and
    single objects alike, is a member of the group, and that stands inside the group's region. One nearest a single
    reference object is left for the single pairing; adjacent pans or racks have overlapping regions, so the nearest
    member decides between them."""
    gid = {s: g for g in G for s in g["members"]}
    words = {g["word"] for g in G}
    by = defaultdict(list)
    pts = defaultdict(list)
    for o in case_objects:
        if word(o["slot"]) in words:
            pts[word(o["slot"])].append((np.asarray(o["center"]), gid.get(o["slot"])))
    for n in scene.objects:
        w = scene.cat[n]
        if w not in pts:
            continue
        p = scene.center(n)
        d, g = min(((float(np.linalg.norm(p - c)), g) for c, g in pts[w]), key=lambda x: x[0])
        if g is not None and inside(p, g):
            by[g["id"]].append(n)
    return by


def pairing(slots, scene, G):
    """(slot -> copy object, group id -> its copy objects). The objects outside groups pair as before (case.correspond)
    on the copy objects no group took; the members then pair the same way among themselves on the copy objects left.
    Members are never scored one by one: their pairs only let a unit that names a member beside another object run on
    the copy's own things (a tray holds its fries; a robot task's surroundings)."""
    if not G:
        return K.correspond(slots, scene), {}
    by = assign(G, scene, slots)
    mem = {s for g in G for s in g["members"]}
    took = {n for v in by.values() for n in v}
    out = K.correspond([o for o in slots if o["slot"] not in mem], scene, [n for n in scene.objects if n not in took])
    used = set(out.values())
    out.update(K.correspond([o for o in slots if o["slot"] in mem], scene, [n for n in scene.objects if n not in used]))
    return out, by


def units(G):
    """One F2 unit per group, its at-rest check (units.py: kind "group", level balance), standing for its members'
    F2.1."""
    return [dict(id=g["id"], kind="group", slots=g["members"]) for g in G]


def score(G, scene, case_objects, settle=True, by=None):
    """present, count and placed at the scene's current state; at_rest after settling (settle=False: left out, see
    rest)."""
    by = assign(G, scene, case_objects) if by is None else by
    rows = []
    for g in G:
        m = by.get(g["id"], [])
        n, N = len(m), g["n"]
        row = dict(id=g["id"], true=N, copy=n, present=n > 0, count=round(min(n, N) / max(n, N), 3) if n else 0.0)
        if n:
            c = np.mean([scene.center(x) for x in m], axis=0)
            reach = max(FLOOR, 0.5 * float(np.linalg.norm(np.subtract(g["hi"], g["lo"]))))
            row["centroid_error"] = round(float(np.linalg.norm(c - np.asarray(g["centroid"]))), 3)
            row["placed"] = row["centroid_error"] <= reach
        else:
            row["placed"] = False
        rows.append(row)
    if settle:
        scene.settle()
        rest(G, rows, scene, by)
    return summarize(G, rows)


def rest(G, rows, scene, by):
    """at_rest per group row, at the scene's settled start (Scene.settle, or the grader's seed-0 F2.1 rollout)."""
    scene.reset()
    for g, row in zip(G, rows):
        m = by.get(g["id"], [])
        stay = sum(inside(scene.center(x), g) for x in m)
        row["at_rest"] = bool(m) and stay / len(m) >= REST_PASS
        row["stayed"] = f"{stay}/{len(m)}"


def summarize(G, rows):
    for row in rows:
        checks = [row["present"], row["count"] >= COUNT_PASS, row["placed"]] + (
            [row["at_rest"]] if "at_rest" in row else []
        )
        row["score"] = round(100 * (sum(checks) / len(checks)), 1)
        row["passed"] = all(checks)
    return dict(
        groups=rows,
        score=round(float(np.mean([r["score"] for r in rows])), 2) if rows else None,
        members_out=sum(g["n"] for g in G),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("scene")
    ap.add_argument("--no-settle", action="store_true")
    a = ap.parse_args()
    from sbv.common.scene import Scene

    case = K.load(a.case)
    G = groups(case)
    S = Scene(a.scene, case["physics"], case["vocabulary"])
    S.reset(S.q0)
    print(json.dumps(score(G, S, case["objects"], not a.no_settle), indent=1))
