"""Scopes and groups in the score (TRAJ_DRIVEN_FIXES 3.1, 3.2). Every report scores two scopes: "full", every reference
object, and "visible", the objects the task's pictures show (case["scope"]["visible"], written once by this tool from a
coverage file, plus the objects the extra pictures show, sbv/appearance/extra.py). Groups (sbv/groups.py) stand for
their members in both: a member leaves G1, G2, G3 and the F2 units that name only members, and its group enters as one
object (G1 present, G2 count and placed, F2 at rest); a group is visible if any member is. A group word's leftovers
(copy objects of the word that join no group and stand for no single reference object) are one object too, in every
scope: one extra in G1, and one unit in the reported placement shares. In G3 placement an extra (a copy object that
stands for no reference object, leftovers included) is a row only when it fails, in every scope: a thousand 3 cm cubes
that pass add nothing under any word, and a failing object counts once under any name. A unit naming a member beside
another object (a tray holds its fries) stays, the member paired among the members (groups.pairing). In a scope, a
readout or unit naming an object outside it is dropped and counted. Per scope, beside the readouts the reward adds up:
coverage C = paired objects / objects (a group is paired when present), and quality Q = the mean of the paired objects'
readouts (pose and size, at rest, and the input-only units whose objects are all paired), so few objects built well and
many built badly read apart. Each geometry part is a mean weighted by object, weights()): the objects a robot task moves
or places on weigh 2 and an object with moving parts 1 + ln(their number), so a can the robot must pick and a cabinet
with ten drawers count more than a book; G1's paired and true counts become weight sums (an extra weighs 1), shape, pose
and size, and navigation weigh by object, relations by their object a, placement's true rows by the object they stand
for (an extra still costs its shortfall). Pose-and-size also writes its weight sum and passed weight sum (weight,
passed_weight), which sbv/reward.py pose_share() pools with the non-rigid units. The absolute columns, C and Q stay
counts.
Usage: python -m sbv.scope CASE COVERAGE_JSON [--px 25]   visible = the file's "visible" list if it has one, else
max_px_pick >= px (the object's most pixels in one input picture, 512 px)
       python -m sbv.scope CASE --rescore REPORT SCENE    a graded report's scopes with the case's visible set, no physics"""

import argparse, copy, json, math, os
import numpy as np
from sbv.groups import COUNT_PASS
from sbv.functionality.aggregate import functionality, levels, level_points, parts_points
from sbv.functionality.units import layer, eligible, F4_KINDS

PX = 25


def mean(v):
    return float(np.mean(v)) if len(v) else None


def wmean(v):
    return sum(x * w for x, w in v) / sum(w for _, w in v) if len(v) else None  # v: [(value, weight)]


ROBOT_KINDS = F4_KINDS  # the robot tasks: pick-and-place, push, shelving, activities (as the level points count them)


def weights(case):
    """{reference object: its geometry weight}, which follows how much scoring depends on it: 2 for an object a robot
    task moves or places on (the task's slots), 1 + ln(n) for an object with n moving parts, else 1; the largest that
    applies. readouts() weighs a group once, as the larger of its points, 1 + ln(members), and its heaviest member."""
    w = {o["slot"]: 1.0 for o in case["objects"]}
    parts = {}
    for u in (case.get("functionality") or {}).get("units", []):
        if u["kind"] in ROBOT_KINDS:
            for s in u.get("slots") or []:
                if s in w:
                    w[s] = max(w[s], 2.0)
        elif u["kind"] == "joint":
            s = (u.get("slots") or u["id"].split("|")[1:2])[0]
            parts[s] = parts.get(s, 0) + 1
    for s, n in parts.items():
        if s in w:
            w[s] = max(w[s], 1.0 + math.log(n))
    return w


def readouts(case, rep, G, grows, cats, corr, by, units, results, E, keep):
    """One scope. keep: its reference objects (slots); grows: groups.score rows in G's order; cats: copy object ->
    word."""
    mem = {s: g["id"] for g in G for s in g["members"]}
    gin = {g["id"] for g in G if keep & set(g["members"])}
    inside = lambda s: s in (None, "world") or (mem[s] in gin if s in mem else s in keep)
    own = [o["slot"] for o in case["objects"] if o["slot"] not in mem and o["slot"] in keep]
    grp = [r for r in grows if r["id"] in gin]
    paired = [s for s in own if s in corr]
    present = [r for r in grp if r["present"]]
    g, taken = rep["geometry"], {n for v in by.values() for n in v}
    single = {
        corr[s]: s for s in corr if s not in mem
    }  # copy object -> the reference object outside groups it stands for
    ok = set(case["geometry"]["inventory"].get("permitted_extras", []))
    extra = [
        n for n in cats if n not in taken and n not in single and cats[n] not in ok
    ]  # in every scope: built, unasked
    gw = {x["word"] for x in G}  # a group word's leftovers count once, as a group counts once
    extra = [n for n in extra if cats[n] not in gw] + sorted({cats[n] for n in extra if cats[n] in gw})
    W = weights(case)
    gpts = {x["id"]: parts_points(len(x["members"])) or 1.0 for x in G}  # a group counts 1 + ln(members) objects
    gwt = {
        x["id"]: max(gpts[x["id"]], max(W.get(s, 1.0) for s in x["members"])) for x in G
    }  # or as its heaviest member: the larger
    wt = (
        lambda s: gwt[mem[s]] if s in mem else W.get(s, 1.0)
    )  # every geometry part is a weighted mean; extras weigh 1 each
    M, Nr = len(paired) + len(present), len(own) + len(grp)
    Np = M + len(extra)
    Mw, Nrw = (
        sum(map(wt, paired)) + sum(gwt[r["id"]] for r in present),
        sum(map(wt, own)) + sum(gwt[r["id"]] for r in grp),
    )
    G1 = dict(
        f1=1.0 if Np + Nr == 0 else 0.0 if M == 0 else 2 * Mw / (Mw + len(extra) + Nrw),
        paired=M,
        copy=Np,
        true=Nr,
        paired_weight=round(Mw, 6),
        true_weight=round(Nrw, 6),
    )
    shape = [(g["G2"]["per_object"][s]["fscore"], wt(s)) for s in paired if s in g["G2"]["per_object"]]
    gpose = {r["id"]: bool(r["present"] and r["count"] >= COUNT_PASS and r["placed"]) for r in grp}
    prow = {r["slot"]: r for r in g["G2"]["pose_size"]["objects"] if r["slot"] in own}
    pv = (
        lambda r, k: float(r.get("score", r["passed"])) if k == "score" else float(r[k])
    )  # a row's smooth credit; before it, pass or fail
    pose = {
        k: [(pv(r, k), wt(r["slot"])) for r in prow.values()] + [(float(v), gwt[i]) for i, v in gpose.items()]
        for k in ("score", "passed", "passed_reference_bar")
    }
    rels = [r for r in g["G3"]["relations"]["relations"] if r["a"] not in mem]
    rel = [(r["passed"], wt(r["a"])) for r in rels if inside(r["a"]) and inside(r.get("b"))]  # weighed by its object a

    def placed(r):  # a missing reference object, or a copy object by what it stands for
        if r.get("missing"):
            return r["object"] in own
        return r["object"] not in taken and (single[r["object"]] in keep if r["object"] in single else True)

    rows = [r for r in g["G3"]["placement"]["objects"] if placed(r)]
    lw = (
        lambda r: None
        if r.get("missing") or r["object"] in single or cats.get(r["object"]) not in gw
        else cats[r["object"]]
    )
    left = {}  # a group word's leftovers are one unit in the shares, as in G1
    for r in rows:
        if lw(r):
            left.setdefault(lw(r), []).append(r)
    extra = lambda r: not r.get("missing") and r["object"] not in single  # built, and stands for no reference object
    val = lambda r: float(r.get("score", r["passed"]))  # a row on placement()'s slope; before it, pass or fail
    place = [
        (val(r), wt(r["object"] if r.get("missing") else single[r["object"]])) for r in rows if not extra(r)
    ]  # the reference objects' rows, by their weight
    short = sum(
        1 - val(r) for r in rows if extra(r)
    )  # an extra only costs: its shortfall weighs as rows of 0, so no padding lifts
    pscore = (
        sum(x * w for x, w in place) / (sum(w for _, w in place) + short) if place or short else None
    )  # (an extra is a row only when it fails)
    built = [
        {k: r[k] for k in ("support", "non_penetration", "in_bounds")}
        for r in rows
        if not r.get("missing") and not lw(r)
    ] + [
        {k: all(x[k] for x in v) for k in ("support", "non_penetration", "in_bounds")} for v in left.values()
    ]  # the same units
    present_rows = [prow[s] for s in paired if s in prow]
    bar = {r["slot"]: r["size_bars"][0] for r in case["geometry"]["pose_size"]}
    share = lambda v: 100 * sum(v) / len(v) if len(v) else None
    median = lambda v: float(np.median(v)) if len(v) else None
    absolute = dict(
        missing=Nr - M,
        required=Nr,
        extra=Np - M,
        built=Np,
        missing_pct=share([True] * (Nr - M) + [False] * M),
        extra_pct=share([True] * (Np - M) + [False] * M),  # the leaderboard's geometry columns (VERIFIERS.md section 7)
        position_error_median_cm=median([100 * r["position_error"] for r in present_rows]),
        size_over_bar_pct=share([r["size_ratio"] > bar[r["slot"]] for r in present_rows]),
        surface_deviation_median_pct=median(
            [100 * g["G2"]["per_object"][s]["chamfer"] for s in paired if s in g["G2"]["per_object"]]
        ),
        unsupported_pct=share([not r["support"] for r in built]),
        penetrating_pct=share([not r["non_penetration"] for r in built]),
        out_of_bounds_pct=share([not r["in_bounds"] for r in built]),
    )
    nav = [
        (r["passed"], wt(r["goal"])) for r in g["G3"]["navigation"]["routes"] if inside(r["goal"])
    ]  # weighed by its goal
    U = [u for u in units if u["kind"] == "group" or layer(u) != "F2" or not all(s in mem for s in u["slots"])]
    kept = [u for u in U if all(inside(s) for s in u["slots"])]
    val = (
        lambda u: float(np.mean([x["passed"]["input_only"] for x in results[u["id"]]]))
        if E and results.get(u["id"])
        else 0.0
    )
    init = {u["slots"][0]: u for u in kept if u["kind"] == "init"}
    gu = {u["id"]: u for u in kept if u["kind"] == "group"}
    q = dict(
        pose=[pv(prow[s], "score") for s in paired if s in prow] + [gpose[r["id"]] for r in present],
        at_rest=[val(init[s]) for s in paired if s in init] + [val(gu[r["id"]]) for r in present if r["id"] in gu],
        units=[
            val(u)
            for u in kept
            if u["kind"] not in ("init", "group") and eligible(u, "input_only") and all(s in corr for s in u["slots"])
        ],
    )
    return dict(
        objects=Nr,
        singles=len(own),
        groups=len(grp),
        paired=M,
        units=len(kept),
        units_out_of_scope=len(U) - len(kept),
        relations_out_of_scope=len(rels) - len(rel),
        C=M / Nr if Nr else None,
        Q=mean(sum(q.values(), [])),
        Q_parts={k: [round(float(sum(v)), 3), len(v)] for k, v in q.items()},
        geometry=dict(
            G1=dict(inventory=G1),
            G2=dict(
                fscore=wmean(shape),
                matched=len(shape),
                pose_size=dict(
                    score=wmean(pose["score"]),
                    score_passed=wmean(pose["passed"]),
                    score_reference_bar=wmean(pose["passed_reference_bar"]),
                    weight=round(sum(w for _, w in pose["score"]), 6),
                    passed_weight=round(sum(x * w for x, w in pose["score"]), 6),
                ),
            ),
            G3=dict(relations=dict(score=wmean(rel)), placement=dict(score=pscore), navigation=dict(score=wmean(nav))),
            absolute=absolute,
        ),
        functionality=functionality(E, kept, results),
        levels=levels(E, kept, results),
        levels_reference_bar=levels(E, kept, results, bar=1),
        level_points=level_points(E, kept, results),
        level_points_reference_bar=level_points(E, kept, results, bar=1),
    )


def scopes(case, rep, G, grows, cats, corr, by, units, results, E):
    """{"full": ..., "visible": ...}; a case without a visible set scores every object in "visible" too, and says so."""
    slots = {o["slot"] for o in case["objects"]}
    vis = (case.get("scope") or {}).get("visible")
    out = {
        k: readouts(case, rep, G, grows, cats, corr, by, units, results, E, keep)
        for k, keep in (("full", slots), ("visible", slots if vis is None else slots & set(vis)))
    }
    out["visible"]["defined"] = vis is not None
    return out


def rescore(case, rep, scene, same_full=True):
    """A graded report's scopes again, after the case's visible set changed: rows, unit results, pairs and group rows
    come from the report, the group regions' copy objects and the words from the copy at its submitted state. Nothing
    runs again, and the full scope comes out the same (a check; same_full=False: this file's rules changed, so it may
    not)."""
    from sbv import groups as GR
    from sbv.common.scene import Scene
    from sbv.common import case as K

    G = GR.groups(case)
    S = Scene(K.scene_xml(scene), case["physics"], case["vocabulary"], textures=False)
    S.reset(S.q0)
    out = scopes(
        case,
        rep,
        G,
        rep["groups"]["groups"],
        dict(S.cat),
        rep["correspondence"],
        GR.assign(G, S, case["objects"]),
        case["functionality"]["units"] + GR.units(G),
        rep["units"],
        1.0 if rep["F1"]["passed"] else 0.0,
    )
    full = out["full"]
    if (
        "weight" not in rep["scopes"]["full"]["geometry"]["G2"]["pose_size"]
    ):  # a report from before the pose weights were written:
        full = copy.deepcopy(full)  # the check compares the numbers it has
        for k in ("weight", "passed_weight"):
            full["geometry"]["G2"]["pose_size"].pop(k)
    same = json.dumps(full, sort_keys=True) == json.dumps(rep["scopes"]["full"], sort_keys=True)
    if same_full and not same:
        raise ValueError("the full scope changed")
    rep.update(
        scopes=out,
        rescored=dict(
            from_case_hash=rep.get("case_hash"), visible=len(case["scope"]["visible"]), full_changed=not same
        ),
        case_hash=case["case_hash"],
    )
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case")
    ap.add_argument("coverage", nargs="?")
    ap.add_argument("--px", type=float, default=PX)
    ap.add_argument(
        "--rescore",
        nargs=2,
        metavar=("REPORT", "SCENE"),
        help="recompute a graded report's scopes with the case's visible set",
    )
    ap.add_argument(
        "--new-rules",
        action="store_true",
        help="with --rescore: the scope rules changed, so the full scope may change too",
    )
    a = ap.parse_args()
    from sbv.common import case as K

    if a.coverage:
        c = K.load(a.case)
        cov = json.load(open(a.coverage))
        slots = [o["slot"] for o in c["objects"]]
        px = cov.get("max_px_pick", {})
        gone = [
            s for s in (cov["visible"] if "visible" in cov else slots) if s not in (slots if "visible" in cov else px)
        ]
        if gone:
            raise SystemExit(
                f"{a.coverage} and the case disagree on {len(gone)} reference objects ({', '.join(gone[:5])}): another room?"
            )
        vis = cov["visible"] if "visible" in cov else [s for s in slots if px[s] >= a.px]
        from sbv.appearance.extra import visible

        vis = visible(c, vis)  # and what the extra pictures show
        c["scope"] = dict(
            visible=sorted(vis),
            source=os.path.basename(a.coverage),
            rule=f"the file's visible list (px_bar {cov.get('px_bar')})"
            if "visible" in cov
            else f"max_px_pick >= {a.px:g}: the object's most pixels in one input picture at 512 px",
        )
        K.save({k: v for k, v in c.items() if not k.startswith("_") and k != "case_hash"}, a.case)
        print(f"visible: {len(c['scope']['visible'])} of {len(slots)} reference objects, written into {a.case}")
    if a.rescore:
        r = rescore(K.load(a.case), json.load(open(a.rescore[0])), a.rescore[1], not a.new_rules)
        with open(a.rescore[0], "w") as f:
            json.dump(r, f, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
        print(f"rescored {a.rescore[0]}: visible {r['rescored']['visible']} reference objects")
