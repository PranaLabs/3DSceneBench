"""One reward: the mean of appearance (A), geometry (G) and functionality (F), each on 0 to 100; F is the mean of the
four functionality levels. A level is its points earned over its points possible: one point per unit, except a robot
task, one point per checkpoint, and an object's moving parts,
1 + ln(their number) (sbv/functionality/aggregate.py unit_points()).

Parts: appearance, geometry and the four functionality levels (balance, response, affordance, interaction; the room's
own robot tasks are interaction). Non-rigid units count in their parts: a fluid's amount and a soft body's size in
geometry (pose-and-size), a fluid staying in its holder in balance, the hidden actions in response. A room's package
tasks (sbv/package_tasks, F4.7) count their stage share, as the robot tasks do, in the level each one names, interaction
unless it says otherwise, one point per stage like a robot task's checkpoints. A part with nothing to score in this room
is left out. A copy that does not load scores 0 on every part; a copy whose own files cannot be rendered scores 0 on
appearance. Appearance the grader could not compute (no metrics environment, a failed reference-room render) has no
value here, and sbv/pipeline.py reward_file() then writes no reward at all: a grader failure, with the reason in
appearance.status.
The input-only setting is the headline, as before: these tasks give pictures only. Geometry and the levels come from one
scope of the report (sbv/scope.py: "full", every reference object, or "visible", what the pictures show; groups for
their members); a report from before scopes has its per-object numbers read instead."""

import numpy as np

LEVEL_ORDER = ("balance", "response", "affordance", "interaction")
PARTS = ("appearance", "geometry") + LEVEL_ORDER
NONRIGID_PART = dict(
    amount="geometry",
    size="geometry",
    contain="balance",
    fluid="response",
    soft_body="response",
    thin_shell="response",
    nonrigid="response",
    holder="geometry",
    pose="geometry",
)  # one staged unit per non-rigid object, and where a fluid sits
# nd where a soft body or thin shell stands, and its size


def nonrigid_part(row):
    """The part a non-rigid unit (a nonrigid_report.json row) or a package task (a package_report.json row) counts in. A
    non-rigid row from before rows carried their kind is read by its id: S0 was an amount or a size, S1 a hidden
    action."""
    if row.get("kind") == "package":
        return row.get("level") or "interaction"
    from sbv.nonrigid.schema import (
        kind,
    )

    return NONRIGID_PART[kind(row.get("kind")) or ("amount" if row["id"].startswith("S0.") else "fluid")]


MIN_SPAN = 0.05  # the least distance between a floor and the reference room's own value that the mapping stretches over
APPEARANCE_SCORED = (
    "siglip2_100",
    "dinov3_100",
)  # LPIPS out of the score: it compares pixels at the


def appearance(mean, floor=None):
    """Each image score on 0 to 100 beside its raw value. With the room's floor (the reference room's architecture
    alone, scored against the reference room: the case's appearance.floor, sbv/appearance/views.py floor()), a
    similarity is placed between the floor (0) and the reference room's own 1 (100), and the LPIPS distance between the
    floor's distance (0) and 0 (100), clipped at 0; the span is never under MIN_SPAN, so a near-empty room cannot blow
    small differences up. Without a floor, as before: SigLIP-2 and DINOv3 similarities (-1 to 1) as 50 (s + 1); LPIPS as
    100 (1 - d)."""
    f = floor or {}

    def up(s, k):
        return 100 * min(1.0, max(0.0, (s - f[k]) / max(1 - f[k], MIN_SPAN))) if f.get(k) is not None else 50 * (s + 1)

    def down(d):
        return (
            100 * min(1.0, max(0.0, (f["lpips_vgg"] - d) / max(f["lpips_vgg"], MIN_SPAN)))
            if f.get("lpips_vgg") is not None
            else 100 * (1 - min(1.0, d))
        )

    to100 = dict(siglip2=lambda s: up(s, "siglip2"), dinov3=lambda s: up(s, "dinov3"), lpips_vgg=down)
    return {k.replace("_vgg", "") + "_100": round(f_(mean[k]), 1) for k, f_ in to100.items() if mean.get(k) is not None}


def pose_share(g, nonrigid=()):
    """G2 pose-and-size, 0 to 1, with the non-rigid units pooled in. nonrigid: per fluid amount and holder, 0 or 1, and
    per soft body or thin shell pose (before, a soft body size unit, 0 or 1), its smooth credit. Each non-rigid unit is
    one more object of weight 1 beside the rigid objects, which carry their geometry weights (sbv/scope.py weights())
    and earn their smooth credit (g2_shape.py pose_size(); before, pass or fail): (sum of w_i * credit_i over the rigid
    objects + sum of nonrigid) / (sum of w_i + number of nonrigid units). With 20 rigid objects of weight 1, a fluid and
    a cushion, the share is over 22."""
    ps = g["G2"]["pose_size"]
    s = ps["score"]
    if not nonrigid:
        return s
    rows = ps.get("objects")
    if rows is not None:  # a report from before scopes: its rows, at their weight if they carry one (1 if not)
        w = sum(float(r.get("weight", 1.0)) for r in rows)
        p = sum(float(r.get("weight", 1.0)) * float(r.get("score", r["passed"])) for r in rows)
    elif ps.get("weight") is not None:  # a scope (sbv/scope.py readouts()): its weight sums
        w, p = float(ps["weight"]), float(ps["passed_weight"])
    elif s is None:  # no rigid object to score
        w = p = 0.0
    else:  # a scope written before the weight sums: G1's true weight (or count, before weights)
        inv = (g.get("G1") or {}).get(
            "inventory"
        ) or {}  # sums the same objects, since every reference object has a pose-and-size
        w = inv.get("true_weight", inv.get("true"))  # row (a missing one fails); s * w is then the passed weight
        if not w:
            return float(np.mean([s, float(np.mean(nonrigid))]))  # no size known: the mean of the two shares
        w = float(w)
        p = s * w
    return (p + sum(nonrigid)) / (w + len(nonrigid))


def geometry(g, nonrigid=()):
    """The mean of G1 inventory, G2 shape, G2 pose-and-size (pose_share(), with the non-rigid units), G3 relations and
    placement, each 0 to 100. G3 navigation is reported, not scored: it scored 100 on a third of the copies, and
    reaching the objects a robot task needs already counts in F4, whose base paths use the same floor map."""
    vals = [
        g["G1"]["inventory"]["f1"],
        g["G2"]["fscore"],
        pose_share(g, nonrigid),
        g["G3"]["relations"]["score"],
        g["G3"]["placement"]["score"],
    ]
    vals = [v for v in vals if v is not None]
    return round(100 * float(np.mean(vals)), 1) if vals else None


def parts(report, nonrigid_rows=(), setting="input_only", scope="full", bar=0):
    """{part: 0 to 100, or None when the room has nothing to score in it}. nonrigid_rows: nonrigid_report.json "units"
    and package_report.json "units"; bar 1
    reads the levels at the reference pass bars (geometry and the non-rigid rows have no second bar here)."""
    if report.get("geometry") is None:  # the copy does not load: nothing can be measured
        return {p: 0.0 for p in PARTS}
    a = report.get("appearance") or {}
    app = appearance(a.get("mean") or {}, a.get("floor"))
    sc = (report.get("scopes") or {}).get(scope) or report
    pts = (sc.get("level_points_reference_bar" if bar else "level_points") or {}).get(setting)
    if (
        pts is not None
    ):  # points earned and possible per unit: checkpoints for robot tasks, 1 + ln(parts) for moving parts
        per = {k: [(float(g), float(t)) for g, t in v] for k, v in pts.items()}
    else:  # a report from before points: one point per unit
        per = {
            k: [(float(x), 1.0) for x in v]
            for k, v in (sc.get("levels_reference_bar" if bar else "levels") or {}).get(setting, {}).items()
        }
    for r in nonrigid_rows:  # a non-rigid unit that cannot decide does not count; a graded one counts its score
        if not r.get("counted", r.get("decidable", True)):
            continue
        if (
            r.get("kind") == "package"
        ):  # a package task's stages are its checkpoints, as a robot task's are (1 point if not given)
            n = float(r.get("points") or 1.0)
            per.setdefault(nonrigid_part(r), []).append((float(r["score"]) * n, n))
            continue
        per.setdefault(nonrigid_part(r), []).append(
            (float(r["score"]) if r.get("score") is not None else 1.0 if r["passed"] else 0.0, 1.0)
        )
    app = [app[k] for k in APPEARANCE_SCORED if k in app]
    out = dict(
        appearance=round(float(np.mean(app)), 1) if app else 0.0 if a.get("copy_render_failed") else None,
        geometry=geometry(sc["geometry"], [g for g, _ in per.pop("geometry", ())]),
    )
    out.update(
        {
            k: round(100 * sum(g for g, _ in per[k]) / sum(t for _, t in per[k]), 1)
            if per.get(k) and sum(t for _, t in per[k])
            else None
            for k in LEVEL_ORDER
        }
    )  # a level: its points earned over its points possible
    return out


def reward(report, nonrigid_rows=(), setting="input_only", scope="full", bar=0):
    """(reward, parts): the mean of A, G and F (functionality()), over the ones this room has."""
    p = parts(report, nonrigid_rows, setting, scope, bar)
    vals = [v for v in (p["appearance"], p["geometry"], functionality(p)) if v is not None]
    return (round(float(np.mean(vals)), 2) if vals else 0.0), p


def functionality(p):
    """The F score: the mean of the four levels the room has (parts(), one setting), 0 to 100, or None."""
    vals = [p[k] for k in LEVEL_ORDER if p.get(k) is not None]
    return round(float(np.mean(vals)), 1) if vals else None
