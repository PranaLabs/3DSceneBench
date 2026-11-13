"""Doc Appendix C.4: F1 gates; within F2 and F3 each unit is the mean of its repeats (each repeat the AND of its
predicates); layer = E_s x mean over units; scene = 100 x mean over the layers that have units. Missing targets and
failed interventions are zeros, never a smaller denominator. Each information setting is scored separately.
Beside the layers, two groups of checks: "general", what calibration finds on every object by itself (physics on every
object), and "declared", what the reference room names (functions.json, tasks.json). The headline is unchanged."""

import math
import numpy as np
from sbv.functionality.units import layer, level, eligible, SETTINGS, F4_KINDS


def scope(unit):
    return "declared" if unit.get("named") else "general"


def value(r, key, s):
    """A unit run's score in a setting, 0 to 1: a robot task's stage score when it has one (an activity, the mean over
    its steps, which counts a step half done where the earlier steps-done credit counts it 0); else 1 when it passed,
    else its credit, else 0."""
    if "stage_score" in r:
        return float(r["stage_score"])
    return 1.0 if r.get(key, r["passed"])[s] else float((r.get("credit") or {}).get(s, 0.0))


def scored(E, units, results, s, bar=0):
    """(unit, score 0..1) for each unit that counts in setting s at a bar; an object's door, drawer and key joints count
    as one unit, the share of them that pass. The object's first joint unit stands for it."""
    key = "passed" if bar == 0 else "passed_reference_bar"
    out, joints = [], {}
    for u in units:
        if not eligible(u, s, bar):
            continue
        reps = results.get(u["id"]) or []
        v = float(np.mean([value(r, key, s) for r in reps])) if reps and E else 0.0
        if u["kind"] == "joint":
            joints.setdefault((u.get("slots") or u["id"].split("|")[1:2])[0], []).append(
                (u, v)
            )  # id: joint|object|joint
        else:
            out.append((u, v))
    return out + [(js[0][0], float(np.mean([v for _, v in js]))) for js in joints.values()]


def functionality(E, units, results, bar=0, slots=None):
    """units: the case's F units; results: unit id -> list of per-repeat results; bar: 0 scores every unit at its main
    pass bar, 1 at its reference bar (docs/reference/VERIFIERS.md, "Pass bars"); slots: the reference room's objects,
    for how many of them a declared check covers."""
    out = {}
    for s in SETTINGS:
        per, groups = {}, {}
        for u, v in scored(E, units, results, s, bar):
            per.setdefault(layer(u), []).append(v)
            groups.setdefault(scope(u), []).append(v)
        layers = {g: (E * float(np.mean(v)) if v else None) for g, v in per.items()}
        A = [v for v in layers.values() if v is not None]
        out[s] = dict(
            F1=100.0 * E,
            layers={g: (None if v is None else round(100 * v, 2)) for g, v in sorted(layers.items())},
            units={g: len(v) for g, v in sorted(per.items())},
            functionality=round(100 * float(np.mean(A)), 2) if A else None,
            groups={
                g: dict(score=round(100 * E * float(np.mean(v)), 2), units=len(v)) for g, v in sorted(groups.items())
            },
        )
    if slots is not None:
        covered = sorted({n for u in units if scope(u) == "declared" for n in u["slots"]} & set(slots))
        out["declared_coverage"] = dict(objects=len(slots), covered=len(covered), names=covered)
    return out


def levels(E, units, results, bar=0):
    """{setting: {level: [unit score 0..1]}}: each unit that counts in the setting, at its main pass bar (bar 0) or its
    reference bar (1), under its level (units.level). A unit that never ran, or a copy that fails the gate, scores 0.
    sbv/reward.py adds them up."""
    out = {}
    for s in SETTINGS:
        per = {}
        for u, v in scored(E, units, results, s, bar):
            per.setdefault(level(u), []).append(v)
        out[s] = per
    return out


def parts_points(n):
    """The points of an object with n moving parts, or of a group of n members: 1 + ln(n). One part is one point like
    any other test. Plain ln(n) would give a single door 0 and a two-door cabinet 0.69, less than a cup."""
    return 1.0 + math.log(n) if n > 0 else 0.0


def task_total(unit, reps):
    """A robot task's checkpoints in all: from a result (stage_points), else from its kind and steps."""
    from sbv.functionality.f4_robot.plan import STAGES, capped

    for r in reps or []:
        if r.get("stage_points"):
            return float(r["stage_points"][1])
    if unit["kind"] != "robot_activity":
        return float(len(STAGES[dict(robot="pick_place", robot_push="push", robot_shelve="shelve")[unit["kind"]]]))
    return float(
        capped(0.0, sum(len(STAGES.get((st or {}).get("kind"), (None,))) for st in unit["params"]["steps"]))[1]
    )


def unit_points(E, units, results, s, bar=0):
    """[(unit, points earned, points possible)] for each unit that counts in setting s: a robot task's checkpoints (one
    point each, an activity at most ACTIVITY_CAP); an object's moving parts parts_points(n), earned by the share that
    pass; a group's at-rest unit parts_points(members); every other unit one point. A copy that fails the gate earns
    none."""
    key = "passed" if bar == 0 else "passed_reference_bar"
    out, joints = [], {}
    for u in units:
        if not eligible(u, s, bar):
            continue
        reps = results.get(u["id"]) or []
        v = float(np.mean([value(r, key, s) for r in reps])) if reps and E else 0.0
        if u["kind"] == "joint":
            joints.setdefault((u.get("slots") or u["id"].split("|")[1:2])[0], []).append((u, v))
            continue
        if u["kind"] in F4_KINDS:
            total = task_total(u, reps)
            got = (
                float(np.mean([r["stage_points"][0] for r in reps]))
                if E and reps and all(r.get("stage_points") for r in reps)
                else v * total
            )
            out.append((u, E * got, total))
            continue
        if u["kind"] == "group":  # a group of n small things: 1 + ln(n), like moving parts
            p = parts_points(len(u.get("slots") or [])) or 1.0
            out.append((u, E * v * p, p))
            continue
        out.append((u, E * v, 1.0))
    for js in joints.values():
        p = parts_points(len(js))
        out.append((js[0][0], E * p * float(np.mean([v for _, v in js])), p))
    return out


def level_points(E, units, results, bar=0):
    """{setting: {level: [[points earned, points possible], ...]}}: unit_points() under each unit's level. The reward
    scores a level as its points earned over its points possible."""
    out = {}
    for s in SETTINGS:
        per = {}
        for u, got, total in unit_points(E, units, results, s, bar):
            per.setdefault(level(u), []).append([round(got, 6), round(total, 6)])
        out[s] = per
    return out
