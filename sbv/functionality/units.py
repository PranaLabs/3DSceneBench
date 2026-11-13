"""One interface over every F2 and F3 unit: run it in a scene through the correspondence, and decide pass or fail in
each information setting. Layers: F2 for physical bundles, F3 for object functions (doc Appendix C.1)."""

from sbv.functionality import f2_physical as P, f3_functions as Fn

RETIRED = {
    "load",
    "chain",
    "remove",
}
# Not calibrated, and units of these kinds in older cases are not scored.
F2_KINDS = {
    "init",
    "push",
    "tilt",
    "slide",
    "remove",
    "edge",
    "chain",
    "group",
}  # group: sbv/groups.py, judged by the grader
F4_KINDS = {
    "robot",
    "robot_push",
    "robot_shelve",
    "robot_activity",
}  # pick-and-place, contact-rich push, shelving, an activity of steps
# Cases calibrated with these run rules judge every rollout by Monitor.ok_rules (one limit for every room); older cases
# keep the per-unit penetration tolerance. R1's sinking limit is off where the test makes things fall and land.
RULES, FALLS = "R1-sink5mm-0.3s/R2", {"push", "remove", "chain", "turnover"}
# Tests run locally (Scene.focus) on their own objects: doors and drawers keep the whole room (their opening sweep is a
# geometry check against every part), the settle test is the whole room by nature, robot tasks build their own model.
LOCAL_KINDS = {
    "push",
    "tilt",
    "slide",
    "remove",
    "edge",
    "chain",
    "grasp",
    "load",
    "container",
    "opening",
    "holds",
    "turnover",
}
LOCAL_M = 0.3  # the radius of a local run, metres


def zone(unit):
    """The objects a local run is centred on, the same in calibration and scoring: a support removal or push chain on
    all its objects (calibrated as one event), every other test on its own object (calibrated in that object's job),
    even when it also names the object's support (tilt, slide)."""
    if unit["kind"] in ("remove", "chain"):  # an event: all its objects, and a removal's control neighbour
        return list(unit["slots"]) + [c for c in [unit.get("params", {}).get("control")] if c]
    return unit["slots"][:1]


# The functionality levels the reward adds up.
LEVELS = dict(
    init="balance",
    group="balance",
    remove="balance",
    edge="balance",
    push="response",
    tilt="response",
    slide="response",
    chain="response",
    joint="affordance",
    container="affordance",
    opening="affordance",
    grasp="affordance",
    load="affordance",
    holds="affordance",
    turnover="affordance",
    robot="interaction",
    robot_push="interaction",
    robot_shelve="interaction",
    robot_activity="interaction",
)


def level(unit):
    return LEVELS[unit["kind"]]


SETTINGS = ("physics_hint", "input_only")
NUMBERS = {
    "init": "F2.1",
    "group": "F2.1",
    "push": "F2.2",
    "tilt": "F2.3",
    "slide": "F2.4",
    "remove": "F2.5",
    "edge": "F2.6",
    "chain": "F2.7",
    "joint": "F3.1",
    "container": "F3.2",
    "opening": "F3.2",
    "grasp": "F3.3",
    "load": "F3.4",
    "holds": "F3.5",
    "turnover": "F3.6",
    "robot": "F4.1",
    "robot_push": "F4.2",
    "robot_shelve": "F4.3",
    "robot_activity": "F4.5",
}  # docs/reference/VERIFIERS.md


def layer(unit):
    return "F2" if unit["kind"] in F2_KINDS else "F4" if unit["kind"] in F4_KINDS else "F3"


def number(unit):
    """The verifier number of a unit, as docs/reference/VERIFIERS.md lists them."""
    return NUMBERS[unit["kind"]]


def eligible(unit, setting, bar=0):
    """Whether the unit counts in a setting, at the main pass bar (0) or the reference bar (1)."""
    hint = (
        unit.get("hint_only", False) if bar == 0 else unit.get("hint_only_reference_bar", unit.get("hint_only", False))
    )
    return setting == "physics_hint" or not hint


def robot_trial(kind):
    """The trial function of an F4 unit kind."""
    from sbv.functionality.f4_robot import pick_place, push, shelve, activity

    return dict(robot=pick_place, robot_push=push, robot_shelve=shelve, robot_activity=activity)[kind].trial


def run(scene, unit, corr=None, qpos=None, seed=0):
    """Outcome and monitors for one non-init unit. A robot task builds its own room-plus-robot model and applies the
    repeat's jitter (seed) there."""
    if unit["kind"] in F4_KINDS:
        return robot_trial(unit["kind"])(scene, unit, corr, seed)
    return (P.trial if unit["kind"] in F2_KINDS else Fn.trial)(scene, unit, corr, qpos)


def judge(unit, outcome, monitors):
    """Serializable result with the pass decision in each setting, at the main pass bar ("passed") and at the reference
    bar ("passed_reference_bar"); a unit without bars gets the same decision for both."""
    ok = (
        (lambda mo: mo.ok_rules(mo.owners, sink=unit["kind"] not in FALLS))
        if unit.get("rules") == RULES
        else (lambda mo: mo.ok(mo.owners, unit["tol"]))
    )
    nb = len(unit.get("params", {}).get("bars", ())) or 1
    mine = lambda b: [mo for mo in monitors if getattr(mo, "bar", b) == b]  # a bracket's rollouts count only at its bar
    res = dict(integrity_ok=all(ok(mo) for mo in monitors), integrity=[mo.report(mo.owners) for mo in monitors])
    passed = []
    if unit["kind"] == "joint":  # a rollout's integrity counts only where its component counts
        res["details"] = {
            k: outcome[k] for k in ("clearance_report", "drive_reports", "diagnosis", "margins") if k in outcome
        }
        hint = unit.get("hint_components", [])
        for b in range(nb):
            comp = Fn.joint_components(unit, outcome, b)
            res["components" if b == 0 else "components_reference_bar"] = comp
            okb = {
                s: all(ok(mo) for mo in mine(b) if s == "physics_hint" or getattr(mo, "component", None) not in hint)
                for s in SETTINGS
            }
            passed.append(
                {
                    s: okb[s] and all(v for k, v in comp.items() if s == "physics_hint" or k not in hint)
                    for s in SETTINGS
                }
            )
    else:
        res["outcome"] = outcome
        for b in range(nb):
            same = P.part(outcome, b, nb) == P.part(unit["expect"], b, nb)
            passed.append({s: bool(all(ok(mo) for mo in mine(b)) and same) for s in SETTINGS})
        if monitors and getattr(monitors[0], "details", None) is not None:
            res["details"] = monitors[0].details
        if unit["kind"] in F4_KINDS:  # partial credit by stage; a run rule broken: 0
            from sbv.functionality.f4_robot.plan import task_score, task_points

            got, total = task_points(unit, res.get("details") or {})
            res["stage_points"] = [
                got if res["integrity_ok"] else 0,
                total,
            ]  # checkpoints passed, checkpoints in all
            res["stage_score"] = round(task_score(unit, res.get("details") or {}), 4) if res["integrity_ok"] else 0.0
        if unit["kind"] == "robot_activity" and "steps" in (res.get("details") or {}):  # reported: steps done in order
            done = sum(1 for st in res["details"]["steps"] if st.get("success"))
            n = max(1, len(unit["params"]["steps"]))
            res["credit"] = {s: (done / n if res["integrity_ok"] else 0.0) for s in SETTINGS}
    res["passed"], res["passed_reference_bar"] = passed[0], passed[-1]
    return res


def judge_init(unit, rec):
    ok = P.init_passes(unit, rec)
    return dict(
        record={k: v for k, v in (rec or {}).items() if k != "integrity"},
        integrity=(rec or {}).get("integrity"),
        passed={s: ok for s in SETTINGS},
    )
