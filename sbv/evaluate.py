#!/usr/bin/env python
"""Score one submission against a frozen case: Geometry (G1 inventory and counts, G2 shape and pose-and-size, G3
relations, placement and navigation) and Functionality (F1 gate; F2 and F3 units; C.4 aggregation, each information
setting separately), F4 robot tasks with the RB-Y1 (pick-and-place, contact-rich push), and optionally Appearance
(--appearance).
Usage: python -m sbv.evaluate CASE.json SUBMISSION (scene dir or .xml) --out report.json [--procs 8] [--repeats 0 1 2]
       [--appearance [--pictures 1|2|4|8|12]] [--keep-textures]
Without --appearance, rooms load with 1-pixel textures (sbv.common.scene.flat_textures): the F1 gate still compiles the
real pictures once.
"""

import argparse, json, os, sys, time
import numpy as np, mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repository root
from sbv.common import case as K  # noqa: E402
from sbv.functionality import f2_physical as P, f3_functions as Fn, sleep as SL  # noqa: E402
from sbv.geometry import g1_inventory as Inv, g2_shape as Geo, g3_layout as Lay  # noqa: E402
from sbv.common.scene import Scene, make, Unstable  # noqa: E402
from sbv.common.workers import Workers, BrokenProcessPool  # noqa: E402
from sbv.functionality.f1_gate import gate, constraints  # noqa: E402
from sbv.functionality.units import run, judge, judge_init, number, SETTINGS, LOCAL_KINDS, RETIRED, zone, F4_KINDS  # noqa: E402
from sbv.functionality.aggregate import functionality, levels, level_points  # noqa: E402
from sbv.appearance import views as App  # noqa: E402
from sbv import groups as GR, reward as R, scope as SC  # noqa: E402

S = CORR = None
SUBS = {}


class Clock:
    """Seconds per part of the scoring, for the report (doc C.2: record the cost separately)."""

    def __init__(self):
        self.t, self.parts = time.time(), {}

    def lap(self, name):
        now = time.time()
        self.parts[name] = round(now - self.t, 3)
        self.t = now


def _init(xml, phys, vocab, slots, units, textures, local_m=None, small=False, start=None, restless=None, groups=()):
    """start: the settled state the main process already computed for the same model (MuJoCo on a CPU is deterministic,
    so settling again in every worker gives the same state); None: settle here. restless: the copy's objects that move
    by themselves, never put to sleep in door and drawer tests (functionality/sleep.py); None: no isolation."""
    global S, CORR, LOCAL, SMALL, RESTLESS
    S = make(xml, phys, vocab, probes=True, textures=textures)
    S.reset(S.q0)
    CORR = GR.pairing(slots, S, groups)[0]
    LOCAL, SMALL, RESTLESS = local_m, small, restless
    SUBS.clear()
    CORR.update(Fn.joint_correspondence(units, S, CORR))
    if start is None or len(start) != S.m.nq:
        S.settle()
    else:
        S.start = np.array(start, float)


def _unit_job(args):
    unit, seeds = args
    out = []
    near = (
        [CORR.get(n) for n in zone(unit)] if unit["kind"] in LOCAL_KINDS else []
    )  # the copy's own objects of the test
    key = tuple(sorted(n for n in near if n))
    if key and SMALL and key not in SUBS:  # a small model per zone, kept for this worker's next units
        if len(SUBS) >= 16:
            SUBS.pop(next(iter(SUBS)))
        SUBS[key] = S.subset(near, LOCAL)
    sub = SUBS.get(key)
    if sub is not None and not all(
        c in sub.objects for c in (CORR.get(x) for x in zone(unit) + list(unit["slots"])) if c
    ):
        sub = None  # the test names an object the small model left out: whole room
    iso = RESTLESS if unit["kind"] in SL.KINDS else None
    mine = [CORR[n] for n in unit["slots"] if n in CORR]

    def once(s):
        if sub is not None:
            return judge(unit, *run(sub, unit, CORR, sub.jittered(s), seed=s))
        with S.focus(near, LOCAL), SL.isolated(S, mine, iso):
            return judge(unit, *run(S, unit, CORR, S.jittered(s), seed=s))

    for s in seeds:
        t = time.time()
        try:
            try:
                r = once(s)
            except mujoco.FatalError as e:  # MuJoCo could not sleep this copy (sleep.awake): the whole room awake
                if SL.AWAKE:
                    raise
                with SL.awake():
                    r = once(s)
                r["awake"] = f"FatalError: {str(e)[:200]}"
        except Unstable as e:  # the copy's own physics diverged: a verdict, this unit fails
            r = dict(passed={"physics_hint": False, "input_only": False}, error=f"Unstable: {str(e)[:300]}")
        except Exception as e:  # anything else is the grader's own failure: no reward (evaluate)
            r = dict(
                passed={"physics_hint": False, "input_only": False},
                error=f"{type(e).__name__}: {str(e)[:300]}",
                grader_error=True,
            )
        r["seconds"] = round(time.time() - t, 3)
        out.append(r)
    return unit["id"], out


def object_gate(case, rep, plain, corr):
    """{copy object: reasons} for the object gate: a bad value it owns (F1.3), or more welds, gravity compensation or
    contact excludes than its reference object (cases from before record none, so that part is skipped)."""
    failed = {n: list(v) for n, v in rep.get("object_gate", {}).items()}
    if plain is None:
        return failed
    held = constraints(plain)
    for o in case["objects"]:
        c, t = corr.get(o["slot"]), o.get("constraints")
        if c is None or t is None or c not in held:
            continue
        more = [
            f"{k} {held[c][k]} (reference room {t[k]})"
            for k in ("equality", "gravcomp", "exclude")
            if held[c][k] > t[k] + 1e-9
        ]
        if more:
            failed.setdefault(c, []).append("hidden constraint: " + ", ".join(more))
    return failed


def _unit_jobs(jobs, procs, initargs):
    """F2.2 to F4.2 in worker processes. If a worker dies (sbv/common/workers.py), units already done keep their results
    and every other unit fails with the cause; the report is still written, with status grader_failed and no reward."""
    out, lost = {}, []
    with Workers(
        procs, _init, initargs, (initargs[:3], dict(probes=True, textures=initargs[5]))
    ) as pool:  # _init's Scene
        for (unit, seeds), f in [(j, pool.submit(_unit_job, j)) for j in jobs]:
            try:
                out[unit["id"]] = f.result()[1]
            except BrokenProcessPool:
                out[unit["id"]] = None
                lost.append(unit["id"])
            except Exception as e:
                out[unit["id"]] = [
                    dict(
                        passed={k: False for k in SETTINGS},
                        error=f"{type(e).__name__}: {str(e)[:300]}",
                        grader_error=True,
                    )
                ]
    if not lost:
        return out, None
    why = pool.why()
    err = f"WorkerDied: {why}; the run stopped before this unit finished: a grader failure, not a verifier verdict, so the run has no reward"
    for unit, seeds in jobs:
        if unit["id"] in lost:
            out[unit["id"]] = [dict(passed={k: False for k in SETTINGS}, error=err) for _ in seeds]
    return out, dict(reason=why, units=lost)


def true_handles(case, units, phys, vocab):
    """Joint units with ref["handle"] read from the reference room (f3_functions.has_handle), for a case that does not
    record it; a part the reference room no longer has keeps no requirement."""
    ref = Scene(K.reference_path(case), phys, vocab, textures=False)
    ref.reset(ref.q0)
    out = []
    for u in units:
        if u["kind"] == "joint" and "handle" not in u["params"]["ref"]:
            n = u["slots"][0]
            j = next((k for k in ref.joints.get(n, []) if Fn.jname(ref.m, k) == u["params"]["ref"]["name"]), None)
            u = dict(
                u,
                params=dict(
                    u["params"], ref=dict(u["params"]["ref"], handle=j is not None and Fn.has_handle(ref, n, j))
                ),
            )
        out.append(u)
    return out


def geometry(case, plain, corr, ref=None):
    g = case["geometry"]
    ref = (
        ref
        if ref is not None
        else Scene(K.reference_path(case), case["physics"], case["vocabulary"], textures=plain.textures)
    )
    ref.reset(ref.q0)
    plain.reset(plain.q0)
    clk = Clock()
    g1 = dict(
        inventory=Inv.inventory(g["inventory"], plain, corr, case["objects"]),
        counts=Inv.counts(g["count_constraints"], plain),
        words=Inv.words(case["objects"], plain, corr),
    )
    clk.lap("G1")
    from sbv.geometry.reference_cache import samples

    rs = samples(ref, case["objects"], g["shape"]["samples"])
    gs = {n: Geo.surface_samples(plain, n, g["shape"]["samples"]) for n in plain.objects}
    shape = {s: Geo.shape_scores(rs[s], gs[n], g["shape"]["tau"]) for s, n in corr.items() if len(rs[s]) and len(gs[n])}
    mean = lambda k: float(np.mean([v[k] for v in shape.values()])) if shape else None
    g2 = dict(
        fscore=mean("fscore"),
        chamfer=mean("chamfer"),
        matched=len(shape),
        required=len(case["objects"]),
        per_object=shape,
    )
    clk.lap("G2.1-G2.2")
    g2["pose_size"] = Geo.pose_size(g["pose_size"], case["objects"], plain, corr, gs, rs)
    clk.lap("G2.3")
    rel = Lay.relations(g["relations"], plain, corr, g["placement"]["support_tol"], g["placement"]["penetration_tol"])
    clk.lap("G3.1")
    place = Lay.placement(g["placement"], case["objects"], plain, corr)
    clk.lap("G3.2")
    nav = Lay.navigation(g["navigation"], plain, corr)
    clk.lap("G3.3")
    return dict(G1=g1, G2=g2, G3=dict(relations=rel, placement=place, navigation=nav), seconds=clk.parts)


def parameter_errors(case, scene, corr):
    """Diagnostics only: mass ratio, friction difference, centre-of-mass offset error against the reference."""
    out = {}
    scene.reset(scene.q0)
    for o in case["objects"]:
        n = corr.get(o["slot"])
        if n is None:
            continue
        b = scene.objects[n]
        lo, hi = scene.aabb(n)
        gsx = [g for g in scene.ogeoms[n] if scene.collides(g)]
        com = scene.d.subtree_com[b] - (lo + hi) / 2
        h = max(o["height"], 1e-6)
        out[o["slot"]] = dict(
            object=n,
            mass_ratio=round(float(scene.m.body_subtreemass[b]) / max(o["mass"], 1e-9), 3),
            friction_diff=round(float(np.mean(scene.m.geom_friction[gsx, 0])) - o["friction"], 3) if gsx else None,
            com_error=round(float(np.linalg.norm(com - o["com_rel"])) / h, 3),
            com_height_error=round(float(com[2] - o["com_rel"][2]) / h, 3),
        )
    return out


def evaluate(case, submission, procs=8, repeats=None, log=print, renders=None, condition=None, textures=None):
    """textures: load the rooms' pictures; None = only when Appearance is scored (renders given)."""
    if case["physics"].get("mujoco") and case["physics"]["mujoco"] != mujoco.__version__:
        raise SystemExit(
            f"the case was calibrated with MuJoCo {case['physics']['mujoco']}; this environment has {mujoco.__version__}"
        )
    t0 = time.time()
    xml = K.scene_xml(os.path.abspath(submission))
    phys, vocab = case["physics"], case["vocabulary"]
    G = GR.groups(case)
    half = {
        o["slot"]: float(o["dims"][2]) / 2 for o in case["objects"] if o.get("dims")
    }  # joints compared from the box bottom
    units = [
        dict(u, params=dict(u["params"], half_h=half[u["slots"][0]]))
        if u["kind"] == "joint" and u["slots"][0] in half
        else u
        for u in case["functionality"]["units"]
        if u["kind"] not in RETIRED
    ]  # older cases still carry retired kinds
    seeds = repeats if repeats is not None else case["functionality"]["repeats"]
    tex = bool(renders) if textures is None else textures
    if any(
        u["kind"] == "joint" and "handle" not in u["params"]["ref"] for u in units
    ):  # a case from before the handle check:
        units = true_handles(case, units, phys, vocab)  # read the reference room's parts
    clk = Clock()
    E, rep, plain = gate(xml, phys, vocab, textures=tex)
    clk.lap("F1")
    if rep.get("inert"):
        phys = dict(phys, inert=rep["inert"])  # objects that blow the copy up stay inert in every Scene below
    from sbv.common.scene import reference_probes

    if "probes" not in phys:
        phys = dict(phys, probes=reference_probes(K.reference_path(case)))
    report = dict(
        case_hash=case.get("case_hash"),
        submission=xml,
        F1=dict(passed=E, **rep),
        verifiers={u["id"]: number(u) for u in units},
    )
    if plain is None:
        report["geometry"] = None
        corr = {}
    else:  # groups' members pair among themselves (sbv/groups.py)
        plain.reset(plain.q0)
        corr, by = GR.pairing(case["objects"], plain, G)
        report["correspondence"] = corr
        report["matching"] = K.match_report(
            case["objects"], plain, corr
        )  # matcher, word mismatches, merges, splits: unscored
        grp = GR.score(G, plain, case["objects"], settle=False, by=by)
        ref = Scene(
            K.reference_path(case), case["physics"], case["vocabulary"], textures=plain.textures
        )  # the reference room, loaded once
        report["geometry"] = geometry(case, plain, corr, ref)
        report["parameter_errors"] = parameter_errors(case, plain, corr)
        if renders:
            ref.reset(ref.q0)
            true_rays = App.ray_depths(case["appearance"]["cameras"], ref)  # for the near-camera warning
        del ref
    clk.lap("G")
    results = {}
    failed = object_gate(case, rep, plain, corr)  # copy object -> why it fails the object gate
    if failed:
        report["object_gate"] = failed
    if E:
        scene = Scene(xml, phys, vocab, probes=True, textures=tex)
        scene.reset(scene.q0)
        c2 = GR.pairing(case["objects"], scene, G)[0]
        c2.update(Fn.joint_correspondence(units, scene, c2))
        keep = {}
        recs = [P.init_record(scene, scene.jittered(s, base=scene.q0), keep=keep if s == 0 else None) for s in seeds]
        diverged = None
        if "q1" in keep:
            scene.start = keep["q1"]  # seed 0's state at settle_s: bit for bit what scene.settle() gives
        else:  # no seed 0 among the repeats, or its run was unstable
            try:
                scene.settle()
            except Unstable as e:  # the copy's own physics diverged: a verdict, as for one unit
                diverged = f"the whole room's settle diverged (Unstable: {str(e)[:100]})"
                scene.start = scene.q0.copy()
                report["settle_diverged"] = diverged  # no settled room: nothing rests, nothing touches, no unit runs
        GR.rest(G, grp["groups"], scene, by)  # each group's F2 unit: at rest
        if diverged:
            for row in grp["groups"]:
                row["at_rest"] = False
        grp = GR.summarize(G, grp["groups"])
        if report.get("geometry"):
            Lay.touching(
                report["geometry"]["G3"]["relations"], scene, c2, settled=not diverged
            )  # "on" needs contact, settled
        results.update(
            {
                u["id"]: [dict(passed={k: bool(r["at_rest"]) for k in SETTINGS}, group=r)]
                for u, r in zip(GR.units(G), grp["groups"])
            }
        )
        for u in units:
            if u["kind"] == "init":
                results[u["id"]] = [judge_init(u, r.get(c2.get(u["slots"][0]))) for r in recs]
        clk.lap("F2.1")
        for u in units:  # a unit on an object that fails the object gate fails, unrun
            why = [f"{c}: {'; '.join(failed[c])}" for c in (corr.get(x) for x in u["slots"]) if c in failed]
            if why:
                results[u["id"]] = [
                    dict(
                        passed={k: False for k in SETTINGS},
                        passed_reference_bar={k: False for k in SETTINGS},
                        object_gate=why,
                    )
                    for _ in seeds
                ]
        if diverged:  # every unit but init fails, unrun: there is no settled room to start from
            for u in units:
                if u["kind"] != "init" and u["id"] not in results:
                    results[u["id"]] = [
                        dict(
                            passed={k: False for k in SETTINGS},
                            passed_reference_bar={k: False for k in SETTINGS},
                            settle_diverged=True,
                        )
                        for _ in seeds
                    ]
        far = {n for n, r in recs[0].items() if P.away(r)} if recs else set()  # a fallen object earns nothing
        for u in units:  # more: its other units fail unrun, "not at rest" (seed 0's rollout)
            gone = sorted({c for c in (c2.get(x) for x in u["slots"]) if c in far})
            if u["kind"] != "init" and u["id"] not in results and gone:
                results[u["id"]] = [
                    dict(
                        passed={k: False for k in SETTINGS},
                        passed_reference_bar={k: False for k in SETTINGS},
                        not_at_rest=gone,
                    )
                    for _ in seeds
                ]
        paired = sorted({c2[o["slot"]] for o in case["objects"] if c2.get(o["slot"]) in (recs[0] if recs else {})})
        loose = [n for n in paired if not recs[0][n].get("still")]  # not still, or the rollout was unstable
        report["at_rest"] = dict(
            objects=len(paired),
            not_still=len(loose),
            share=round(len(loose) / len(paired), 4) if paired else 0.0,
            flag=bool(paired) and len(loose) / len(paired) > P.FLAG_SHARE,
            away=sorted(far & set(paired)),
        )
        jobs = [(u, seeds) for u in units if u["kind"] != "init" and u["id"] not in results]
        iso = SL.restless(recs) if case["functionality"].get("isolate") else None
        res, lost = _unit_jobs(
            jobs,
            procs,
            (
                xml,
                phys,
                vocab,
                case["objects"],
                units,
                False,
                case["functionality"].get("local_m"),  # no pictures: physics never reads them
                case["functionality"].get("small_models", False),
                scene.start.copy(),
                iso,
                G,
            ),
        )
        results.update(res)
        if lost:
            report["incomplete"] = lost
        clk.lap("F2.2-F4.2")
    report["functionality"] = functionality(
        1.0 if E else 0.0, units, results, slots=[o["slot"] for o in case["objects"]]
    )
    report["functionality_reference_bar"] = functionality(
        1.0 if E else 0.0, units, results, bar=1
    )  # beside the main bar
    report["levels"] = levels(1.0 if E else 0.0, units, results)  # per object, as before; the reward reads the scopes
    report["levels_reference_bar"] = levels(1.0 if E else 0.0, units, results, bar=1)
    report["level_points"] = level_points(1.0 if E else 0.0, units, results)  # the points the reward reads
    report["level_points_reference_bar"] = level_points(1.0 if E else 0.0, units, results, bar=1)
    if plain is not None:  # the scored numbers: groups for their members, full and visible scope
        report["groups"] = dict(grp, defs=[{k: g[k] for k in ("id", "word", "container", "n", "members")} for g in G])
        report["verifiers"].update({u["id"]: number(u) for u in GR.units(G)})
        report["scopes"] = SC.scopes(
            case, report, G, grp["groups"], dict(plain.cat), corr, by, units + GR.units(G), results, 1.0 if E else 0.0
        )
    if renders and plain is not None:  # Appearance, rendered at the submitted initial state
        plain.reset(plain.q0)
        near = App.near_cameras(
            case["appearance"]["cameras"], plain, true=true_rays
        )  # a warning only, never scored: what the reference room lacks
        report["appearance"] = dict(
            App.evaluate(case, xml, renders, condition, plain), near_camera=near, floor=case["appearance"].get("floor")
        )
        clk.lap("A")
    lost, bad = report.get("incomplete"), sorted(k for k, v in results.items() if any(x.get("grader_error") for x in v))
    woke = sorted(k for k, v in results.items() if v and any(x.get("awake") for x in v))
    if woke:
        report["awake_reruns"] = woke  # units run again with the whole room awake (sleep.awake)
    why = ([f"INCOMPLETE: {lost['reason']}; {len(lost['units'])} units not run to the end"] if lost else []) + (
        [f"an exception escaped {len(bad)} units, e.g. {bad[0]}: {results[bad[0]][0]['error']}"] if bad else []
    )
    report["status"] = (
        "grader_failed" if why else "ok"
    )  # grader_failed: reward.json gets no reward (pipeline.reward_file)
    if why:
        report["failure"] = "; ".join(why)
    from sbv.functionality.f4_robot.plan import mode as robot_mode  # how each robot task settled here (frozen or not)

    robot = {u["id"]: robot_mode(u) or "whole room" for u in units if u["kind"] in F4_KINDS}
    if robot:
        report["robot_settle"] = robot
    report["units"] = results
    report["seconds"] = round(time.time() - t0, 1)
    report["seconds_by_part"] = clk.parts
    return report


def summary(r):
    g, f = r.get("geometry"), r["functionality"]
    lines = [
        f"GRADER FAILED (no reward): {r['failure']}"
        if r.get("status") == "grader_failed"
        else f"status: {r.get('status', 'ok')}"
    ]
    lines += [
        f"F1 gate: {'pass' if r['F1']['passed'] else 'FAIL ' + json.dumps({k: v for k, v in r['F1'].items() if v and k not in ('passed', 'cost')})}"
    ]
    pct = lambda x: "n/a" if x is None else f"{100 * x:.1f}"
    if r.get("groups"):
        gr = r["groups"]
        bad = [x for x in gr["groups"] if not x["passed"]]
        lines.append(
            f"Groups: {len(gr['groups'])} stand for {gr['members_out']} reference objects, score {gr['score']}; failed {len(bad)}: "
            + ", ".join(
                f"{x['id']} ("
                + (
                    "absent"
                    if not x["copy"]
                    else f"copy {x['copy']}/{x['true']}{'' if x['placed'] else ', off place'}"
                    f"{', stayed ' + x['stayed'] if x.get('stayed') else ''}"
                )
                + ")"
                for x in bad[:8]
            )
        )
    for k, s in (r.get("scopes") or {}).items():
        rw, p = R.reward(r, scope=k)
        q, geo = s["Q_parts"], s["geometry"]
        lines += [
            f"Scope {k}{'' if s.get('defined', True) else ' (the case has no visible set: every object)'}: reward {rw} (non-rigid apart) "
            f"{ {n: v for n, v in p.items() if v is not None} }",
            f"  C {pct(s['C'])} ({s['paired']}/{s['objects']} objects paired; in scope {s['singles']} single objects and {s['groups']} groups), Q {pct(s['Q'])} "
            f"(pose {q['pose'][0]:g}/{q['pose'][1]}, at rest {q['at_rest'][0]:g}/{q['at_rest'][1]}, units {q['units'][0]:g}/{q['units'][1]}); "
            f"out of scope: {s['units_out_of_scope']} units, {s['relations_out_of_scope']} relations",
            f"  G1 inventory F1 {pct(geo['G1']['inventory']['f1'])}, G2 shape {pct(geo['G2']['fscore'])} on {geo['G2']['matched']}, pose-and-size "
            f"{pct(geo['G2']['pose_size']['score'])}, G3 relations {pct(geo['G3']['relations']['score'])}, placement {pct(geo['G3']['placement']['score'])}, "
            f"navigation {pct(geo['G3']['navigation']['score'])}; functionality (input only) {s['functionality']['input_only']['functionality']} "
            f"layers {s['functionality']['input_only']['layers']}",
        ]
    if r.get("scopes"):
        lines.append("Per object, as before (every reference object one by one, no groups, no scope):")
    for s in ("physics_hint", "input_only"):
        lines.append(
            f"Functionality ({s}): {f[s]['functionality']}  layers {f[s]['layers']}  units {f[s]['units']}  groups {f[s].get('groups')}"
        )
        if r.get("functionality_reference_bar"):
            fr = r["functionality_reference_bar"][s]
            lines.append(f"  at the reference bars: {fr['functionality']}  layers {fr['layers']}")
    if f.get("declared_coverage"):
        c = f["declared_coverage"]
        lines.append(f"Declared checks cover {c['covered']} of {c['objects']} objects")
    if r.get("appearance"):
        a = r["appearance"]
        lines.append(
            f"Appearance: {a.get('mean', a.get('status'))}"
            + (f"  unavailable: {sorted(a['unavailable'])}" if a.get("unavailable") else "")
        )
        if a.get("near_camera"):
            lines.append(
                f"Appearance WARNING: {len(a['near_camera'])} geoms within {App.NEAR} m in front of held-out cameras, nearer than the reference room shows there: "
                + ", ".join(f"{w['body']}/{w['geom']} {w['distance']} m ({w['camera']})" for w in a["near_camera"][:10])
            )
    if g:
        w = g["G1"].get("words") or {}
        lines += [
            f"G1 inventory F1 {pct(g['G1']['inventory']['f1'])}, counts {pct(g['G1']['counts']['score'])}, "
            f"words right {pct(w.get('accuracy'))} of {w.get('paired', 0)} paired (not scored)",
            f"G2 surface F-score {pct(g['G2']['fscore'])} on {g['G2']['matched']}/{g['G2']['required']} matched, pose-and-size {pct(g['G2']['pose_size']['score'])}",
            f"G3 relations {pct(g['G3']['relations']['score'])}, placement {pct(g['G3']['placement']['score'])}, navigation {pct(g['G3']['navigation']['score'])}",
        ]
    failed = [
        f"{r.get('verifiers', {}).get(k, '?')} {k}"
        for k, v in r["units"].items()
        if not all(x["passed"]["physics_hint"] for x in v)
    ]
    lines.append(f"failed units ({len(failed)}): {', '.join(failed[:25])}{' ...' if len(failed) > 25 else ''}")
    mm = lambda v: "n/a" if v is None else f"{1000 * v:.1f} mm"
    for k, v in [
        (k, v[0])
        for k, v in r["units"].items()
        if k.startswith("joint|") and not all(x["passed"]["physics_hint"] for x in r["units"][k])
    ][:30]:
        d = v.get("details") or {}
        cr, mg = d.get("clearance_report") or {}, d.get("margins") or {}  # the first repeat's margins
        bad = [c for c, ok in (v.get("components") or {}).items() if not ok] or [
            v.get("error") or "; ".join(v.get("object_gate") or []) or "?"
        ]
        lines.append(
            f"  {k}: fails {', '.join(bad)}; worst penetration {mm(cr.get('worst_penetration'))} (bar {mm(cr.get('tolerance'))}), pivot off "
            f"{mm(mg.get('pivot_m'))} (bar {mm(mg.get('pivot_bar_m'))}), moving part off {mm(mg.get('connectivity_m'))} (bar {mm(mg.get('connectivity_bar_m'))})"
        )
    lines.append("seconds: " + ", ".join(f"{k} {v}" for k, v in r.get("seconds_by_part", {}).items()))
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("submission")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--repeats", type=int, nargs="*")
    ap.add_argument(
        "--appearance", action="store_true", help="also render the case cameras and score SigLIP-2, DINOv3, LPIPS"
    )
    ap.add_argument(
        "--pictures",
        choices=["1", "2", "4", "8", "12"],
        help="the picture condition the copy was made from: also score its input cameras",
    )
    ap.add_argument(
        "--keep-textures",
        action="store_true",
        help="load the rooms' pictures even without --appearance (no score changes; more memory)",
    )
    a = ap.parse_args()
    renders = os.path.splitext(os.path.abspath(a.out))[0] + "_renders" if a.appearance else None
    r = evaluate(
        K.load(a.case),
        a.submission,
        a.procs,
        a.repeats or None,
        renders=renders,
        condition=a.pictures,  # --repeats with no seeds: the case's
        textures=True if a.keep_textures else None,
    )
    with open(a.out, "w") as f:
        json.dump(r, f, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    print(summary(r))
    print("wrote", a.out)
    sys.exit(
        1 if r.get("status") == "grader_failed" else 0
    )  # the report is written either way; reward.json says grader_failed
