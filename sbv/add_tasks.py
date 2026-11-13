#!/usr/bin/env python
"""Put a room's named robot tasks (tasks.json) into a frozen case without calibrating the room again.
The units are built as calibrate builds them (f4_robot/tasks.named, isolated, params.near) and accepted by its rule
(calibrate._robot_job): the nominal run and every jittered run succeed, cross no barrier, sink nothing over 1 cm, and
the task is hint-only if a plausible copy error (target mass x1.3 and friction +0.15, or x0.7 and -0.15) fails it.
Every robot unit the case had goes; the accepted named tasks join the headline `tasks` part. A hint-only task, or one
whose objects or surroundings are not all in the visible set, is dropped unless --keep-hint-only / no --visible.
Speed: the trials run in parallel processes. The 5 s settle of the room plus robot does not depend on the task, so
the trials of one start and jitter seed share it: a process settles once (plan.World.settle_steps) and forks one child
per task, which isolates for its task and runs it, the same steps in the same order as an unshared trial. The edited
runs settle with their edit applied and run awake, as in calibrate; nothing is shared there. With frozen trials
(SBV_ROBOT_FREEZE=1, recorded as params.freeze; plan.mode) every trial runs calibrate's own path instead, one process
per trial: the parent keeps the whole-room settle of each start once (plan.World.restore) before it forks, so the
nominal runs restore it, each jitter seed restores its own kept whole-room settle, and the edited and nudged runs
settle only their zone, exactly as calibrate does.
With --keep DIR each trial's result is kept there under a key over the unit (without its words), the scene file and the
physics, code and runtime, and reused. --nominal runs only each task's seed-0 trial, a quick feasibility check.
Usage: python -m sbv.add_tasks CASE ROOM_DIR TASKS_JSON [--procs N] [--out OUT] [--visible COVERAGE_JSON] [--keep DIR]
       [--keep-hint-only] [--dry-run] [--nominal]
"""

import argparse, hashlib, json, multiprocessing as mp, os, pickle, sys, tempfile, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repository root
from sbv.common import case as K  # noqa: E402
from sbv.common.scene import Scene  # noqa: E402
from sbv.functionality import f2_physical as P, f2_calibrate as C  # noqa: E402
from sbv.functionality.f4_robot import plan as RP, tasks as T  # noqa: E402
from sbv.functionality.units import robot_trial, RULES, F4_KINDS  # noqa: E402
from sbv.calibrate import _why  # noqa: E402

EDITS = (dict(mass=1.3, friction=0.15), dict(mass=0.7, friction=-0.15))  # calibrate._robot_job's plausible copy errors
FORK = mp.get_context("fork")
S = SEM = UNITS = OUT = None
T0 = time.time()
ISO_EDITS = False
TRIAL_INPUTS = None


def world_start(unit):
    """The base pose run_task and run_push build the World at (nearest free floor to params.start)."""
    p = unit["params"]
    xs, ys, occ = RP.floor_map(S, p["bounds"])
    s0 = RP.nearest_free(xs, ys, RP.yaw_maps(occ), p["start"])
    return None if s0 is None else (s0[0], s0[1], 0.0)


def summary(uid, seed, edit, outcome, mons, t):
    mo = mons[0]
    det = mo.details
    return dict(
        id=uid,
        seed=seed,
        edit=edit,
        ok=outcome == [True],
        details=det,
        crossed=bool(mo.crossed(mo.owners)),
        unresolved=bool(mo.report(mo.owners).get("sweep_unresolved")),
        pen=max(getattr(mo, "pen", {}).values() or [0.0]),
        seconds=round(time.time() - t, 1),
        done_at=round(time.time() - T0, 1),
    )


def trial_name(uid, trial):
    """trial: seed_<s> or edit_<k>; the file name keys everything the result depends on."""
    u = {k: v for k, v in UNITS[uid].items() if k != "words"}
    trial += "_isolated" if ISO_EDITS and trial.startswith("edit") else ""
    from sbv.common.calibration_inputs import rigid_identity

    inputs = TRIAL_INPUTS if TRIAL_INPUTS is not None else rigid_identity(os.path.dirname(S.xml), {})
    return (
        trial
        + "_"
        + hashlib.sha256(json.dumps([u, trial, inputs, S.phys], sort_keys=True, default=str).encode()).hexdigest()[:20]
    )


def _save(name, value):
    with open(os.path.join(OUT, name), "wb") as f:
        pickle.dump(value, f)


def _load(name):
    with open(os.path.join(OUT, name), "rb") as f:
        return pickle.load(f)


def _edit_job(uid, k):
    if ISO_EDITS:  # --isolate-edits: the edited run sleeps what its own (edited) settle left at rest, as a seed
        settle, keep = RP.World.settle, set(UNITS[uid]["slots"])  # run does ({obj, goal} or {obj}, pick_place / push)
        RP.World.settle = lambda self, k=None, awake=None: settle(self, keep if k is None else k, awake)
    with SEM:  # calibrate's own path: a World of its own, edited, awake
        t = time.time()
        S.reset()
        r = robot_trial(UNITS[uid]["kind"])(S, UNITS[uid], None, 0, EDITS[k])
        _save(trial_name(uid, f"edit_{k}"), summary(uid, 0, EDITS[k], *r, t))


def _child(w, x0, x1, start, uid, seed, warm):
    built = RP.World

    def shared(xml, st, textures=True, phys=None):  # the settled world of this start and seed, once
        return (
            w
            if (xml, tuple(st), textures, phys) == (S.xml, start, S.textures, S.phys)
            else built(xml, st, textures, phys)
        )

    RP.World = shared
    w.jitter = lambda seed: None
    w.settle_steps = lambda *a: (x0, x1)  # already jittered and settled
    with SEM:
        t = time.time()
        S.reset()
        r = robot_trial(UNITS[uid]["kind"])(S, UNITS[uid], None, seed)
        _save(trial_name(uid, f"seed_{seed}"), dict(summary(uid, seed, None, *r, t), shared_settle_s=warm))


def _trial_job(uid, trial):
    """A frozen trial (plan.mode): calibrate's own path, a World of its own; the whole-room settle it restores or reads
    (plan.World.restless) is the parent's kept one (_KEPT, inherited by the fork)."""
    kind, k = trial.split("_")
    k = int(k)
    seed, edit = (k, None) if kind == "seed" else (0, EDITS[k])
    with SEM:
        t = time.time()
        S.reset()
        r = robot_trial(UNITS[uid]["kind"])(S, UNITS[uid], None, seed, edit)
        _save(trial_name(uid, trial), summary(uid, seed, edit, *r, t))


def _seed_job(start, seed, uids):
    with SEM:
        t = time.time()
        w = RP.World(S.xml, start, S.textures, S.phys)
        w.jitter(seed)
        x0, x1 = w.settle_steps()
        warm = round(time.time() - t, 1)
    print(f"  settled start {tuple(round(v, 3) for v in start)} seed {seed} in {warm} s; {len(uids)} tasks", flush=True)
    kids = [FORK.Process(target=_child, args=(w, x0, x1, start, uid, seed, warm)) for uid in uids]
    for p in kids:
        p.start()
    for p in kids:
        p.join()


def verdict(unit, runs, edits):
    """calibrate._robot_job's decision on finished trials: (unit, None) or (None, why)."""
    if not runs[0]["details"]["success"]:
        return None, f"reference task fails: {_why(runs[0]['details'])}"
    if not all(r["ok"] for r in runs):
        return None, "task outcome changes under 1 mm jitter"
    if any(r["crossed"] for r in runs):
        return None, "the reference task crosses a barrier"
    if any(r["unresolved"] for r in runs):
        return None, "reference task motion check is unresolved"
    pen = max(r["pen"] for r in runs)
    if pen > 0.01:
        return None, f"the reference task pushes something {100 * pen:.1f} cm into another object"
    from sbv.calibrate import needs_and_nudges
    from sbv.functionality.units import robot_trial

    why = needs_and_nudges(S, unit, runs[0]["details"], robot_trial(unit["kind"]))
    if why:
        return None, why
    unit.update(
        expect=[True],
        tol=round(max(P.PEN_TOL, 1.5 * pen + 0.002), 4),
        hint_only=not all(e["ok"] for e in edits),
        rules=RULES,
    )
    return unit, None


ISO_WHY = (
    "the robot World decides what sleeps after its own 5 s settle, which runs with the edit applied, so whatever the edit "
    "sets moving stays awake; F2/F3 decide sleep from unedited init records, which is why their edited runs stay awake"
)


def add(
    case,
    room,
    spec,
    procs=8,
    visible=None,
    keep_hint=False,
    isolate=True,
    log=print,
    keep=None,
    nominal=False,
    pending=False,
    rule="all",
):
    """Run the named tasks' acceptance trials in the reference room; returns (accepted units, [(id, why)], trial
    rows)."""
    global S, SEM, UNITS, OUT, TRIAL_INPUTS
    from sbv.common.calibration_inputs import rigid_identity

    xml = K.scene_xml(os.path.abspath(room))
    jitter = case["calibration"]["jitter_seeds"]
    TRIAL_INPUTS = rigid_identity(room, {})
    S = Scene(xml, case["physics"], probes=True, textures=False)
    S.settle()
    ctx = C.context(S, case["room"]["center"])
    units, bad = T.named(S, ctx, spec, case["geometry"]["navigation"]["start"] + [0.0], case["room"]["bounds"])
    S.reset()
    dropped = [(i, "named task: " + w.split("named task: ")[-1]) for i, w in bad]
    go = []
    for u in units:
        u["params"]["near"] = RP.nearby(S, u["slots"])
        if isolate:
            u["params"]["isolate"] = True
        if RP.requested():
            u["params"]["freeze"] = RP.FROZEN  # frozen trials, as calibrate records them
        out = [
            n
            for n in u["slots"] + (u["params"]["near"] if rule == "all" else [])
            if visible is not None and n not in visible
        ]
        if out:
            dropped.append((u["id"], f"named task: not in the visible set: {', '.join(out)}"))
            continue
        go.append(u)
    UNITS = {u["id"]: u for u in go}
    starts = {}
    for u in go:
        st = world_start(u)
        if st is None:
            dropped.append(
                (u["id"], "named task: reference task fails: no free floor for the robot within 0.5 m of its start")
            )
        else:
            starts.setdefault(st, []).append(u["id"])
    UNITS = {i: UNITS[i] for ids in starts.values() for i in ids}
    log(
        f"{len(UNITS)} tasks to try ({len(dropped)} dropped before any trial); {len(starts)} start(s), seeds 0-{jitter}, {len(EDITS)} edits, {procs} processes"
    )
    trials = (
        ["seed_0"]
        if nominal or pending == "seed0"
        else [f"seed_{s}" for s in range(jitter + 1)] + [f"edit_{k}" for k in range(len(EDITS)) if not pending]
    )
    with tempfile.TemporaryDirectory(prefix="sbv_add_tasks_") as tmp:
        OUT = keep or tmp
        os.makedirs(OUT, exist_ok=True)
        SEM = FORK.BoundedSemaphore(procs)
        todo = lambda i, t: not os.path.exists(os.path.join(OUT, trial_name(i, t)))
        if RP.requested():  # frozen trials: calibrate's path, one whole-room settle per start
            for st, ids in starts.items():
                if any(todo(i, t) for i in ids for t in trials):
                    RP.World(S.xml, st, S.textures, S.phys).restore()
            jobs = [FORK.Process(target=_trial_job, args=(i, t)) for i in UNITS for t in trials if todo(i, t)]
        else:
            jobs = [
                FORK.Process(target=_edit_job, args=(i, int(t[5:])))
                for i in UNITS
                for t in trials
                if t.startswith("edit") and todo(i, t)
            ]
            jobs += [
                FORK.Process(target=_seed_job, args=(st, int(t[5:]), [i for i in ids if todo(i, t)]))
                for st, ids in starts.items()
                for t in trials
                if t.startswith("seed") and any(todo(i, t) for i in ids)
            ]
        log(
            f"{sum(todo(i, t) for i in UNITS for t in trials)} of {len(UNITS) * len(trials)} trials to run, the rest kept in {keep}"
        )
        for p in jobs:
            p.start()
        for p in jobs:
            p.join()
        rows = {
            i: [dict(id=i, trial=t, died=True) if todo(i, t) else _load(trial_name(i, t)) for t in trials]
            for i in UNITS
        }
    accepted = []
    for i, rs in rows.items() if not nominal or pending == "seed0" else ():
        if any(r.get("died") for r in rs):
            dropped.append((i, "a trial process died: " + ", ".join(r["trial"] for r in rs if r.get("died"))))
            continue
        u, why = verdict(UNITS[i], [r for r in rs if r["edit"] is None], [r for r in rs if r["edit"] is not None])
        if u is None:
            dropped.append((i, "named task: " + why))
        elif u["hint_only"] and not keep_hint:
            dropped.append(
                (
                    i,
                    "hint-only: fails under a plausible copy error "
                    + "; ".join(
                        f"(mass x{r['edit']['mass']}, friction {r['edit']['friction']:+}: {_why(r['details'])})"
                        for r in rs
                        if r["edit"] and not r["ok"]
                    ),
                )
            )
        else:
            accepted.append(u)
    if rigid_identity(room, {}) != TRIAL_INPUTS:
        raise RuntimeError("robot calibration dependencies changed during the run; retained trials must not be frozen")
    TRIAL_INPUTS = None
    return accepted, dropped, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case")
    ap.add_argument("room")
    ap.add_argument("tasks")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", help="write the case here (default: CASE, in place)")
    ap.add_argument("--visible", help="coverage JSON with a 'visible' list (default: the case's scope.visible, if any)")
    ap.add_argument("--keep", help="directory that keeps each trial's result, reused when the trial is the same")
    ap.add_argument("--keep-hint-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="trials only, no case written")
    ap.add_argument(
        "--isolate-edits",
        action="store_true",
        help="edited (hint-only check) runs isolated like the seed runs, not awake",
    )
    ap.add_argument(
        "--edits-pending",
        action="store_true",
        help="accept on the seed runs alone, hint_only False until the edited runs are done",
    )
    ap.add_argument(
        "--seed0-only",
        action="store_true",
        help="accept on the nominal run alone; jitter seeds and edited runs pending",
    )
    ap.add_argument(
        "--visible-rule",
        choices=["all", "object+goal"],
        default="all",
        help="all: the task's objects and its near objects must be visible; object+goal: only the task's own objects",
    )
    ap.add_argument(
        "--nominal",
        action="store_true",
        help="only each task's nominal trial (seed 0), a quick feasibility check; implies --dry-run",
    )
    a = ap.parse_args()
    t0 = time.time()
    global ISO_EDITS
    ISO_EDITS = a.isolate_edits
    with open(a.case) as f:
        case = json.load(f)
    if case["physics"].get("mujoco") != __import__("mujoco").__version__:
        raise SystemExit("the case was calibrated with another MuJoCo")
    with open(a.tasks) as f:
        spec = json.load(f)
    vis = json.load(open(a.visible))["visible"] if a.visible else (case.get("scope") or {}).get("visible")
    accepted, dropped, rows = add(
        case,
        a.room,
        spec,
        a.procs,
        None if vis is None else set(vis),
        a.keep_hint_only,
        keep=a.keep,
        nominal=a.nominal,
        pending="seed0" if a.seed0_only else a.edits_pending,
        rule=a.visible_rule,
    )
    f = case["functionality"]
    gone = [u["id"] for u in f["units"] if u["kind"] in F4_KINDS]
    f["units"] = [u for u in f["units"] if u["kind"] not in F4_KINDS] + accepted
    f["excluded"] = [e for e in f["excluded"] if e["id"].split("|")[0] not in F4_KINDS] + [
        dict(id=i, reason=w) for i, w in dropped
    ]
    done = not (a.edits_pending or a.seed0_only)
    case["robot_tasks_added"] = dict(
        tool="sbv.add_tasks",
        seeds_pending=a.seed0_only,
        edits_pending=not done,
        edits_isolated=ISO_EDITS and done,
        **(dict(edits_isolated_why=ISO_WHY) if ISO_EDITS and done else {}),
        visible_rule=a.visible_rule,
        room=os.path.abspath(a.room),
        tasks=os.path.abspath(a.tasks),
        visible=a.visible or ("case scope" if vis else None),
        removed=gone,
        added=[u["id"] for u in accepted],
        dropped=[dict(id=i, reason=w) for i, w in dropped],
        trials={
            i: [
                {k: r.get(k) for k in ("seed", "edit", "ok", "seconds", "done_at", "shared_settle_s", "died")}
                | dict(
                    success=(r.get("details") or {}).get("success"),
                    why=_why(r["details"]) if r.get("details") and not r["ok"] else None,
                )
                for r in rs
            ]
            for i, rs in rows.items()
        },
        seconds=round(time.time() - t0, 1),
        date=time.strftime("%Y-%m-%d %H:%M"),
    )
    for i, rs in case["robot_tasks_added"]["trials"].items():
        print(
            f"{i}: "
            + "; ".join(
                f"{'edit ' + str(r['edit']) if r['edit'] else 'seed ' + str(r['seed'])} "
                f"{'ok' if r['ok'] else 'FAIL ' + str(r['why'] or 'died')} {r['seconds']} s"
                for r in rs
            )
        )
    print(
        f"added {len(accepted)}: {[u['id'] for u in accepted]}; removed {len(gone)}; dropped {len(dropped)}: {dropped}; {case['robot_tasks_added']['seconds']} s"
    )
    if not (a.dry_run or a.nominal):
        K.save(case, a.out or a.case)
        print("wrote", a.out or a.case)


if __name__ == "__main__":
    main()
