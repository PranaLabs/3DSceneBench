"""Calibrate a reference room's package tasks and score a copy with them (schema.py has the file and the entry
contract).
  python -m sbv.package_tasks.verify validate ROOM_DIR
  python -m sbv.package_tasks.verify calibrate ROOM_DIR --out package_case.json [--procs N]
  python -m sbv.package_tasks.verify evaluate package_case.json COPY_DIR --report report.json --out package_report.json [--procs N]
Calibration runs each task on the reference room: the nominal run (seed 0) and its jittered runs (seeds 1 to jitter)
must pass every stage with a finite state and the same stage names, and the jittered runs must say they moved the
objects. Then each declared wrong copy (a copy of the reference room with its sbv.perturb edits and nonrigid_set
changes) must run to the end and fail. Anything else refuses the room with every reason, as robot and non-rigid
calibration do: a check that no wrong copy fails adds nothing, and a crash is not a failure of the check. The case
freezes each task's code by SHA-256 with its params and the measured metrics.
Scoring takes the copy's objects from the main report's correspondence (rigid) and the non-rigid verifiers' pairing
(nonrigid); a task with an unmatched object scores 0 without running. A copy scores the share of the task's stages
passed in order, as a robot task does; a run whose stage names differ from the frozen ones scores 0. The code always
comes from the reference room beside the case, never from the copy, and a changed hash stops the grader. One nominal run
per task, as the robot tasks are scored."""

import argparse
import hashlib
import json
import re
import os
import shutil
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from . import schema as S
from sbv.common.dependencies import operation, current_session, use_session

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repository root
CASE_SCHEMA, REPORT_SCHEMA = "package_case/v0", "package_report/v0"


def python(engine):
    return (
        (os.environ.get("SBV_NONRIGID_PYTHON") or os.environ.get("SBV_SOFT_PYTHON") or sys.executable)
        if engine == "genesis"
        else sys.executable
    )


def code_files(room, entry):
    """The files a task's code is frozen by: every file in the entry's folder and below (helpers it imports), or the
    entry alone when it sits at the room's root."""
    folder = os.path.dirname(os.path.join(room, entry))
    if os.path.realpath(folder) == os.path.realpath(room):
        return [entry]
    return sorted(
        os.path.relpath(os.path.join(r, f), room)
        for r, ds, fs in os.walk(folder)
        for f in fs
        if "__pycache__" not in r.split(os.sep) and not f.endswith(".pyc")
    )


def code_hash(room, entry):
    h = hashlib.sha256()
    for rel in code_files(room, entry):
        h.update(rel.encode())
        with open(os.path.join(room, rel), "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def context(scene_dir, objects, params, seed, files=(), entry=None):
    from sbv.nonrigid.schema import side_file

    scene_dir = os.path.abspath(scene_dir)
    nonrigid = side_file(scene_dir)
    nonrigid = nonrigid if os.path.exists(nonrigid) else None
    return dict(
        scene_dir=scene_dir,
        scene_xml=os.path.join(scene_dir, "scene.xml"),
        nonrigid=nonrigid,
        soft_liquid=nonrigid,
        objects=objects,
        params=params,
        seed=seed,
        files=list(files),
        entry=entry or "nested/entry.py",
    )


def run_one(entry, engine, ctx, work, timeout, threads=1):
    """One run in its own process group, killed whole at the timeout. Returns the runner's result or {"error": ...}."""
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(os.path.join(work, "out"))
    from .input_bundle import prepare

    try:
        entry, ctx = prepare(entry, ctx, work)
    except (OSError, ValueError) as e:
        return dict(error=f"task inputs: {e}")  # e.g. a copy without a declared file
    ctx = dict(ctx, work_dir=os.path.join(work, "out"), threads=threads)
    cp, out = os.path.join(work, "ctx.json"), os.path.join(work, "result.json")
    with open(cp, "w") as f:
        json.dump(ctx, f, indent=1)
    env = dict(
        os.environ, PYTHONPATH=HERE + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    )
    with open(os.path.join(work, "stdout.log"), "w") as so, open(os.path.join(work, "stderr.log"), "w") as se:
        p = subprocess.Popen(
            [python(engine), "-m", "sbv.package_tasks.runner", entry, cp, out],
            cwd=HERE,
            env=env,
            stdout=so,
            stderr=se,
            start_new_session=True,
        )
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            return dict(error=f"timed out after {timeout} s")
    if not os.path.exists(out):
        with open(os.path.join(work, "stderr.log")) as f:
            tail = f.read()[-500:]
        return dict(error=f"the run wrote no result (exit {p.returncode}): {tail}")
    with open(out) as f:
        return json.load(f)


def names(r):
    return [s[0] for s in r["stages"]]


def first_failed(r):
    return next((s[0] for s in r["stages"] if not s[1]), None)


def merge(base, extra):
    for k, v in extra.items():
        base[k] = merge(dict(base.get(k) or {}), v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return base


def wrong_copy(room, neg, dest, files=()):
    """A copy of the reference room with the negative's edits: sbv.perturb on scene.xml, then nonrigid_set on
    nonrigid.json."""
    shutil.rmtree(dest, ignore_errors=True)
    if neg.get("perturb"):
        from sbv import perturb

        try:
            perturb.main([room, dest, *neg["perturb"]])
        except SystemExit as e:
            raise ValueError(f"negative {neg['name']}: sbv.perturb {' '.join(neg['perturb'])} failed: {e}") from None
    else:
        shutil.copytree(room, dest)
    edits = neg.get("nonrigid_set", neg.get("soft_set"))  # soft_set.
    if edits:
        from sbv.nonrigid.schema import side_file

        side = side_file(dest)
        d = json.load(open(side))
        for o in d["objects"]:
            if o.get("name") in edits:
                merge(o, edits[o["name"]])
        with open(side, "w") as f:
            json.dump(d, f, indent=1)
    return dest


@operation
def calibrate(room, out, procs=4):
    tasks, problems = S.load(room)
    if problems:
        raise SystemExit("package tasks: " + "; ".join(problems))
    room = os.path.abspath(room)
    work = os.path.join(os.path.dirname(os.path.abspath(out)), "package_cal")
    jobs = []
    from .input_bundle import identity as input_identity

    input_files = {t["id"]: input_identity(room, os.path.join(room, t["entry"]), t["files"]) for t in tasks}
    if os.path.realpath(work).startswith(os.path.realpath(room) + os.sep):
        raise SystemExit("write the package case outside the room folder")
    frozen = {
        t["id"]: (code_files(room, t["entry"]), code_hash(room, t["entry"])) for t in tasks
    }  # the code these runs used
    for t in tasks:
        ident, entry = {o: o for o in t["objects"]}, os.path.join(room, t["entry"])
        jobs += [
            (t, f"seed{s}", entry, context(room, ident, t["params"], s, t["files"], t["entry"]))
            for s in range(t["jitter"] + 1)
        ]
        for n in t["negatives"]:
            try:
                scene = wrong_copy(room, n, os.path.join(work, t["id"], "wrong_" + n["name"]), t["files"])
            except ValueError as e:
                raise SystemExit(f"package task {t['id']}: {e}") from None
            jobs.append((t, "wrong_" + n["name"], entry, context(scene, ident, t["params"], 0, t["files"], t["entry"])))
    with ThreadPoolExecutor(max(1, procs), initializer=use_session, initargs=(current_session(),)) as ex:

        def trial(j):
            from .calibration_cache import completed

            task, label, entry, ctx = j
            target = os.path.join(work, task["id"], "run_" + label)
            return completed(
                entry,
                task["engine"],
                ctx,
                target,
                task["timeout_s"],
                1,
                python(task["engine"]),
                lambda: run_one(entry, task["engine"], ctx, target, task["timeout_s"]),
                code_room=room,
            )

        res = list(ex.map(trial, jobs))
    for t in tasks:  # the wrong copies are rebuilt from their edits; their run logs stay
        for n in t["negatives"]:
            from sbv.common.dependencies import defer

            path = os.path.join(work, t["id"], "wrong_" + n["name"])
            defer(lambda path=path: shutil.rmtree(path, ignore_errors=True))
    by = {(j[0]["id"], j[1]): r for j, r in zip(jobs, res)}
    refused, units = [], []
    for t in tasks:
        why = []
        for s in range(t["jitter"] + 1):
            r = by[(t["id"], f"seed{s}")]
            run = "nominal run" if s == 0 else f"jittered run {s}"
            if "error" in r:
                why.append(f"the reference room's {run} stopped: {r['error']}")
            elif not r["success"]:
                why.append(f"the reference room fails its {run}: {r['why'] or 'no reason given'}")
            elif not r["finite"]:
                why.append(f"the reference room's {run} ends in a non-finite state")
            elif s and not r["jitter_applied"]:
                why.append(f"{run} moved nothing: run() must apply ctx['seed'] (sbv.package_tasks.api.jitter)")
            elif s and "error" not in by[(t["id"], "seed0")] and names(r) != names(by[(t["id"], "seed0")]):
                why.append(f"{run} reports other stages than the nominal run")
        for n in t["negatives"]:
            r = by[(t["id"], "wrong_" + n["name"])]
            if "error" in r:
                why.append(
                    f"wrong copy {n['name']} stopped instead of failing the check, which is no evidence the check catches it: {r['error']}"
                )
            elif r["success"]:
                why.append(f"wrong copy {n['name']} passes the check, so it cannot tell that copy ({n['why']})")
            elif "error" not in by[(t["id"], "seed0")] and names(r) != names(by[(t["id"], "seed0")]):
                why.append(f"wrong copy {n['name']} reports other stages than the nominal run")
        if frozen[t["id"]] != (code_files(room, t["entry"]), code_hash(room, t["entry"])):
            why.append("its code changed during calibration")
        if why:
            refused.append(f"package task {t['id']}: " + "; ".join(why))
            continue
        nom = by[(t["id"], "seed0")]
        units.append(
            dict(
                id=f"P1.{t['id']}",
                task=t["id"],
                kind="package",
                level=t["level"],
                words=t["words"],
                controller=t["controller"],
                engine=t["engine"],
                entry=t["entry"],
                code=frozen[t["id"]][0],
                code_sha256=frozen[t["id"]][1],
                objects=t["objects"],
                params=t["params"],
                timeout_s=t["timeout_s"],
                files=t["files"],
                inputs=input_files[t["id"]],
                stages=names(nom),
                truth=dict(metrics=nom["metrics"], seconds=nom["seconds"]),
                jitter=[by[(t["id"], f"seed{s}")]["metrics"] for s in range(1, t["jitter"] + 1)],
                negatives=[
                    dict(
                        name=n["name"],
                        why=n["why"],
                        **{k: n[k] for k in ("perturb", "nonrigid_set") if k in n},
                        metrics=by[(t["id"], "wrong_" + n["name"])]["metrics"],
                        score=by[(t["id"], "wrong_" + n["name"])]["score"],
                        failed_stage=first_failed(by[(t["id"], "wrong_" + n["name"])]),
                        failed_because=by[(t["id"], "wrong_" + n["name"])]["why"],
                    )
                    for n in t["negatives"]
                ],
            )
        )
    if any(input_identity(room, os.path.join(room, t["entry"]), t["files"]) != input_files[t["id"]] for t in tasks):
        refused.append("package calibration inputs changed during the run; completed artifacts retained")
    if refused:
        raise SystemExit("; ".join(refused))
    with open(out, "w") as f:
        json.dump(
            dict(schema=CASE_SCHEMA, room=os.path.relpath(room, os.path.dirname(os.path.abspath(out))), units=units),
            f,
            indent=1,
        )
    return units


def matches(units, room, copy, report):
    """{reference object: copy object or None} for every object the units use."""
    corr = (report or {}).get("correspondence") or {}
    from sbv.nonrigid.schema import side_file, upgrade

    side = side_file(room)
    nonrigid = {}
    if os.path.exists(side):
        every = upgrade(json.load(open(side))).get("objects", [])
        truth = [o for o in every if any(o.get("name") in u["objects"] for u in units)]
        if truth:
            from sbv.nonrigid import verify as Nonrigid, schema as SS

            cside = side_file(copy)
            copies, problems = SS.load(cside) if os.path.exists(cside) else ([], [])
            copies = [
                c
                for c in copies
                if not [
                    p for p in problems if p.startswith(repr(c.get("name"))) or p.startswith(str(c.get("name")) + ":")
                ]
            ]
            sup = Nonrigid.scene_supports(os.path.join(copy, "scene.xml"))
            kind = {o["name"]: o["kind"] for o in every}
            one = Nonrigid.pair(
                every, copies, room, copy, same_kind=False
            )  # the non-rigid verifiers' own correspondence
            nonrigid = {
                t: c["name"] for t, c in one.items() if c["kind"] == kind[t] and not Nonrigid.unbuildable(c, sup, copy)
            }
            nonrigid.update({o["name"]: nonrigid.get(o["name"]) for o in truth})
    return {o: (nonrigid[o] if o in nonrigid else corr.get(o)) for u in units for o in u["objects"]}


def body_centres(scene_xml):
    """{body name: (world centre at rest, has geometry)} of a scene, {} if it does not load."""
    import mujoco

    try:
        m = mujoco.MjModel.from_xml_path(scene_xml)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
    except Exception:
        return {}
    return {m.body(i).name: (d.xipos[i].copy(), int(m.body_geomnum[i]) > 0) for i in range(m.nbody) if m.body(i).name}


def by_role(objs, true_at, copy_at, claimed):
    """One task's objects with each one the pairing left unpaired standing for the copy body of its category word (its
    name without __<n>) nearest its true centre: first one no reference object claimed, else any; never one this task
    already uses, and only a body with geometry. Before, one unpaired object (telephone__10, cup__18) scored the whole
    task 0 without running it. Centres are the reference room's and the copy's own scenes at rest; the task's code,
    stages, efforts and bars are unchanged."""
    import numpy as np

    out = dict(objs)
    used = {c for c in out.values() if c}
    for o in [o for o, c in out.items() if not c]:
        if o not in true_at:
            continue
        w = re.sub(r"__\d+$", "", o)
        t0 = true_at[o][0]
        pool = [
            n for n, (_, geom) in copy_at.items() if geom and re.fullmatch(rf"{re.escape(w)}__\d+", n) and n not in used
        ]
        for cand in ([n for n in pool if n not in claimed], pool):
            if cand:
                n = min(cand, key=lambda n: float(np.linalg.norm(copy_at[n][0] - t0)))
                out[o] = n
                used.add(n)
                break
    return out


@operation
def evaluate(case_path, copy, report_path, out, procs=4, qualification=False):
    with open(case_path) as f:
        case = json.load(f)
    if case.get("schema") != CASE_SCHEMA:
        raise ValueError(f"{case_path}: not a {CASE_SCHEMA} case")
    room = os.path.join(os.path.dirname(os.path.abspath(case_path)), case["room"])
    copy = os.path.abspath(copy)
    from .input_bundle import identity as input_identity

    if any(
        u.get("inputs") != input_identity(room, os.path.join(room, u["entry"]), u.get("files", []))
        for u in case["units"]
    ):
        raise ValueError(
            "package case inputs are missing or changed; bind retained calibration input hashes or recalibrate affected tasks"
        )
    for u in case["units"]:  # the frozen code, or nothing: never a silently different check
        if (
            S.inside(room, u["entry"]) is None
            or code_files(room, u["entry"]) != u["code"]
            or code_hash(room, u["entry"]) != u["code_sha256"]
        ):
            raise ValueError(
                f"{u['id']}: the reference room's task code differs from the frozen case; calibrate the room again"
            )
    report = json.load(open(report_path)) if report_path and os.path.exists(report_path) else None
    loads = bool(report) and report.get("geometry") is not None
    found = matches(case["units"], room, copy, report) if loads else {}
    work = os.path.join(os.path.dirname(os.path.abspath(out)), "package_eval")
    jobs, rows = [], []
    true_at = copy_at = None
    claimed = set(((report or {}).get("correspondence") or {}).values())
    for u in case["units"]:
        objs = {o: found.get(o) for o in u["objects"]}
        if loads and not all(objs.values()):  # a role stands in, per task (see by_role)
            if true_at is None:
                true_at, copy_at = (
                    body_centres(os.path.join(room, "scene.xml")),
                    body_centres(os.path.join(copy, "scene.xml")),
                )
            objs = by_role(objs, true_at, copy_at, claimed)
        row = dict(
            id=u["id"],
            task=u["task"],
            kind="package",
            level=u["level"],
            controller=u["controller"],
            objects=objs,
            points=len(u["stages"]),
        )
        missing = [o for o, c in objs.items() if not c]
        if not loads:
            row.update(passed=False, score=0.0, why="the copy does not load")
        elif missing:
            row.update(passed=False, score=0.0, why=f"no object in the copy matches {missing}")
        else:
            jobs.append((u, row))
        rows.append(row)

    def trial(j):
        u, row = j
        entry = os.path.join(room, u["entry"])
        target = os.path.join(work, u["task"])
        ctx = context(copy, row["objects"], u["params"], 0, u.get("files", []), u["entry"])
        execute = lambda: run_one(entry, u["engine"], ctx, target, u["timeout_s"])
        if not qualification:
            return execute()  # live grading and oracles always execute
        from .calibration_cache import completed

        return completed(
            entry,
            u["engine"],
            ctx,
            target,
            u["timeout_s"],
            1,
            python(u["engine"]),
            execute,
            code_room=room,
            purpose="qualification",
        )

    with ThreadPoolExecutor(max(1, procs), initializer=use_session, initargs=(current_session(),)) as ex:
        res = list(ex.map(trial, jobs))
    for (u, row), r in zip(jobs, res):
        if "error" in r:
            row.update(passed=False, score=0.0, why=f"the task stopped on this copy: {r['error']}")
        elif names(r) != u["stages"]:
            row.update(passed=False, score=0.0, why=f"the run reports stages {names(r)}, not the frozen {u['stages']}")
        else:
            row.update(
                passed=bool(r["success"] and r["finite"]),
                score=r["score"],
                stages=r["stages"],
                failed_stage=first_failed(r),
                why=r["why"] if r["finite"] else "the run ends in a non-finite state",
                metrics=r["metrics"],
                seconds=r["seconds"],
            )
    rep = dict(
        schema=REPORT_SCHEMA,
        units=rows,
        package_tasks=round(100 * sum(r["score"] for r in rows) / len(rows), 1) if rows else None,
    )
    with open(out, "w") as f:
        json.dump(rep, f, indent=1)
    return rep


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["validate", "calibrate", "evaluate"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--report", help="evaluate: the copy's main report.json (its correspondence)")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "validate":
        tasks, problems = S.load(a.paths[0])
        print(json.dumps(dict(tasks=[t["id"] for t in tasks], problems=problems), indent=1))
        sys.exit(1 if problems else 0)
    if not a.out:
        ap.error("--out is required")
    if a.cmd == "calibrate":
        print(
            json.dumps(
                [{k: u[k] for k in ("id", "truth", "negatives")} for u in calibrate(a.paths[0], a.out, a.procs)],
                indent=1,
            )
        )
    else:
        print(json.dumps(evaluate(a.paths[0], a.paths[1], a.report, a.out, a.procs), indent=1))
