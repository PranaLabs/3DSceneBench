#!/usr/bin/env python
"""The fixed end-to-end run of the benchmark, the same for every room; the Docker image runs this. Per room only the
details differ (which objects, which tests, which robot tasks: all frozen in the room's case.json); the steps, the
folders and the report format do not.
    WORK/rooms/<room>/scene.xml (+ tasks.json, meshes, textures)   reference rooms from the room builder
    WORK/rooms/<room>/nonrigid.json                             optional: its non-rigid objects (scored in Genesis)
    WORK/rooms/<room>/tasks.json "package" (+ its code folder)     optional: tasks the room brings in its own code (F4.7)
    WORK/submissions/<room>/<entrant>/scene.xml (+ its files)      each entrant's copy of that room
    WORK/results/<room>/case.json                                  the room's frozen tests (calibrate), with its appearance floor
    WORK/results/<room>/check.json, self.json                      the room scored against itself: ready only if perfect
    WORK/results/<room>/entrants/<entrant>.json, <entrant>_renders/  each entrant's full report and renders
    WORK/results/<room>/nonrigid_case.json, entrants/<entrant>_nonrigid/ rooms with nonrigid.json: the frozen non-rigid
                                                                   verifiers, and each entrant's nonrigid_report.json
    WORK/results/<room>/package_case.json, entrants/<entrant>_package/  rooms with package tasks: their frozen package
                                                                   tasks (sbv/package_tasks), and each entrant's
                                                                   package_report.json
    WORK/results/summary.json, summary.md                          one row per room and entrant
Commands:
    python -m sbv.pipeline run [--work WORK]                   check every new or changed room, then score every submission
    python -m sbv.pipeline check ROOM --out DIR                one room: calibrate it, then score it against itself
    python -m sbv.pipeline score CASE SUBMISSION --out REPORT  one submission against one room's case
    python -m sbv.pipeline selftest                            the self-tests
"""

import argparse, glob, hashlib, json, os, platform, subprocess, sys
from collections import Counter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)  # the repository root
from sbv import VERSION, calibrate, evaluate, reward as R  # noqa: E402
from sbv.common.calibration_jobs import CalibrationIncomplete, atomic, identity, read
from sbv.common import case as K  # noqa: E402
from sbv.nonrigid import verify as Nonrigid  # noqa: E402
from sbv.package_tasks import verify as Pkg  # noqa: E402
from sbv.appearance import views as App  # noqa: E402
from sbv import qualification as Q

COLS = [
    ("room", "Room"),
    ("entrant", "Entrant"),
    ("reward", "Overall"),
    ("reward_visible", "Overall (visible)"),
    ("parse", "Parse"),
    ("sim_ready", "Sim-ready"),
    ("reward_appearance", "A: appearance"),
    ("reward_geometry", "G: geometry"),
    ("reward_functionality", "F: functionality"),
    ("reward_functionality_hint", "F (hint)"),
    ("reward_functionality_reference_bar", "F (reference bars)"),
    ("reward_balance", "Balance"),
    ("reward_response", "Response"),
    ("reward_affordance", "Affordance"),
    ("reward_interaction", "Interaction"),
    ("siglip2", "SigLIP-2"),
    ("dinov3", "DINOv3"),
    ("lpips", "LPIPS"),
    ("inventory", "Inventory"),
    ("shape", "Shape"),
    ("pose_size", "Pose-size"),
    ("relations", "Relations"),
    ("placement", "Placement"),
    ("navigation", "Navigation"),
]


def version():
    """What a frozen check depends on besides the room: the verifier code, the robot model, the engine and the machine
    type (physics results differ slightly between, say, an arm64 Mac and an amd64 container)."""
    import mujoco
    from sbv.functionality.f4_robot import rby1

    h = hashlib.sha256(f"{mujoco.__version__}|{platform.machine()}".encode())
    for p in sorted(glob.glob(os.path.join(HERE, "sbv", "**", "*.py"), recursive=True)) + [
        os.path.join(rby1.ROBOT_DIR, "rby1a_1.2.xml")
    ]:
        if os.path.exists(p):
            with open(p, "rb") as f:
                h.update(f.read())
    return h.hexdigest()[:16]


def write(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))


def result_files(out):
    """Finished evidence, not mutable checkpoints, entrant runs or check.json itself."""
    from pathlib import Path
    from sbv.common.dependencies import sha

    root = Path(out)
    names = ("case.json", "self.json", "nonrigid_case.json", "package_case.json")
    paths = [root / n for n in names if (root / n).is_file()]
    for name in ("self_nonrigid", "self_package", "self_renders"):
        paths += [p for p in (root / name).rglob("*") if p.is_file()]
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(paths)}


def reusable(old, room, out, calib=None):
    """Never promote an existence-only legacy receipt to verified evidence."""
    try:
        if (
            not old
            or old.get("status") in ("running", "interrupted", "error")
            or old.get("retry")
            or old.get("inputs") != identity(room, calib or {})
            or old.get("version") != version()
            or "artifacts" not in old
            or old["artifacts"] != result_files(out)
        ):
            return False
        if old.get("ready"):
            if old["inputs"].get("weights") is None and any(
                (old.get("self") or {}).get(k) is not None for k in ("siglip2", "dinov3", "lpips")
            ):
                return False  # A measured visual pass needs identifiable metric weights before reuse.
            required = ["case.json", "self.json"]
            if os.path.isfile(Nonrigid.side_file(room)):
                required += ["nonrigid_case.json", "self_nonrigid/nonrigid_report.json"]
            if Pkg.S.has(room):
                required += ["package_case.json", "self_package/package_report.json"]
            if not all(n in old["artifacts"] for n in required):
                return False
            K.reference_path(K.load(os.path.join(out, "case.json")))
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def scores(r, nonrigid=None, package=None):
    """The headline numbers of one report, with the same keys for every room. nonrigid: the copy's nonrigid_report.json,
    if the room has non-rigid objects; package: its package_report.json, if the room has package tasks (they count in
    their level like any unit). reward (the leaderboard's Overall): one number, the mean of appearance, geometry and
    functionality, each on 0 to 100 (sbv/reward.py), on every reference object; reward_visible on the ones the pictures
    show; parse and sim_ready (0 or 1, rates over rooms on the leaderboard); reward_functionality, the F column, the
    mean of the four levels; the geometry readouts from the full scope (sbv/scope.py)."""
    sc = r.get("scopes") or {}
    full = sc.get("full") or r
    a = (r.get("appearance") or {}).get("mean") or {}
    g = full["geometry"] if r.get("geometry") else None
    rows = list((nonrigid or {}).get("units") or ()) + list((package or {}).get("units") or ())
    pct = lambda x: None if x is None else round(100 * x, 1)
    rw, parts = R.reward(r, rows)
    out = dict(
        reward=rw,
        **{f"reward_{k}": v for k, v in parts.items()},
        parse=float(r.get("geometry") is not None),  # the files load (the copy can be measured at all)
        sim_ready=float(bool((r.get("F1") or {}).get("passed"))),  # and it runs: the whole gate passed
        reward_functionality=R.functionality(parts),
        reward_functionality_hint=R.functionality(R.parts(r, rows, setting="physics_hint")),
        reward_functionality_reference_bar=R.functionality(R.parts(r, rows, bar=1)),
        siglip2=a.get("siglip2"),
        dinov3=a.get("dinov3"),
        lpips=a.get("lpips_vgg"),
        **{
            f"input_{'lpips' if k == 'lpips_vgg' else k}": v
            for k, v in (((r.get("appearance") or {}).get("input") or {}).get("mean") or {}).items()
        },
        failed_units=sorted(k for k, v in r["units"].items() if not all(x["passed"]["physics_hint"] for x in v)),
        units_not_run=len((r.get("incomplete") or {}).get("units", [])),  # above 0: a grader worker died, rerun
        not_still_share=(r.get("at_rest") or {}).get("share"),
        not_still_flag=float(bool((r.get("at_rest") or {}).get("flag"))),
        **R.appearance(a, (r.get("appearance") or {}).get("floor")),
    )
    if sc or g is None:  # the visible scope (0 for a copy that does not load), C and Q (sbv/scope.py)
        rv, pv = R.reward(r, rows, scope="visible")
        out.update(reward_visible=rv, **{f"reward_visible_{k}": v for k, v in pv.items()})
        out.update(
            {f"{n}_{k}": pct(sc[k][x]) for k in sc for n, x in (("coverage", "C"), ("quality", "Q"))},
            groups=(r.get("groups") or {}).get("score"),
        )
    if (
        g
    ):  # the geometry readouts on the rigid objects (reward_geometry also holds the non-rigid sizes and fluid amounts)
        out.update(
            inventory=pct(g["G1"]["inventory"]["f1"]),
            shape=pct(g["G2"]["fscore"]),
            pose_size=pct(g["G2"]["pose_size"]["score"]),
            pose_size_passed=pct(g["G2"]["pose_size"].get("score_passed")),
            pose_size_reference_bar=pct(g["G2"]["pose_size"].get("score_reference_bar")),
            relations=pct(g["G3"]["relations"]["score"]),
            placement=pct(g["G3"]["placement"]["score"]),
            navigation=pct(g["G3"]["navigation"]["score"]),
            **{
                k: None if v is None else round(v, 1)
                for k, v in (g.get("absolute") or {}).items()
                if k.endswith(("_pct", "_cm"))
            },
        )
    return out


def reward_file(report, out, nonrigid=None, package=None):
    """Harbor's reward.json from report.json (and nonrigid_report.json, when the room has non-rigid objects, and
    package_report.json, when it has package tasks). When grading itself failed (a worker died, a unit raised, or no
    report was written, the package report included), status grader_failed with the reason and no reward, also written
    into report.json, so the run is repeated, not scored 0. Returns the exit code."""
    try:
        r = json.load(open(report)) if os.path.exists(report) else None
    except ValueError:
        r = None  # cut short while it was being written: as if not written
    if r is None:
        r = dict(
            status="grader_failed",
            failure=f"no {os.path.basename(report)}: the grader stopped before writing it (see summary.txt)",
        )
        write(report, r)
    if r.get("status") == "grader_failed":
        write(out, dict(status="grader_failed", reason=r.get("failure")))
        return 1
    for path, what in (
        (nonrigid, "non-rigid"),
        (package, "package task"),
    ):  # a grader that stopped: never scored as if the room had no such units
        if path is not None and not os.path.exists(path):
            r.update(
                status="grader_failed",
                failure=f"no {os.path.basename(path)}: the {what} grader stopped (see {os.path.basename(path).replace('_report.json', '_summary.txt')})",
            )
            write(report, r)
            write(out, dict(status="grader_failed", reason=r["failure"]))
            return 1
    a = r.get("appearance") or {}
    if a.get("status") and not a.get("copy_render_failed"):
        # the grader's own trouble (no metrics environment, a failed reference-room render): no reward, so the run is
        # graded again; a mean of geometry and functionality alone would not compare with any other result
        r.update(status="grader_failed", failure=f"appearance could not be computed: {a['status']}")
        write(report, r)
        write(out, dict(status="grader_failed", reason=r["failure"]))
        return 1
    d = None if nonrigid is None else json.load(open(nonrigid))
    p = None if package is None else json.load(open(package))
    numbers = {k: float(v) for k, v in scores(r, d, p).items() if isinstance(v, (int, float))}
    r["verifier"] = dict(version=VERSION, code=version())
    write(report, r)
    write(out, {"reward": 0.0, **numbers, "verifier_version": float(VERSION)})
    return 0


def score(case_path, submission, report, procs=8, log=print):
    r = evaluate.evaluate(
        K.load(case_path), submission, procs, log=log, renders=os.path.splitext(report)[0] + "_renders"
    )
    write(report, r)
    if r.get("status") == "grader_failed":
        raise RuntimeError(f"grader failed: {r['failure']}")  # check() retries; run() notes it
    nonrigid = Nonrigid.side_file(os.path.dirname(case_path), Nonrigid.CASE)
    s = None
    if os.path.exists(nonrigid):  # the room has non-rigid objects: Genesis, 4 threads a run
        d = os.path.splitext(report)[0] + "_nonrigid"
        os.makedirs(d, exist_ok=True)
        s = Nonrigid.evaluate(nonrigid, submission, os.path.join(d, "nonrigid_report.json"), max(1, procs // 4))
    pk = os.path.join(os.path.dirname(case_path), "package_case.json")
    p = None
    if os.path.exists(pk):  # the room's package tasks, on the copy's matched objects
        d = os.path.splitext(report)[0] + "_package"
        os.makedirs(d, exist_ok=True)
        p = Pkg.evaluate(pk, submission, report, os.path.join(d, "package_report.json"), max(1, procs // 2))
    return scores(r, s, p)


@Q.authoring
def check(room, out, procs=8, log=print, calib=None, job_timeout=3600, resume=True):
    """Calibrate the room, then score it against itself. Ready only if calibration succeeds (it needs working robot
    tasks) and the room scores 100 on everything it is scored on, with renders that match (SigLIP-2, DINOv3 >=
    0.999)."""
    os.makedirs(out, exist_ok=True)
    res = dict(
        room=os.path.basename(os.path.normpath(room)), fingerprint=K.fingerprint(room), version=version(), ready=False
    )
    res.update(status="running", stage="calibration")
    atomic(os.path.join(out, "check.json"), res)
    options = dict(checkpoint=os.path.join(out, "calibration"), job_timeout=job_timeout, resume=resume)
    options.update(calib or {})
    try:
        res["inputs"] = identity(room, calib or {})
        case = Q.calibration(room, out, options, procs, log, resume)  # calib: test-only overrides (draws, jitter)
        case["reference"] = os.path.relpath(K.scene_xml(os.path.abspath(room)), os.path.abspath(out))
        K.save(case, os.path.join(out, "case.json"))
        fl = Q.floor(
            K.load(os.path.join(out, "case.json")), out, log, resume
        )  # the room's architecture alone, scored against the room: appearance 0
        if fl:
            case["appearance"]["floor"] = fl
            K.save(case, os.path.join(out, "case.json"))
            log(f"appearance floor: {fl}")
        else:
            log(
                "appearance floor: not computed here, so appearance keeps the fixed mapping (check the room inside the image, or run sbv.appearance.views --floor)"
            )
        nonrigid = os.path.join(out, "nonrigid_case.json")
        if os.path.exists(Nonrigid.side_file(room)):
            Q.nonrigid_calibration(room, out, max(1, procs // 4), resume)  # score() then uses it
        elif os.path.exists(nonrigid):
            os.remove(nonrigid)  # the room no longer has non-rigid objects
        pk = os.path.join(out, "package_case.json")
        if Pkg.S.has(room):
            Pkg.calibrate(room, pk, max(1, procs // 2))  # tasks.json "package": refuses with reasons
        elif os.path.exists(pk):
            os.remove(pk)  # the room no longer has package tasks
        res.update(stage="self_score")
        atomic(os.path.join(out, "check.json"), res)
        s = Q.score(os.path.join(out, "case.json"), room, out, procs, log, resume)
        if res["inputs"] != identity(room, calib or {}):
            raise RuntimeError("qualification inputs changed during the check; completed job evidence is retained")
    except KeyboardInterrupt:
        res.update(status="interrupted", retry=True, reason="interrupted by user")
        atomic(os.path.join(out, "check.json"), res)
        raise
    except (SystemExit, Exception) as e:  # SystemExit: calibrate's refusal of the room, with its reasons
        res["reason"] = str(e) if isinstance(e, SystemExit) else f"{type(e).__name__}: {str(e)[:500]}"
        res["status"] = (
            e.status if isinstance(e, CalibrationIncomplete) else "failed" if isinstance(e, SystemExit) else "error"
        )
        res["retry"] = not isinstance(e, SystemExit)  # any other error is checked again on the next run
        res["artifacts"] = result_files(out)
        write(os.path.join(out, "check.json"), res)
        return res
    perfect = [
        k
        for k in ("reward", "reward_visible", "reward_geometry", "reward_functionality", "reward_functionality_hint")
        + tuple(f"reward_{k}" for k in R.LEVEL_ORDER)
        + ("inventory", "shape", "pose_size", "relations", "placement", "navigation")
        + tuple(k for k in s if k.endswith("_reference_bar"))  # at both pass bars
        if s.get(k) is not None and s[k] < 99.95
    ]
    problems = (["the gate fails"] if not s["sim_ready"] else []) + [f"{k} is {s[k]} against itself" for k in perfect]
    problems += [
        f"{k} is {s[k]} against itself" for k in ("siglip2", "dinov3") if s.get(k) is not None and s[k] < 0.999
    ]
    units = case["functionality"]["units"]
    res.update(
        ready=not problems,
        status="passed" if not problems else "failed",
        stage="complete",
        problems=problems,
        self=s,
        units=dict(Counter(u["kind"] for u in units)),
        robot_tasks=[u["id"] for u in units if u["kind"].startswith("robot")],
        excluded=len(case["functionality"]["excluded"]),
        **({"package_tasks": [u["id"] for u in json.load(open(pk))["units"]]} if os.path.exists(pk) else {}),
    )
    res["artifacts"] = result_files(out)
    write(os.path.join(out, "check.json"), res)
    from sbv.common.dependencies import published

    def invalidate():
        res.update(ready=False, status="error", retry=True, reason="dependencies changed before check publication")
        write(os.path.join(out, "check.json"), res)

    published(None, check=lambda: res["inputs"] == identity(room, calib or {}), invalidate=invalidate)
    return res


def table(rows):
    fmt = lambda v: "" if v is None else "pass" if v is True else "FAIL" if v is False else str(v)
    lines = ["| " + " | ".join(h for _, h in COLS) + " | Note |", "|" + "---|" * (len(COLS) + 1)]
    for r in rows:
        note = r.get("note") or (
            f"failed: {', '.join(r['failed_units'][:6])}{' ...' if len(r['failed_units']) > 6 else ''}"
            if r.get("failed_units")
            else ""
        )
        lines.append("| " + " | ".join(fmt(r.get(k)) for k, _ in COLS) + f" | {note} |")
    return "\n".join(lines) + "\n"


def unreadable(rd):
    """Why an entry under WORK/rooms cannot be checked, or None. Such a room used to be left out without a word."""
    if os.path.islink(rd) and not os.path.exists(rd):
        return (
            f"a link to {os.readlink(rd)}, which does not exist here (in Docker, copy the room's files in, not a link)"
        )
    if not os.path.isdir(rd):
        return "not a folder"
    if not os.access(rd, os.R_OK | os.X_OK):
        return "the folder cannot be read"
    if not os.path.isfile(os.path.join(rd, "scene.xml")):
        return "no scene.xml in the folder"
    return None


def run(work, procs=8, log=print, calib=None, job_timeout=3600, resume=True):
    rooms, subs, results = (os.path.join(work, d) for d in ("rooms", "submissions", "results"))
    rows = []
    for room in sorted(d for d in os.listdir(rooms) if not d.startswith(".")):
        rd, out = os.path.join(rooms, room), os.path.join(results, room)
        path = os.path.join(out, "check.json")
        why = unreadable(rd)
        if why:
            log(f"{room}: skipped: {why}")
            rows.append(dict(room=room, entrant="(the room itself)", note=f"skipped: {why}"))
            continue
        old = read(path)
        if resume and reusable(old, rd, out, calib):
            c = old
            log(f"{room}: unchanged since its last check")
        else:
            log(f"{room}: checking")
            c = check(rd, out, procs, log, calib, job_timeout, resume)
        rows.append(
            dict(
                room=room,
                entrant="(the room itself)",
                **(c.get("self") or {}),
                note=c.get("reason") or "; ".join(c.get("problems", [])) or "ready",
            )
        )
        if not c["ready"]:
            log(f"{room}: not ready, so its submissions are not scored: {rows[-1]['note']}")
            continue
        sd = os.path.join(subs, room)
        for ent in sorted(d for d in os.listdir(sd) if os.path.isdir(os.path.join(sd, d))) if os.path.isdir(sd) else []:
            log(f"{room}/{ent}: scoring")
            os.makedirs(os.path.join(out, "entrants"), exist_ok=True)
            try:
                row = score(
                    os.path.join(out, "case.json"),
                    os.path.join(sd, ent),
                    os.path.join(out, "entrants", ent + ".json"),
                    procs,
                    log,
                )
            except Exception as e:
                row = dict(
                    note=f"scoring stopped: {type(e).__name__}: {str(e)[:300]}"
                )  # one broken copy never stops the run
            rows.append(dict(room=room, entrant=ent, **row))
    os.makedirs(results, exist_ok=True)
    write(os.path.join(results, "summary.json"), rows)
    with open(os.path.join(results, "summary.md"), "w") as f:
        f.write(table(rows))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    procs = min(8, os.cpu_count() or 1)
    p = sub.add_parser("run")
    p.add_argument("--work", default="/work")
    p.add_argument("--procs", type=int, default=procs)
    p.add_argument("--job-timeout", type=float, default=3600)
    p.add_argument("--no-resume", action="store_true")
    p = sub.add_parser("check")
    p.add_argument("room")
    p.add_argument("--out", required=True)
    p.add_argument("--procs", type=int, default=procs)
    p.add_argument("--job-timeout", type=float, default=3600)
    p.add_argument("--no-resume", action="store_true")
    p = sub.add_parser("score")
    p.add_argument("case")
    p.add_argument("submission")
    p.add_argument("--out", required=True)
    p.add_argument("--procs", type=int, default=procs)
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.work, a.procs, job_timeout=a.job_timeout, resume=not a.no_resume)
        print(open(os.path.join(a.work, "results", "summary.md")).read())
    elif a.cmd == "check":
        r = check(a.room, a.out, a.procs, job_timeout=a.job_timeout, resume=not a.no_resume)
        print(json.dumps(r, indent=1, default=str))
        sys.exit(0 if r["ready"] else 1)
    elif a.cmd == "score":
        print(json.dumps(score(a.case, a.submission, a.out, a.procs), indent=1, default=str))
    else:
        sys.exit(subprocess.call([sys.executable, "-m", "unittest", "tests.test_verifiers", "-v"], cwd=HERE))
