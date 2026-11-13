"""Non-rigid verifiers: calibrate on the reference room, score a copy. Genesis on CPU, one process per run.

  python -m sbv.nonrigid.verify calibrate ROOM_DIR --out nonrigid_case.json [--seed N]
  python -m sbv.nonrigid.verify evaluate nonrigid_case.json COPY_DIR --out nonrigid_report.json
Each Genesis run uses the Python in SBV_NONRIGID_PYTHON (default: this one), so these commands can run from .venv while
Genesis lives in .venv-nonrigid (requirements/nonrigid.txt).

A unit = one object x one hidden action, plus, per fluid, whether it stays in its holder at rest (contained(), read
from the same run before the action starts), and per fluid and soft body its amount or size.

Hidden actions. Calibration runs the reference object three times: as built, and with one property (SENSE: the amount of
fluid, the stiffness, the cloth's weight per square metre) halved and doubled. A spill is read as the fluid poured as a
share of the true amount. Scoring passes a copy when its readout is within a factor of 2 of the truth either way, or
within the kind's FLOOR of it: a yes-or-no threshold, the rigid tests' factor-2 bar. A unit whose half and double runs
would both pass cannot tell a wrong copy from a right one: it is marked not decidable and does not count. A thin shell's
strip is pulled up (the swing before it is retired, RETIRED), and its calibration swaps its fabric for one of its own
class and one of the other instead of halving and doubling. Each object's hidden action is frozen in the case: scoring
reads it from there only.
"""

import argparse
import json
import re
import os
import secrets
import subprocess
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .schema import (
    CASE,
    FABRIC,
    FABRIC_CLASS,
    FIXED,
    HELD_SDF_MAX_RES,
    SETTLE_MU,
    VISCOSITY,
    bounds,
    center,
    load,
    sheet,
    side_file,
    upgrade,
    volume_ml,
)

CLOTH_RULES = "rigid friction, bending about the starting shape, settling 0.05 s, SX.7 at 1.5 m/s"


def dims_cm(o, folder):
    """Outer size in cm, longest side first, in the object's own frame (a turned pillow is not a bigger one)."""
    lo, hi = bounds(dict(o, shape=dict(o["shape"], yaw_deg=0)), folder)
    return sorted((round(float(x) * 100, 1) for x in np.asarray(hi) - np.asarray(lo)), reverse=True)


# The ranges each kind's hidden action is drawn from (public). Calibration draws one action per reference object from
# the room's seed and the object's name, and freezes it in nonrigid_case.json; scoring reads it from there. (lo, hi) is
# drawn evenly, rounded to STEP; a list is one choice from it. A fluid tips toward one of the room's four axes; a
# cloth's hem is pushed toward its front (+y) within 30 degrees; a soft body's ball lands within 0.3 half-widths of its
# middle.
RANGES = {
    "fluid": dict(
        yaw_deg=[0, 90, 180, 270], angle_deg=(60, 90), t_start=(0.4, 0.6), t_ramp=(0.4, 0.8), duration=(2.0, 2.5)
    ),
    "soft_body": dict(offset=[(-0.3, 0.3), (-0.3, 0.3)], height=(0.15, 0.35), duration=(1.0, 1.4)),
    "thin_shell": dict(
        t_start=(0.4, 0.6), yaw_deg=[0, 90, 180, 270], duration=(2.0, 2.5)
    ),  # the edge a lying cloth is pulled by
}
STEP = dict(angle_deg=1, yaw_deg=1, t_start=0.05, t_ramp=0.05, duration=0.1, offset=0.05, height=0.01, impulse=0.01)
# reference room's own nonrigid.json (private) may give an object its own "action" (its keys win over the draw, e.g. a
# flat weight instead of the ball). A "nudge" ({"<key>": step}) is still read and checked but no longer used:
# calibration stopped nudging actions, when the hidden actions became a factor-2 threshold.
NUDGE = {"fluid": ("angle_deg", 2.0), "soft_body": ("height", 0.02), "thin_shell": ("t_start", 0.05)}
SENSE = {"fluid": "amount", "soft_body": "youngs_kpa", "thin_shell": "areal_density"}
BAND = (
    2.0  # a hidden action passes when the copy's readout is within a factor of 2 of the truth either way, or within the
)
# ind's FLOOR of it
CAL_RUNS = ("base", "half", "double")
RETIRED = {
    "thin_shell": "S1.3 cloth swing, retired: the swing did not follow the cloth's weight, so it could not tell a wrong cloth from a right one"
}
LIFT_SHARE = 0.5  # the pull: half the true cloth's weight, frozen in newtons
SWAP = {
    "silk": ("cotton", "denim"),
    "cotton": ("silk", "denim"),
    "denim": ("wool_coat", "cotton"),
    "wool_coat": ("denim", "cotton"),
}  # (own class, other)


def retired(u):
    """A unit of a retired action: the cloth swing, any thin shell unit without the lift's force."""
    return u["kind"] in RETIRED and "force_n" not in (u.get("action") or {})


def probes(o):
    """{label: scale} of an object's calibration runs beside the base: a thin shell's fabric swapped within and across
    its class, any other object's SENSE property halved and doubled."""
    if o["kind"] == "thin_shell":
        return dict(zip(("same", "other"), ({"fabric": f} for f in SWAP[o["material"]["fabric"]])))
    return {k: {SENSE[o["kind"]]: v} for k, v in SCALE.items()}


# A retired kind gets no hidden action: calibration makes no unit for it and scoring never counts one from an older
# case. Its objects may still be described in the side file; the cloth simulation stays for a test that replaces the
# swing.
SCALE = dict(half=0.5, double=2.0)  # the property's factor in each calibration run


def poured(o, run, scale=1.0):
    """A run's readout; a spill's as the fluid poured as a share of the TRUE amount."""
    return run["readout"] * scale if o["kind"] == "fluid" else run["readout"]


def within(r, truth, kind, floor=None):
    """The yes-or-no threshold: r within a factor of BAND of the truth either way, or within floor of it: the reference
    object's own measured repeatability, else the kind's FLOOR."""
    return bool(truth / BAND <= r <= truth * BAND or abs(r - truth) <= (FLOOR[kind] if floor is None else floor))


# epeatability. Calibration runs the reference object twice more with the action moved a little (a nonrigid solid's
# landing spot by 0.1 half-widths, a fluid's tilt by 3 degrees): the spread of those readouts is the readout's own
# noise. It replaces the fixed FLOOR, and a unit whose readout is not repeatable (a repeat more than REPEAT_MAX times
# off the truth) does not count.
JITTER = {
    "soft_body": [dict(offset=(0.1, 0.0)), dict(offset=(0.0, -0.1))],
    "fluid": [dict(angle_deg=3.0), dict(angle_deg=-3.0)],
    "thin_shell": [dict(t_start=0.1), dict(yaw_deg=20.0)],
}
REPEAT_MAX = 2.0 ** (1 / 3)  # a third of the factor-2 bar, on a log scale


def jittered(action, kind, k):
    """The action of repeat k: offsets add to the drawn offset, angles to the drawn angle."""
    a, d = dict(action), JITTER[kind][k]
    for key, v in d.items():
        a[key] = (
            [round(x + y, 4) for x, y in zip(a.get(key, [0, 0]), v)]
            if isinstance(v, tuple)
            else round(a.get(key, 0) + v, 4)
        )
    return a


AMOUNT_BAR = 2.0  # a fluid's amount passes within a factor of 2 of the truth either way, like every
# amount guessed from pictures: push, friction, grip); before, within 30%. A case's stored tolerance
# for an amount unit is no longer read


def amount_ok(v, truth):
    """The copy's fluid volume against the true one: within AMOUNT_BAR times either way."""
    return bool(truth / AMOUNT_BAR <= v <= truth * AMOUNT_BAR)


FLOOR = {"fluid": 0.08, "soft_body": 0.008, "thin_shell": 0.015}
HELD, REST_MPS = (
    0.95,
    0.05,
)  # containment: at least 95% of the fluid inside its holder, moving under 5 cm/s on average over the 0.2 s before the action (actions.REST_WINDOW)
CAP_S = 600  # CPU-track rule: at most 10 minutes of computing per hidden action
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repository root


THREADS = os.environ.get(
    "NONRIGID_THREADS", "4"
)  # Genesis CPU threads per run; scoring uses the case's (fluid readouts change with it)
VIDEO = os.environ.get("NONRIGID_VIDEO", "0") == "1"  # off by default: a GPU-less grader may not draw
PYTHON = (
    os.environ.get("SBV_NONRIGID_PYTHON") or os.environ.get("SBV_SOFT_PYTHON") or sys.executable
)  # the Python with Genesis


from sbv.common.dependencies import operation, current_session, use_session


def _run_nonrigid(scene_xml, obj, action, scale, work, video, threads, cap=None):
    cap = cap or CAP_S
    t = time.time()
    job = json.dumps(dict(scene=scene_xml, obj=obj, action=action, scale=scale, work=work, video=video))
    try:
        r = subprocess.run(
            [
                PYTHON,
                "-c",
                "import json,sys,genesis as gs; from sbv.nonrigid.world import run; gs.init(backend=gs.cpu, "
                "logging_level='error', seed=0); j=json.loads(sys.stdin.read()); print('RESULT', json.dumps(run(j['scene'], j['obj'], "
                "j['action'], j['work'], j['video'], j['scale'])))",
            ],
            input=job,
            cwd=HERE,
            capture_output=True,
            text=True,
            timeout=cap,  # the job on stdin: one argument holds at most 128 KB on Linux, and the object is a copy's
            env=dict(os.environ, QD_NUM_THREADS=str(threads)),
        )
    except subprocess.TimeoutExpired:
        return dict(error=f"over the {cap:g} s compute cap", over_cap=True)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    errs = "\n".join(
        ln for ln in (r.stderr or "").splitlines() + r.stdout.splitlines() if "warn" not in ln.lower()
    )  # Genesis logs errors to stdout
    out = json.loads(lines[-1][7:]) if lines else dict(error=(errs or "no result")[-600:])
    out["seconds"] = round(time.time() - t, 1)
    return out


@operation
def _one(scene_xml, obj, action, scale, work, video=None, threads=THREADS, reuse=True, cap=None):
    from pathlib import Path
    from sbv.common.dependencies import runtime, environment, PHYSICS_ENV, SCOPES, sha, model_files
    from sbv.common.code_dependencies import native
    from sbv.common.scene_inputs import physical
    from sbv.common.phase_cache import cached

    def identity():
        sh = obj.get("shape") or {}
        mesh = (
            sh.get("file")
            if sh.get("type") == "mesh"
            and isinstance(sh.get("file"), str)
            and os.path.basename(sh["file"]) == sh["file"]
            else None
        )  # only a mesh shape names a file, next to the scene (schema.entry_problems)
        return dict(
            schema=3,
            model=physical(scene_xml),
            obj=obj,
            action=action,
            scale=scale,
            mesh=sha(Path(scene_xml).parent / mesh) if mesh else None,
            runtime=runtime(PYTHON, packages=SCOPES["nonrigid"]),
            environment=environment(PHYSICS_ENV),
            code=native("nonrigid"),
            threads=str(threads),
            cap_s=cap or CAP_S,
            video=model_files(scene_xml) if video else None,
        )

    out = Path(work).parent
    names = [Path(work).name]
    if video and Path(video).parent.resolve() != out.resolve():
        reuse = False
    if video:
        names.append(Path(video).name)
    result = cached(
        out,
        "nonrigid",
        identity,
        lambda: _run_nonrigid(scene_xml, obj, action, scale, work, video, threads, cap),
        names,
        accept=lambda r: "error" not in r and r.get("finite") is True,
        root=os.environ.get("SBV_NONRIGID_CACHE") or out / ".nonrigid-cache",
        resume=reuse,
    )
    return result


def safety(obj, r, folder="."):
    """Broken rules, as text; empty means safe. A cloth not at rest (SX.7) is not one: it fails its behaviour only,
    since copies a few percent over the bar would otherwise score 0."""
    bad = [] if "error" not in r or "start_speed_mps" in r else [r["error"]]
    if r.get("finite") is False:
        bad.append("blew up")
    s = obj["shape"]
    if obj["kind"] == "soft_body" and "ball_lowest" in r:
        bottom = float(bounds(obj, folder)[0][2])
        if r["ball_lowest"] < bottom + 0.005:
            bad.append("ball went through")
        if r["object_lowest"] < bottom - 0.02:
            bad.append("fell through its support")
    if obj["kind"] == "thin_shell" and "top_z" in r and r["top_z"] < r["rail_z"] - 0.05:
        bad.append("fell off the rail")
    return bad


def contained(r, rest_mps=REST_MPS):
    """Why a fluid did not stay put in its holder before the action started: the run failed or blew up, under HELD of
    it is inside the holder's real inside, or it still moves faster than rest_mps on average (a fluid flying off or
    draining through a wall is not at rest). Empty means it stayed put. A run without a rest readout has no verdict."""
    bad = [] if "error" not in r else [r["error"]]
    if r.get("finite") is False:
        bad.append("blew up")
    h, v = r.get("held_at_rest"), r.get("rest_speed")
    if h is None and not bad:
        bad.append("no rest readout")
    elif h is not None and h < HELD:
        bad.append(f"leaks at rest: {h:.0%} inside")
    if v is not None and v > rest_mps:
        bad.append(f"not at rest: moving {v:.2f} m/s before the action")
    return bad


def readout_problems(obj, run, folder):
    """Why a calibration run cannot serve as evidence: a broken safety rule, no finite readout, or no finite state. A
    failed nudge run must never become an infinite tolerance that passes every copy."""
    problems = safety(obj, run, folder) + ([run["error"]] if "error" in run and "start_speed_mps" in run else [])
    value = run.get("readout")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        problems.append("missing or non-finite readout")
    if run.get("finite") is not True:
        problems.append("finite state was not confirmed")
    return problems


def _draw(spec, rng, step):
    if isinstance(spec, list) and not isinstance(spec[0], tuple):
        return spec[int(rng.integers(len(spec)))]
    if isinstance(spec, list):
        return [_draw(s, rng, step) for s in spec]
    return round(round(float(rng.uniform(*spec)) / step) * step, 4)


def default_action(kind, name, seed):
    """The action drawn from the kind's RANGES; the same seed and name always give the same one."""
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
    return {k: _draw(spec, rng, STEP[k]) for k, spec in RANGES[kind].items()}


def action_of(o, seed):
    """(action, nudge) for a reference object: the drawn action under its own "action" block, its own "nudge" or its
    kind's."""
    own, n = o.get("action", {}), o.get("nudge")
    if not isinstance(own, dict):
        raise SystemExit(f"{o['name']}: action must be an object of action keys, not {own!r}")
    a = {**default_action(o["kind"], o["name"], seed), **own}
    if n is not None and not (
        isinstance(n, dict) and len(n) == 1 and next(iter(n)) in a and isinstance(next(iter(n.values())), (int, float))
    ):
        raise SystemExit(f"{o['name']}: nudge must be one {{key: step}} of its action, not {n!r}")
    return a, (next(iter(n.items())) if n else NUDGE[o["kind"]])


def reference_inputs(room, objects):
    """Files a frozen non-rigid reference reads, excluding its generated cases and reports."""
    from sbv.common.dependencies import sha
    from sbv.common.scene_inputs import physical
    from pathlib import Path

    xml = Path(room) / "scene.xml"
    files = dict(model=physical(xml)) if xml.is_file() else {}
    side = Path(
        side_file(room)
    )  # keyed by the file's own name: a case frozen on soft_liquid.json still matches its room
    if side.is_file():
        files[side.name] = sha(side)
    for obj in objects:
        file = obj.get("shape", {}).get("file")
        if file:
            files[file] = sha(Path(room) / file)
    return files


@operation
def calibrate(room, out, procs=5, seed=None):
    """seed: draws the actions of objects without their own; default the side file's "seed", else a fresh one."""
    from sbv.common.dependencies import tree

    input_files = tree(room)
    objs, problems = load(side_file(room))
    assert not problems, problems
    from .declare import missing

    undeclared = missing(room)  # a warning for the room's builder (python -m sbv.nonrigid.declare drafts entries)
    if undeclared:
        print(
            f"non-rigid calibration: {len(undeclared)} bodies look non-rigid but are not in nonrigid.json: "
            + ", ".join(f"{n} ({k})" for n, k in undeclared[:20])
            + (" ..." if len(undeclared) > 20 else ""),
            file=sys.stderr,
        )
    if seed is None:
        seed = json.load(open(side_file(room))).get("seed", secrets.randbelow(2**31))
    xml = os.path.abspath(os.path.join(room, "scene.xml"))
    work = os.path.abspath(os.path.join(os.path.dirname(out) or ".", "nonrigid_cal"))
    acts = {o["name"]: action_of(o, seed) for o in objs}
    for o in objs:
        if o["kind"] == "thin_shell":
            acts[o["name"]][0].setdefault(
                "force_n", round(LIFT_SHARE * 9.81 * FABRIC[o["material"]["fabric"]][0] * sheet(o, room).area, 4)
            )
    objs = [
        {k: v for k, v in o.items() if k not in ("action", "nudge")} for o in objs
    ]  # the action is kept once, in its unit

    act = (
        lambda o, lab: jittered(acts[o["name"]][0], o["kind"], int(lab[6:]))
        if lab.startswith("repeat")
        else acts[o["name"]][0]
    )

    def cal_run(j):  # a run over the cap on a busy host runs once more with RETRY_CAP times the cap
        args = (
            xml,
            j[0],
            act(j[0], j[1]),
            probes(j[0]).get(j[1]),
            j[2],
            j[2] + ".mp4" if j[1] == "base" and VIDEO else None,
        )
        r = _one(*args)
        return _one(*args, cap=RETRY_CAP * CAP_S) if r.get("over_cap") else r

    # irst the base run and its repeats; the half and double runs only for objects whose readout repeats
    first = [
        (o, lab, os.path.join(work, o["name"], lab))
        for o in objs
        for lab in ("base",) + tuple(f"repeat{k}" for k in range(len(JITTER.get(o["kind"], ()))))
    ]
    with ThreadPoolExecutor(procs, initializer=use_session, initargs=(current_session(),)) as ex:
        res = list(ex.map(cal_run, first))
    pre = {(j[0]["name"], j[1]): r for j, r in zip(first, res)}

    def repeats(o):
        base = pre[(o["name"], "base")]
        t = None if readout_problems(o, base, room) else poured(o, base)
        vals = [
            None
            if readout_problems(o, pre[(o["name"], f"repeat{k}")], room)
            else poured(o, pre[(o["name"], f"repeat{k}")])
            for k in range(len(JITTER.get(o["kind"], ())))
        ]
        return t is not None and all(
            v is not None and v * t > 0 and abs(np.log(v / t)) <= np.log(REPEAT_MAX) for v in vals
        )

    second = [(o, lab, os.path.join(work, o["name"], lab)) for o in objs if repeats(o) for lab in probes(o)]
    with ThreadPoolExecutor(procs, initializer=use_session, initargs=(current_session(),)) as ex:
        res2 = list(ex.map(cal_run, second))
    jobs, res = first + second, res + res2
    by = {(j[0]["name"], j[1]): r for j, r in zip(jobs, res)}
    units = []
    for o in objs:
        base = by[(o["name"], "base")]
        bad = readout_problems(o, base, room)
        if bad:
            raise SystemExit(f"reference room fails its own {o['name']} base run: {bad}")
        truth = poured(o, base)
        errs = {k: readout_problems(o, by[(o["name"], k)], room) for k in probes(o) if (o["name"], k) in by}
        sens = {
            k: None if errs.get(k, True) else poured(o, by[(o["name"], k)], SCALE.get(k, 1.0)) for k in probes(o)
        }  # not run when the readout does not repeat
        reps = [by[(o["name"], f"repeat{k}")] for k in range(len(JITTER.get(o["kind"], ())))]
        repeat = [None if readout_problems(o, r, room) else poured(o, r) for r in reps]
        noise = max([abs(v - truth) for v in repeat if v is not None], default=FLOOR[o["kind"]])
        steady = all(v is not None and v * truth > 0 and abs(np.log(v / truth)) <= np.log(REPEAT_MAX) for v in repeat)
        resolves = any(v is not None and not within(v, truth, o["kind"], noise) for v in sens.values())
        if (
            o["kind"] == "thin_shell"
        ):  # the other class must fall outside, and its own class inside (it passes the material stage)
            resolves = (
                sens["other"] is not None
                and sens["same"] is not None
                and not within(sens["other"], truth, o["kind"], noise)
                and within(sens["same"], truth, o["kind"], noise)
            )
        decidable = steady and resolves
        why_not = (
            f"its readout does not repeat: {', '.join('failed' if v is None else f'{v:.4g}' for v in repeat)} against {truth:.4g} with the action moved a little"
            if not steady
            else f"{SWAP[o['material']['fabric']][1]} is within a factor of 2 of the truth, or {SWAP[o['material']['fabric']][0]} is not"
            if o["kind"] == "thin_shell"
            else f"half and double the {SENSE[o['kind']].replace('_', ' ')} both stay within a factor of 2 of the truth"
        )
        units.append(
            dict(
                id=f"S1.{o['name']}",
                object=o["name"],
                kind=o["kind"],
                action=acts[o["name"]][0],
                truth=truth,
                rule="factor2",
                tolerance=round(float(noise), 9),
                repeat=repeat,
                sensitivity=sens,
                decidable=decidable,
                **(
                    {
                        "stiffness_band": stiffness_band(dict(truth=truth, sensitivity=sens), noise)
                        if steady
                        else [BAND, BAND]
                    }
                    if o["kind"] == "soft_body"
                    else {}
                ),
                **({} if decidable else {"not_decidable": why_not}),
                **(dict(readout="poured_of_true_amount", true_ml=volume_ml(o, room)) if o["kind"] == "fluid" else {}),
                seconds=base.get("seconds"),
                truth_object=o,
                **({"sensitivity_errors": {k: v for k, v in errs.items() if v}} if any(errs.values()) else {}),
            )
        )
        if o["kind"] == "fluid":
            why = contained(base)
            if why:
                raise SystemExit(f"reference room's {o['name']} does not stay in its holder at rest: {why}")
            units.append(
                dict(
                    id=f"contain.{o['name']}",
                    object=o["name"],
                    kind="contain",
                    truth=HELD,
                    tolerance=round(1 - HELD, 4),
                    rest_mps=REST_MPS,
                    decidable=True,
                    truth_object=o,
                )
            )
            units.append(
                dict(
                    id=f"S0.{o['name']}",
                    object=o["name"],
                    kind="amount",
                    truth=volume_ml(o, room),
                    tolerance=0.3,
                    decidable=True,
                    truth_object=o,
                )
            )
        if o["kind"] == "soft_body":
            units.append(
                dict(
                    id=f"S0.{o['name']}",
                    object=o["name"],
                    kind="size",
                    truth=dims_cm(o, room),
                    tolerance=0.4,
                    decidable=True,
                    truth_object=o,
                )
            )
    if tree(room) != input_files:
        raise SystemExit("non-rigid calibration inputs changed during the run; completed artifacts retained")
    with open(out, "w") as f:
        json.dump(
            dict(
                schema="nonrigid_case/v1",
                room=os.path.relpath(room, os.path.dirname(os.path.abspath(out))),
                threads=THREADS,
                inputs=reference_inputs(room, [u["truth_object"] for u in units]),
                seed=seed,
                units=units,
            ),
            f,
            indent=1,
        )  # the room relative to the case file
    return units


def word(name):
    """An object's category word: its name without the __<n> suffix (the vocabulary's naming rule)."""
    return re.sub(r"__\d+$", "", str(name))


def pair(truth, copies, tfolder=".", cfolder=".", same_kind=True):
    """{true name: the copy object that stands for it}, each copy object used once: the one with the reference object's
    name, else the nearest with its category word and a box diagonal within SAME_SIZE of the true one, else the nearest
    of its kind within PLACE_SIZE, at any distance: the rigid correspond()'s size limits on its word and place steps,
    without their distances. Placement is the geometry verifiers' job."""
    used, out = set(), {}

    def take(t, pool):
        pool = [c for c in pool if c["name"] not in used]
        if not pool:
            return
        c0 = center(t, tfolder)

        def dist(c):
            try:
                return float(np.linalg.norm(center(c, cfolder) - c0))
            except Exception:
                return float("inf")  # a shape that cannot be measured still pairs (then fails as unbuildable)

        d, n = min((dist(c), c["name"]) for c in pool)
        used.add(n)
        out[t["name"]] = dict(next(c for c in copies if c["name"] == n), _distance=round(d, 3))

    def diag(o, folder):
        try:
            return float(np.linalg.norm(np.subtract(*bounds(o, folder)[::-1])))
        except Exception:
            return None  # a shape that cannot be measured still pairs (then fails as unbuildable)

    def alike(c, t, k):
        a, b = diag(c, cfolder), diag(t, tfolder)
        return a is None or b is None or 1 / k <= a / max(b, 1e-9) <= k

    ok = (lambda c, t: c["kind"] == t["kind"]) if same_kind else (lambda c, t: True)
    for rule in (
        lambda c, t: c["name"] == t["name"],
        lambda c, t: word(c["name"]) == word(t["name"]) and alike(c, t, SAME_SIZE),
        lambda c, t: c["kind"] == t["kind"] and alike(c, t, PLACE_SIZE),
    ):
        for t in truth:
            if t["name"] not in out:
                take(t, [c for c in copies if ok(c, t) and rule(c, t)])
    return out


def finite(v):
    """v is a real number that converts to a finite float (a huge integer or a bool is not)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    try:
        return np.isfinite(float(v))
    except (OverflowError, ValueError):
        return False


SHAPES = {
    "fluid": {"cylinder": ("pos", "radius", "height")},
    "soft_body": {"box": ("pos", "size"), "mesh": ("file", "pos")},
    "thin_shell": {
        "draped_sheet": ("rail", "rail_radius", "width", "front_length", "back_length"),
        "mesh": ("file", "rail"),
    },
}  # a cloth mesh's pos is optional (schema.mesh_of: 0)


def scene_supports(scene_xml):
    """What a side-file object may sit on or in, by the Genesis builder's own rule (world.run()): "floor", or a
    top-level body of the scene with collision geometry (room._aabbs). Empty when the scene does not load."""
    from .room import _aabbs

    try:
        return {"floor"} | (
            set(_aabbs(scene_xml)) - {"world"}
        )  # "world" is loose worldbody geometry, not a body (the actor run drops it)
    except Exception:
        return set()


def unbuildable(c, supports, folder="."):
    """Why the builder could not make this side-file object, or None; anything the checks raise is a reason, never a
    crash."""
    try:
        return _unbuildable(c, supports, folder)
    except Exception as e:
        return f"its declaration cannot be read ({type(e).__name__})"


def _unbuildable(c, supports, folder="."):
    """Why the builder could not make this side-file object, or None: a shape type its kind does not run with, a missing
    or non-positive field that type needs, a shape whose bounds cannot be computed, or a holder or support that is
    neither the floor nor a collidable top-level body."""
    s = c.get("shape") if isinstance(c.get("shape"), dict) else {}
    if not isinstance(s.get("type"), str) or not isinstance(c.get("on"), (str, type(None))):
        return "its shape type or support is not a name"
    need = SHAPES.get(c["kind"], {}).get(s.get("type"))
    if need is None:
        return f"a {c['kind']} cannot be a {s.get('type')}"
    for k in need:
        v = s.get(k)
        if v is None:
            return f"its {s.get('type')} has no {k}"
        if k in ("pos", "rail") and not (isinstance(v, list) and len(v) == 3 and all(finite(x) for x in v)):
            return f"its {k} is not three finite numbers"
        if k == "file" and not isinstance(v, str):
            return "its file is not a file name"
        if k in ("radius", "height", "rail_radius", "width", "front_length", "back_length") and not (
            finite(v) and v > 0
        ):
            return f"its {k} is not a positive finite number"
        if k == "size" and not (isinstance(v, list) and len(v) == 3 and all(finite(x) and x > 0 for x in v)):
            return "its size is not three positive finite numbers"
    if "yaw_deg" in s and not finite(s["yaw_deg"]):
        return "its yaw_deg is not a finite number"
    if "pos" in s and not (isinstance(s["pos"], list) and len(s["pos"]) == 3 and all(finite(x) for x in s["pos"])):
        return "its pos is not three finite numbers"  # optional where not needed, checked where given
    try:
        lo, hi = bounds(c, folder)
    except Exception as e:
        return f"its shape cannot be built ({type(e).__name__})"
    if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        return "its shape's bounds are not finite"
    m = c.get("material") if isinstance(c.get("material"), dict) else {}
    num = lambda v, lo, hi: finite(v) and lo < float(v) < hi
    if c["kind"] == "soft_body" and not (
        num(m.get("youngs_kpa"), 0, np.inf) and num(m.get("density"), 0, np.inf) and num(m.get("poisson", 0.3), -1, 0.5)
    ):
        # what MPM.Elastic takes
        return "its material is not Young's modulus and density, positive and finite, with a Poisson ratio under 0.5"
    if c["kind"] == "fluid" and (m.get("viscosity") not in VISCOSITY or not num(m.get("density", 1000.0), 0, np.inf)):
        return "its material is not a known viscosity class with a positive, finite density"
    if c["kind"] == "thin_shell" and m.get("fabric") not in FABRIC:
        return "its fabric is not one of the presets"
    if c.get("on") not in supports:
        return f"its {'holder' if c['kind'] == 'fluid' else 'support'} {c.get('on')} is neither the floor nor a collidable body of the scene"
    if c["kind"] == "fluid" and c.get("on") == "floor":
        return "a fluid needs a holder body, not the floor"  # the spill tips and measures its holder
    if (
        c["kind"] == "thin_shell"
    ):  # the builder's own cloth mesh must have area (a sheet narrower than its 2.5 cm spacing has no faces)
        if s["type"] == "draped_sheet":
            from .drape import draped_sheet

            sheet = draped_sheet(s["rail"], s["rail_radius"], s["width"], s["front_length"], s["back_length"])
        else:
            from .schema import mesh_of

            sheet = mesh_of(c, folder)
        if len(sheet.faces) == 0 or not float(sheet.area) > 0:
            return "its cloth mesh has no area"
    if "attachments" in c and not isinstance(c["attachments"], dict):
        return "its attachments are not an object"  # world.run() reads .get() on them
    if c.get(
        "attachments"
    ):  # ring clips: the builder binds them to a collidable capsule or cylinder rod on the support
        from .attachments import problems as clip_problems, rod

        why = clip_problems(c)
        if why:
            return "; ".join(why)
        try:
            origin, axis, half = rod(os.path.join(folder, "scene.xml"), c)
        except ValueError as e:
            return str(e)
        declared = np.asarray(c["attachments"]["axis"], float)
        declared /= np.linalg.norm(declared)
        if abs(float(axis @ declared)) < 0.99:
            return "clip axis is not parallel to the rigid rod"  # RingTrack's own rule
        from .attachments import header  # the builder's own header rule, on the shape's own vertices

        if s["type"] == "draped_sheet":
            from .drape import draped_sheet

            pts = draped_sheet(s["rail"], s["rail_radius"], s["width"], s["front_length"], s["back_length"]).vertices
        else:
            from .schema import mesh_of

            pts = mesh_of(c, folder).vertices
        try:
            header(pts, origin, axis if float(axis @ declared) >= 0 else -axis, half, c["attachments"]["count"])
        except ValueError as e:
            return str(e)
    return None


def object_units(case, every, one, bodies, rows, copies, dropped, room, copy):
    """One unit per true non-rigid object, scored as a stage share like a robot task: kind (built as its kind,
    buildable), material (its class: see material_check), then behaviour (its fluid stays in and pours like the true
    one; its soft body presses like it), behaviour only where calibration found a decidable or containment check. A
    stage counts only after the ones before it pass, so one mistake costs once: a towel built rigid loses the unit, not
    its kind, material and behaviour separately. The old per-check rows stay in the report, not counted. Plus a holder
    row (geometry) per fluid: on a body of the true holder's word; and a pose row (geometry) per soft body and thin
    shell: pose_unit()."""
    by = {r["id"]: r for r in rows}
    hidden = {u["object"]: u for u in case["units"] if u["kind"] in ("fluid", "soft_body")}
    out = []
    for t in every:
        c = one.get(t["name"])
        why = None
        if c is None:
            why = unpaired_why(t, copies, dropped, room, copy)
        elif c["kind"] != t["kind"]:
            why = f"{c['name']} is a {c['kind']}, the true {t['name']} a {t['kind']}"
        else:
            why = unbuildable(c, bodies, copy)
        stages = [("kind", why is None, why or f"{c['name']}: {c['kind']}")]
        if why is None:
            u = hidden.get(t["name"])
            ok, mwhy = material_check(t, c, frozen_band(case, t["name"]))
            stages.append(("material", ok, mwhy))
        else:
            stages.append(("material", False, "not built as a " + t["kind"]))
        checks = [
            by[i]
            for i in (
                f"contain.{t['name']}",
                f"S1.{t['name']}",
            )  # the behaviour checks calibration kept (never a retired kind)
            if i in by and not by[i].get("retired") and (by[i]["kind"] == "contain" or by[i].get("decidable"))
        ]
        if checks:
            stages.append(
                (
                    "behaviour",
                    stages[-1][1] and all(r.get("passed") for r in checks),
                    "; ".join(f"{r['id']}: {r.get('why', '')}" for r in checks),
                )
            )
        broken = [b for r in checks for b in (r.get("safety") or [])]
        ran = any(r.get("ran") for r in checks)  # a simulation actually ran (not skipped, not oversized)
        done = 0
        for _, ok, _ in stages:
            if not ok:
                break
            done += 1
        if broken:
            done = 0  # a broken run rule (it blew up, fell through, ran over its cap twice) scores the object 0
        out.append(
            dict(
                id=f"N.{t['name']}",
                object=t["name"],
                kind="nonrigid",
                copy_object=c and c["name"],
                counted=True,
                points=len(stages),
                stages=[dict(stage=n, passed=bool(ok), why=w) for n, ok, w in stages],
                evidence="static and simulated" if ran else "static",
                passed=done == len(stages),
                score=done / len(stages),
                why=("broken run rule: " + "; ".join(broken))
                if broken
                else next((w for n, ok, w in stages if not ok), "all stages pass"),
            )
        )
        if t["kind"] != "fluid":
            out.append(pose_unit(t, c, room, copy))  # a fluid's place is its holder's (H, and the rigid G2)
        if t["kind"] == "fluid":
            h = (
                c is not None
                and c["kind"] == "fluid"
                and not unbuildable(c, bodies, copy)
                and word(c.get("on")) == word(t.get("on"))
            )
            out.append(
                dict(
                    id=f"H.{t['name']}",
                    object=t["name"],
                    kind="holder",
                    copy_object=c and c["name"],
                    counted=True,
                    passed=h,
                    score=1.0 if h else 0.0,
                    why=f"in {c.get('on')}, the true one in {t.get('on')}" if c else "not built",
                )
            )
    for r in rows:  # folded into the object units above: reported, not counted
        if r["kind"] in ("contain", "fluid", "soft_body", "thin_shell") and r.get("counted"):
            r.update(counted=False, folded_into=f"N.{r['object']}")
        if r["kind"] == "size" and r.get("counted"):  # a soft body's size now counts in its pose unit
            r.update(counted=False, folded_into=f"P0.{r['object']}")
    return out


# The rigid pose-and-size and pairing numbers, kept here so that the non-rigid code (whose hash keys the Genesis runs)
# does not take in the rigid modules (tests/test_nonrigid.py checks they agree): the G2 pose bar (sbv/calibrate.py) and
# the side it leaves out; pair_gate.py POSE_FACTOR and POSE_CAP; g2_shape.py SIZE_BARS and RAMP; case.py SAME_SIZE,
# PLACE_SIZE.
POSE_BAR, THIN, POSE_FACTOR, POSE_CAP, SIZE_BARS, RAMP, SAME_SIZE, PLACE_SIZE = (
    0.10,
    0.005,
    0.25,
    0.5,
    (1.25, 2.0),
    2.0,
    2.0,
    3.0,
)


def ramp(x, bar, end):
    """g2_shape.ramp(): 1 up to bar, 0 from end on, a half cosine between."""
    return 1.0 if x <= bar else 0.0 if x >= end else 0.5 * (1 + np.cos(np.pi * (x - bar) / (end - bar)))


def pose_unit(t, c, room, copy):
    """P0, the rigid G2 pose-and-size rule on a soft body or thin shell at rest as declared, in geometry: its box centre
    within max(POSE_BAR, POSE_FACTOR x the true box diagonal, at most POSE_CAP), every side (own frame, longest first; a
    thin shell's two longest) within SIZE_BARS, each with the smooth credit of g2_shape.ramp(); the object earns their
    product. Paired as N is, of any kind (placement is geometry's question, whatever the copy is built as); no
    simulation."""
    row = dict(
        id=f"P0.{t['name']}",
        object=t["name"],
        kind="pose",
        copy_object=c and c["name"],
        counted=True,
        passed=False,
        score=0.0,
    )
    if c is None:
        return dict(row, why="not declared")
    try:
        lo, hi = (np.asarray(v, float) for v in bounds(t, room))
        tdims = np.array(dims_cm(t, room)) / 100
        err = float(np.linalg.norm(center(c, copy) - (lo + hi) / 2))
        cdims = np.array(dims_cm(c, copy)) / 100
    except Exception as e:
        return dict(row, why=f"its shape cannot be measured ({type(e).__name__})")
    if not (np.isfinite(err) and np.isfinite(cdims).all()):
        return dict(row, why="its shape is not finite")  # a mesh with a NaN vertex: never a NaN score
    tol = max(POSE_BAR, min(POSE_CAP, POSE_FACTOR * float(np.linalg.norm(hi - lo))))
    sides = (
        2 if t["kind"] == "thin_shell" else 3
    )  # a sheet: its third side is how its folds were drawn (curtains: 9 vs 4 cm)
    ratio = max(
        [max(a / b, b / a) if a > 0 else np.inf for a, b in zip(cdims[:sides], tdims[:sides]) if b >= THIN] or [1.0]
    )
    score = ramp(err, tol, RAMP * tol) * ramp(float(np.log(ratio)), np.log(SIZE_BARS[0]), np.log(SIZE_BARS[-1]))
    return dict(
        row,
        passed=bool(err <= tol and ratio <= SIZE_BARS[0]),
        score=round(float(score), 6),
        position_error=round(err, 3),
        position_tol=round(tol, 3),
        size_ratio=round(float(ratio), 3),
        why=f"{err:.2f} m off (bar {tol:.2f}), sides "
        + " x ".join(f"{a:.0f}" for a in cdims * 100)
        + " vs "
        + " x ".join(f"{b:.0f}" for b in tdims * 100)
        + " cm",
    )


RELATIVE_CAP = 3.0  # a copy's run may take RELATIVE_CAP times the true run, never under CAP_S: a copy at the
# reference object's size and amount runs about as long; 600 s alone cut real copies of a 311 s true fluid
RETRY_CAP = 3.0  # a run over its cap runs once more with RETRY_CAP times the cap: a busy host, not the copy, slows it
VOLUME_MAX = 4.0  # a run's cost follows its particles: a copy object over VOLUME_MAX times the true volume is not run


def copy_cap(u):
    """The compute cap for a copy's run of unit u, in seconds."""
    return max(CAP_S, RELATIVE_CAP * float(u.get("seconds") or 0))


def oversized(u, c, copy, room):
    """Why a copy object is too big to run (its volume over VOLUME_MAX times the true one's), or None. Every run then
    finishes: a copy's cost is bounded by its particles, and the true run's cost is known (an oversized object fails its
    behaviour, as it fails its amount or size unit)."""
    try:
        if u["kind"] == "fluid":
            v, t = volume_ml(c, copy), volume_ml(u["truth_object"], room)
        else:  # a sheet by its area (its two longest sides): its third side is how its folds were drawn, as in P0
            k = 2 if u["kind"] == "thin_shell" else 3
            v, t = float(np.prod(dims_cm(c, copy)[:k])), float(np.prod(dims_cm(u["truth_object"], room)[:k]))
    except Exception:
        return None
    return (
        f"{c['name']} is {v / t:.1f} times the true volume: over {VOLUME_MAX:g}, not run"
        if t > 0 and v / t > VOLUME_MAX
        else None
    )


def unpaired_why(t, copies, dropped, tfolder=".", cfolder="."):
    """Why a reference object has no copy object: no side file, or none of its kind declared (before, it read "missing"
    or "not within 1.0 m")."""
    if dropped is None:
        return "no nonrigid.json in the copy"
    bad = [str(n) for n, k in dropped if k == t["kind"]]  # a dropped entry's name may not be text
    if any(c.get("kind") == t["kind"] for c in copies):  # declared, but each of another topic or size, or taken
        return f"no {t['kind']} of its word or size left in the copy's nonrigid.json"
    return f"no {t['kind']} in the copy's nonrigid.json" + (
        f" ({', '.join(bad)} dropped for problems in the file)" if bad else ""
    )


# The kind and material checks: no simulation, one of each per true non-rigid object of every kind. kind: the copy
# builds the object as the same kind (a towel built rigid, or declared a soft body, fails). material: the declared
# material is in the true one's class: a cloth's fabric preset; a fluid's viscosity class (its amount is the amount
# unit); a soft body's Young's modulus and density within a band. The bands follow the bench's factor-2 bar on
# behaviour: the stiffness band is the factor that moves the reference object's own hidden-action readout by a factor of
# 2 (from its calibration's half and double runs: 2 ** (1 / slope) on log-log), never under BAND and at most
# MATERIAL_MAX; without a usable sensitivity it is BAND. Density, which sets the mass, takes BAND as every mass guessed
# from pictures does.
MATERIAL_MAX = 4.0


def frozen_band(case, name):
    """The stiffness band of a soft body as its case froze it, else from its unit's numbers, else [BAND, BAND]."""
    u = next((u for u in case["units"] if u["object"] == name and u["kind"] == "soft_body"), None)
    if (u or {}).get("stiffness_band"):
        return u["stiffness_band"]
    if "does not repeat" in (u or {}).get("not_decidable", ""):
        return [BAND, BAND]  # a case frozen before the band was: unsteady readouts
    return stiffness_band(u)


def stiffness_band(u, noise=None):
    """[softer, stiffer]: how many times softer or stiffer than the reference object a copy's Young's modulus may be,
    each side the factor that moves the true hidden-action readout by a factor of 2 (2 ** (1 / slope) on log-log, from
    the half run for the softer side and the double run for the stiffer one), at least BAND and at most MATERIAL_MAX. A
    side whose run moves the readout no more than its noise has no slope to speak of: MATERIAL_MAX, the same limit a
    vanishing slope reaches, so a tiny change in the numbers cannot flip the band. A side without a usable run, or with
    the readout moving the wrong way, takes BAND. Frozen in the case at calibration (stiffness_band)."""
    s = (u or {}).get("sensitivity") or {}
    t = (u or {}).get("truth")
    n = (u or {}).get("tolerance", 0.0) if noise is None else noise
    if not isinstance(t, (int, float)) or t <= 0:
        return [BAND, BAND]

    def side(v, softer):
        if not isinstance(v, (int, float)) or v <= 0:
            return BAND
        if abs(v - t) <= n:
            return MATERIAL_MAX
        if (v > t) != softer:
            return BAND  # a softer object presses deeper; anything else is not this model
        slope = abs(np.log(v / t)) / np.log(2.0)
        return float(min(MATERIAL_MAX, max(BAND, 2.0 ** min(10.0, 1.0 / slope))))

    return [side(s.get("half"), True), side(s.get("double"), False)]


def material_check(t, c, band=(BAND, BAND)):
    """(passed, why) for a copy object c of the reference object t's kind."""
    mt, mc = t.get("material", {}), c.get("material", {})
    if t["kind"] == "thin_shell":
        k = [FABRIC_CLASS.get(m.get("fabric")) for m in (mc, mt)]  # its class, light or heavy
        return k[0] is not None and k[0] == k[1], f"fabric {mc.get('fabric')} ({k[0]}) vs {mt.get('fabric')} ({k[1]})"
    if t["kind"] == "fluid":
        return mc.get("viscosity") == mt.get("viscosity"), f"viscosity {mc.get('viscosity')} vs {mt.get('viscosity')}"
    lo, hi = (band, band) if isinstance(band, (int, float)) else band
    e, d = mc["youngs_kpa"] / mt["youngs_kpa"], mc["density"] / mt["density"]
    ek, dk = 1 / lo <= e <= hi, 1 / BAND <= d <= BAND
    return ek and dk, (
        f"Young's modulus {mc['youngs_kpa']:g} vs {mt['youngs_kpa']:g} kPa ({'within' if ek else 'outside'} 1/{lo:.2f} to x{hi:.2f}), "
        f"density {mc['density']:.0f} vs {mt['density']:.0f} kg/m3 ({'within' if dk else 'outside'} x{BAND:g})"
    )


@operation
def evaluate(case_path, copy, out, procs=5, qualification=False):
    case = upgrade(json.load(open(case_path)))
    for u in case["units"]:  # a case frozen before this check may hold an infinite tolerance
        if (
            not np.isfinite(u["tolerance"])
            or u["tolerance"] < 0
            or not np.isfinite(np.asarray(u["truth"], dtype=float)).all()
        ):
            raise ValueError(f"{u['id']}: invalid frozen truth or tolerance; recalibration required")
    room = os.path.join(os.path.dirname(os.path.abspath(case_path)), case["room"])  # an absolute path stays as it is
    if (
        case.get("inputs") is None
        or reference_inputs(room, [u["truth_object"] for u in case["units"]]) != case["inputs"]
    ):
        raise ValueError(
            "non-rigid case reference inputs are missing or changed; bind retained calibration hashes or recalibrate affected units"
        )
    side = side_file(copy)
    copies, problems = load(side) if os.path.exists(side) else ([], ["no nonrigid.json"])
    bad = lambda c: [p for p in problems if p.startswith(repr(c.get("name"))) or p.startswith(str(c.get("name")) + ":")]
    dropped = [(c.get("name"), c.get("kind")) for c in copies if bad(c)] if os.path.exists(side) else None
    copies = [c for c in copies if not bad(c)]
    truth = {u["object"]: u["truth_object"] for u in case["units"]}
    side_t = side_file(room)
    every = (
        load(side_t)[0] if os.path.exists(side_t) else list(truth.values())
    )  # every non-rigid object, the retired cloth included
    bodies = scene_supports(os.path.join(copy, "scene.xml"))
    one = pair(every, copies, room, copy, same_kind=False)  # one correspondence, used by every check
    tkind = {t["name"]: t["kind"] for t in every}
    used = {c["name"] for c in one.values()}  # by name: pair() hands back copies of the entries
    problems = problems + [
        f"{c['name']!r}: unpaired and {why}"
        for c in copies
        if c["name"] not in used
        for why in [unbuildable(c, bodies, copy)]
        if why
    ]  # reported, no credit either way
    paired = {n: c for n, c in one.items() if c["kind"] == tkind[n] and not unbuildable(c, bodies, copy)}
    static_ok = {
        n
        for n, c in paired.items()  # material passes: only these need a behaviour run
        if material_check(next(t for t in every if t["name"] == n), c, frozen_band(case, n))[0]
    }
    xml = os.path.abspath(os.path.join(copy, "scene.xml"))
    work = os.path.abspath(os.path.join(os.path.dirname(out) or ".", "nonrigid_eval"))
    runs = [
        u
        for u in case["units"]
        if u["kind"] in ("fluid", "soft_body", "thin_shell")
        and not retired(u)
        and u["object"] in static_ok
        and (u["decidable"] or any(v["kind"] == "contain" and v["object"] == u["object"] for v in case["units"]))
    ]
    big = {u["id"]: oversized(u, paired[u["object"]], copy, room) for u in runs}

    def go(u):
        if big[u["id"]]:
            return dict(error=big[u["id"]], oversized=True, ran=False)
        args = (
            xml,
            paired[u["object"]],
            u["action"],
            None,
            os.path.join(work, u["object"]),
            os.path.join(work, u["object"] + ".mp4") if VIDEO else None,
            case.get("threads", THREADS),
        )
        r = _one(*args, reuse=qualification, cap=copy_cap(u))
        r = _one(*args, reuse=qualification, cap=RETRY_CAP * copy_cap(u)) if r.get("over_cap") else r
        return dict(r, ran=True)

    with ThreadPoolExecutor(procs, initializer=use_session, initargs=(current_session(),)) as ex:
        res = dict(zip([u["id"] for u in runs], ex.map(go, runs)))
    for u in case["units"]:  # not run: its kind or material already failed, so its behaviour cannot score
        if (
            u["kind"] in ("fluid", "soft_body", "thin_shell")
            and not retired(u)
            and u["object"] in paired
            and u["id"] not in res
        ):
            res[u["id"]] = dict(skipped=True, why="not run: its material is outside the true one's class")
    rows = []
    for u in case["units"]:
        c = paired.get(u["object"])
        row = dict(
            id=u["id"],
            object=u["object"],
            kind=u["kind"],
            copy_object=c and c["name"],
            distance_m=c and c["_distance"],
            decidable=u["decidable"],
            counted=bool(u["decidable"]) and not retired(u),
            truth=u["truth"],
        )  # a unit that cannot decide does not count
        if retired(u):  # a case from before the swing was retired
            row.update(counted=False, passed=False, score=None, retired=True, why="not counted: " + RETIRED[u["kind"]])
        elif not u["decidable"]:
            row.update(
                passed=False,
                score=None,
                why="not counted: " + u.get("not_decidable", "calibration marked it not decidable"),
            )
        elif c is None:
            row.update(passed=False, why=unpaired_why(truth[u["object"]], copies, dropped, room, copy))
        elif res.get(f"S1.{u['object']}", {}).get("skipped") and u["kind"] in (
            "contain",
            "fluid",
            "soft_body",
            "thin_shell",
        ):
            row.update(passed=False, skipped=True, why=res[f"S1.{u['object']}"]["why"])
        elif u["kind"] == "contain":  # read from the fluid's own run, before its action starts
            r = res.get(f"S1.{u['object']}") or dict(error="the fluid was not run")
            bad = contained(r, u.get("rest_mps", REST_MPS))
            row.update(
                value=r.get("held_at_rest"),
                rest_speed=r.get("rest_speed"),
                passed=not bad,
                safety=safety(c, r, copy),
                ran=bool(r.get("ran")),  # its run's broken rules
                why="; ".join(bad) or f"{r['held_at_rest']:.0%} inside, at rest",
            )
        elif u["kind"] == "amount":
            v = volume_ml(c, copy)
            row.update(value=v, passed=amount_ok(v, u["truth"]), why=f"{v:.0f} vs {u['truth']:.0f} ml")
        elif (
            u["kind"] == "size"
        ):  # outer size, longest to shortest side, each within the tolerance (a pillow is not a box)
            v = dims_cm(c, copy)
            row.update(
                value=v,
                passed=all(abs(a - b) <= u["tolerance"] * b for a, b in zip(v, u["truth"])),
                why=" x ".join(f"{a:.0f}" for a in v) + " vs " + " x ".join(f"{b:.0f}" for b in u["truth"]) + " cm",
            )
        elif u.get("rule") == "factor2":  # a plain threshold
            r = res[u["id"]]
            bad = safety(c, r, copy)  # an oversized object, or a run over its cap twice, fails here
            val = (
                None
                if bad or not isinstance(r.get("readout"), (int, float))
                # a spill: poured as a share of the true amount
                else (
                    r["readout"] * volume_ml(c, copy) / u["true_ml"]
                    if u.get("readout") == "poured_of_true_amount"
                    else r["readout"]
                )
            )
            ok = val is not None and within(val, u["truth"], u["kind"], u["tolerance"] if "repeat" in u else None)
            row.update(
                value=val,
                seconds=r.get("seconds"),
                safety=bad,
                score=1.0 if ok else 0.0,
                passed=ok,
                ran=bool(r.get("ran")),
                why="; ".join(bad)
                or (
                    r.get("error", "no readout")
                    if val is None
                    else f"{val:.3f} vs {u['truth']:.3f}: {'within' if ok else 'outside'} a factor of 2 (or {u['tolerance'] if 'repeat' in u else FLOOR[u['kind']]:.3g})"
                ),
            )
        else:
            r = res[u["id"]]
            bad = safety(c, r, copy)
            row.update(
                value=r.get("readout"),
                seconds=r.get("seconds"),
                safety=bad,
                ran=bool(r.get("ran")),
                passed=not bad and abs(r["readout"] - u["truth"]) <= u["tolerance"],
                why="; ".join(bad) or f"{r['readout']:.3f} vs {u['truth']:.3f} (tolerance {u['tolerance']})",
            )
        if row["counted"] and row.get("score") is None:
            row["score"] = 1.0 if row["passed"] else 0.0
        rows.append(row)
    rows = object_units(case, every, one, bodies, rows, copies, dropped, room, copy) + rows
    counted = [r["score"] for r in rows if r["counted"]]
    rep = dict(
        schema="nonrigid_report/v1",
        side_file_problems=problems,
        units=rows,
        nonrigid=round(100 * float(np.mean(counted)), 1) if counted else 0.0,
        not_counted=[r["id"] for r in rows if not r["counted"]],
    )
    with open(out, "w") as f:
        json.dump(rep, f, indent=1)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "evaluate"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument(
        "--seed", type=int, help="calibrate: draws the hidden actions (default: the side file's, else a fresh one)"
    )
    a = ap.parse_args()
    r = (
        calibrate(a.paths[0], a.out, a.procs, a.seed)
        if a.cmd == "calibrate"
        else evaluate(a.paths[0], a.paths[1], a.out, a.procs)
    )
    print(
        json.dumps(
            r if a.cmd == "evaluate" else [{k: u[k] for k in ("id", "truth", "tolerance", "decidable")} for u in r],
            indent=1,
        )
    )
