"""The layer-1 gate: the engine's speed modes (the settle cache and frozen robot trials)
against the plain engine, for one room. A mode with any mismatch is switched off for that room, with the reason.

- Robot: every named task's nominal run and jitter seed 1 (and edit 0 with edits=True), plain and frozen, compared by
  gate.same (verdicts and judged quantities).
- Copies: the kept whole-room settle replayed against the same settle stepped, full MuJoCo state, bit for bit (the
  property that keeps a copy's scored, nominal run unchanged).
- Nudges (nudge_check): calibrate's nudge draw both ways. Both move the objects in the reference room (calibrate._nudged),
  so the frozen settle is the only difference, and any draw decided differently is a mismatch.
Writes layer1.json: records, comparisons and the decision."""

import json, os, time
import numpy as np

from sbv.speed import run as R, gate as G

EDIT0 = dict(mass=1.3, friction=0.15)


def _env(mode, cache):
    for k, v in R.mode_env(mode, cache).items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _xml(room):
    from sbv.common import case as K

    return K.scene_xml(os.path.abspath(room))


def robot(xml, case, units, specs, mode, cache, procs, extra=None, log=print):
    """Each (task, trial) is its own job, so the trials of one task run in parallel; the records are merged per task in
    the order of specs. extra(unit id): more params for the frozen mode."""
    jobs = [("trials", R.as_mode(u, mode, extra(u["id"]) if extra else None), [sp]) for u in units for sp in specs]
    res, wall = R.pool(xml, case, jobs, procs, mode, cache, log=log)
    merged = {}
    for r in res:
        if r.get("error"):
            merged.setdefault("__errors__", dict(job="trials", id=r.get("id"), error=r["error"]))
            continue
        m = merged.setdefault(r["id"], dict(r, trials=[]))
        m["trials"] = m["trials"] + r["trials"]
    out = [v for k, v in merged.items() if k != "__errors__"] + (
        [merged["__errors__"]] if "__errors__" in merged else []
    )
    return G._by_id(out), wall, res


def decision(rep):
    """On or off per mode: any robot mismatch or error, or a nudge mismatch, turns frozen trials off; a copy whose
    replayed settle is not bit for bit turns the settle cache off."""
    r = rep["robot"]
    bad = list(r["diffs"].items())[:3]
    d = dict(
        robot_freeze=dict(
            on=not r["diffs"] and not r["errors"],
            mismatch_units=r["mismatch_units"],
            why=("; ".join(f"{k}: {v[:3]}" for k, v in bad) or "; ".join(r["errors"][:3]))
            or f"{len(r['tasks'])} tasks x {r['trials']} match the plain engine",
        )
    )
    nd = (rep.get("nudge") or {}).get("diffs") or {}
    if nd:  # a nudge decided differently: frozen trials change a verdict input
        d["robot_freeze"] = dict(
            on=False,
            why=f"nudge check differs: {list(nd.items())[:3]}",
            mismatch_units=sorted({k.split(" ")[0] for k in nd}),
        )
    elif rep.get("nudge"):
        n = len(rep["nudge"]["plain"])
        d["robot_freeze"]["why"] += (
            f"; nudge draws match ({n})" if n else "; no nudge draw ran, so the nudges are not checked"
        )
    cp = rep.get("copies") or []
    badc = [c for c in cp if not c.get("bit_for_bit")]
    d["settle_cache"] = dict(
        on=(not badc) if cp else None,
        why=(
            f"{len(badc)} of {len(cp)} copies: replayed settle differs or failed: {badc[:2]}"
            if badc
            else f"{len(cp)} copies: kept settle replays bit for bit"
            if cp
            else "no copy checked; tests/test_settle_cache.py"
        ),
    )
    return d


def layer1(room, case_path, out, procs=4, copies=(), edits=False, log=print, reuse_plain=None):
    """Plain engine against layer 1 on the reference room and on copies. reuse_plain: a layer1.json whose plain records
    to reuse (valid while the plain code path is unchanged). Returns and writes layer1.json."""
    os.makedirs(out, exist_ok=True)
    case = json.load(open(case_path))
    xml = _xml(room)
    cache = os.path.join(out, "cache")
    _env("l1", cache)
    units, bad = R.named_units(xml, case)
    specs = [(0, None), (1, None)] + ([(0, EDIT0)] if edits else [])
    log(f"layer1: {len(units)} robot tasks, trials {specs}, plain then layer 1")
    if reuse_plain:
        old = json.load(open(reuse_plain))
        plain, wp = old["records"]["plain"], old["robot"]["plain_wall_s"]
        log(f"layer1: plain records reused from {reuse_plain}")
    else:
        plain, wp, _ = robot(xml, case, units, specs, "plain", cache, procs, log=log)
    l1, wl, _ = robot(xml, case, units, specs, "l1", cache, procs, log=log)
    diffs = G.robot_diffs(
        {k: v for k, v in plain.items() if k != "__errors__"}, {k: v for k, v in l1.items() if k != "__errors__"}
    )
    errs = plain.get("__errors__", []) + l1.get("__errors__", [])
    rep = dict(
        room=os.path.abspath(room),
        case=os.path.abspath(case_path),
        plain_reused_from=reuse_plain,
        refused_at_build=bad,
        robot=dict(
            tasks=[u["id"] for u in units],
            trials=[f"seed:{s}" if not e else "edit:0" for s, e in specs],
            plain_wall_s=wp,
            l1_wall_s=wl,
            diffs=diffs,
            errors=errs,
            mismatch_units=sorted({k.split(" ")[0] for k in diffs}),
            seconds={
                i: dict(plain=[t["seconds"] for t in plain.get(i, [])], l1=[t["seconds"] for t in l1.get(i, [])])
                for i in plain
                if i != "__errors__"
            },
        ),
        records=dict(plain=plain, l1=l1),
    )
    rep["copies"] = [copy_replay(c, case, log) for c in copies]
    rep["decision"] = decision(rep)
    with open(os.path.join(out, "layer1.json"), "w") as f:
        json.dump(R.jsonable(rep), f, indent=1)
    log(f"layer1: robot mismatches {len(diffs)}, errors {len(errs)}; decision {json.dumps(rep['decision'])}")
    return rep


def nudge_diffs(a, b):
    """Plain against frozen nudge draws: every draw decided differently, either way, or not run on one side. Both nudges
    move the objects in the reference room (calibrate._nudged); the "lenient" exception this had while the plain nudge
    rewrote the room through MjSpec.to_xml is gone."""
    diffs = {}
    for x, y in zip(a, b):
        px, py = x.get("passed", x.get("error")), y.get("passed", y.get("error"))
        if px != py or not isinstance(px, bool):
            diffs[f"{x['id']} {x.get('sigma')}:{x.get('seed')}"] = f"plain {px} vs frozen {py}"
    return diffs


def nudge_check(room, case_path, out, procs=4, draws=((0.05, 1),), log=print, ids=None):
    """calibrate's nudge check, plain against frozen, for each task whose plain nominal run in
    layer1.json passes (ids: only these), for the given (sigma, seed) draws. Adds 'nudge' to layer1.json."""
    case = json.load(open(case_path))
    p = os.path.join(out, "layer1.json")
    rep = json.load(open(p)) if os.path.exists(p) else {}
    xml = _xml(room)
    cache = os.path.join(out, "cache")
    _env("l1", cache)
    units, _ = R.named_units(xml, case)
    plain = (rep.get("records") or {}).get("plain") or {}
    needed = {}
    for u in units:
        t0 = next((t for t in plain.get(u["id"], []) if t["trial"] == "seed:0"), None)
        if t0 and t0.get("ok") and (not ids or u["id"] in ids):
            needed[u["id"]] = [n for n in (t0.get("passed_by") or []) if n not in u["slots"]]
    jobs = lambda mode: [
        ("nudge", R.as_mode(u, mode), needed[u["id"]], sg, sd) for u in units if u["id"] in needed for sg, sd in draws
    ]
    a, wa = R.pool(xml, case, jobs("plain"), procs, "plain", cache, log=log)
    b, wb = R.pool(xml, case, jobs("l1"), procs, "l1", cache, log=log)
    diffs = nudge_diffs(a, b)
    rep["nudge"] = dict(plain=a, l1=b, plain_wall_s=wa, l1_wall_s=wb, diffs=diffs)
    if "robot" in rep:
        rep["decision"] = decision(rep)
    with open(p, "w") as f:
        json.dump(R.jsonable(rep), f, indent=1)
    log(f"nudge check: {len(diffs)} differences {diffs}")
    return rep


def recheck(out):
    """Recompute layer1.json's comparisons and decision from the records on disk (after a rule change)."""
    p = os.path.join(out, "layer1.json")
    rep = json.load(open(p))
    pl, l1 = rep["records"]["plain"], rep["records"]["l1"]
    rep["robot"]["diffs"] = G.robot_diffs(
        {k: v for k, v in pl.items() if k != "__errors__"}, {k: v for k, v in l1.items() if k != "__errors__"}
    )
    rep["robot"]["mismatch_units"] = sorted({k.split(" ")[0] for k in rep["robot"]["diffs"]})
    if rep.get("nudge"):
        rep["nudge"].pop("lenient", None)
        rep["nudge"]["diffs"] = nudge_diffs(rep["nudge"]["plain"], rep["nudge"]["l1"])
    rep["decision"] = decision(rep)
    with open(p, "w") as f:
        json.dump(R.jsonable(rep), f, indent=1)
    return rep


def copy_replay(scene_dir, case, log=print):
    """On a copy: a World's kept whole-room settle (replayed) against the same settle stepped, full MuJoCo state, bit
    for bit (plan.World.kept). Run in this process with the settle cache 'mem'."""
    from sbv.functionality.f4_robot import plan as RP

    t0 = time.time()
    xml = _xml(scene_dir)
    start = tuple(case["geometry"]["navigation"]["start"]) + (0.0,)
    saved = os.environ.get(RP.CACHE_ENV)
    os.environ[RP.CACHE_ENV] = "off"
    try:
        a = RP.World(xml, start, False, case["physics"])
        a.steps()
        sa = a.d.qpos.copy(), a.d.qvel.copy(), a.d.qacc.copy()
        os.environ[RP.CACHE_ENV] = "mem"
        RP._KEPT.clear()
        b = RP.World(xml, start, False, case["physics"])
        b.kept()
        c = RP.World(xml, start, False, case["physics"])
        c.kept()
        same = all(np.array_equal(x, y) for x, y in zip(sa, (c.d.qpos, c.d.qvel, c.d.qacc)))
    except Exception as e:
        return dict(copy=scene_dir, error=f"{type(e).__name__}: {e}")
    finally:
        RP._KEPT.clear()
        if saved is None:
            os.environ.pop(RP.CACHE_ENV, None)
        else:
            os.environ[RP.CACHE_ENV] = saved
    log(
        f"  copy {os.path.basename(scene_dir.rstrip('/'))}: replayed settle bit for bit: {same} ({time.time() - t0:.0f} s)"
    )
    return dict(copy=scene_dir, bit_for_bit=bool(same), seconds=round(time.time() - t0, 1))
