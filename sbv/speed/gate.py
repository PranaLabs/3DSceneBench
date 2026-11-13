"""Comparing robot trials of two engine modes (the layer-1 gate, sbv/speed/layer1.py).

Rules (same()):
- verdicts and booleans equal: trial ok, success, every predicate the task checks, crossings, unresolved sweeps, run
  rules, stage points, the reason (as a set of clauses), passed_by on the nominal run;
- judged numbers: |a - b| <= max(ABS_TOL, REL_TOL * max(|a|, |b|)); penetration within PEN_TOL;
- wall times are never compared.
"""

import math

ABS_TOL, REL_TOL, PEN_TOL = 5e-3, 0.05, 1e-3
SKIP = {"seconds", "wall_s", "sim_s", "build_s", "elapsed", "t", "time"}


def _num_same(a, b):
    if a is None or b is None:
        return a is b
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return abs(a - b) <= max(ABS_TOL, REL_TOL * max(abs(a), abs(b)))


def same(a, b, path="", out=None):
    """Differences between two judged records (dicts, lists, numbers, booleans, strings), by the rules above."""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in SKIP or k in ("stages", "samples", "trace"):
                continue
            if k not in a or k not in b:
                out.append(f"{path}{k}: only in {'new' if k in b else 'reference'}")
                continue
            same(a[k], b[k], f"{path}{k}.", out)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path[:-1]}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                same(x, y, f"{path}{i}.", out)
    elif (
        isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str) or a is None or b is None
    ):
        if a != b:
            out.append(f"{path[:-1]}: {str(a)[:80]} vs {str(b)[:80]}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        tol_ok = abs(a - b) <= PEN_TOL if path.endswith("pen.") else _num_same(float(a), float(b))
        if not tol_ok:
            out.append(f"{path[:-1]}: {a} vs {b}")
    elif a != b:
        out.append(f"{path[:-1]}: {str(a)[:80]} vs {str(b)[:80]}")
    return out


def robot_diffs(ref, new):
    """ref, new: {task id: [trial records]} (run.pool 'trials' results). Differences per task and trial label."""
    out = {}
    for i in sorted(set(ref) | set(new)):
        a, b = {t["trial"]: t for t in ref.get(i, [])}, {t["trial"]: t for t in new.get(i, [])}
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out[f"{i} {k}"] = ["trial missing on one side"]
                continue
            # the reason lists broken predicates in the order of a set of names (string hashing differs per process):
            # compared as a set of clauses
            norm = lambda w: "; ".join(sorted(w.split("; "))) if isinstance(w, str) else w
            keep = lambda t: dict(
                {
                    x: t.get(x)
                    for x in ("ok", "success", "stage_points", "pen", "crossed", "unresolved", "ok_rules", "details")
                    + (("passed_by",) if k == "seed:0" else ())
                },
                why=norm(t.get("why")),
            )
            d = same(keep(a[k]), keep(b[k]))
            if d:
                out[f"{i} {k}"] = d
    return out


def _by_id(results, kind="trials"):
    """{task id: [trial records]} from pool results; failed jobs under "__errors__"."""
    out = {}
    for r in results:
        if r is None:
            continue
        if r.get("error"):
            out.setdefault("__errors__", []).append(f"{r.get('job')} {r.get('id')}: {r['error']}")
            continue
        out[r["id"]] = r["trials"]
    return out
