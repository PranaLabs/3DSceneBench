"""One run of one package task, in its own process: python -m sbv.package_tasks.runner ENTRY CTX.json OUT.json.
Loads ENTRY by its path, calls run(ctx) and writes the checked result (or the error) to OUT.json."""

import importlib.util
import json
import math
import sys
import time
import traceback


def lead(stages):
    """The share of stages passed before the first one that fails, 0 to 1: the robot tasks' rule (plan.py
    task_score)."""
    return next((i for i, (_, ok) in enumerate(stages) if not ok), len(stages)) / len(stages)


def checked(r):
    """The result in the contract's shape, or ValueError naming what is wrong: a malformed result is never a pass."""
    if not isinstance(r, dict):
        raise ValueError("run() must return a dict")
    st = r.get("stages")
    if (
        not isinstance(st, list)
        or not st
        or not all(
            isinstance(s, (list, tuple)) and len(s) == 2 and isinstance(s[0], str) and s[0] and type(s[1]) is bool
            for s in st
        )
    ):
        raise ValueError("run() must return stages: [[name, true or false], ...] in order, at least one")
    st = [[s[0], s[1]] for s in st]
    if len({s[0] for s in st}) != len(st):
        raise ValueError("stage names must be unique")
    success = all(ok for _, ok in st)
    if "success" in r and r["success"] is not success:
        raise ValueError("success, when given, must say whether every stage passed")
    out = dict(
        success=success,
        stages=st,
        score=lead(st),
        finite=r.get("finite", True),
        jitter_applied=r.get("jitter_applied", False),
        metrics=r.get("metrics", {}),
        why=r.get("why", ""),
    )
    if type(out["finite"]) is not bool or type(out["jitter_applied"]) is not bool:
        raise ValueError("finite and jitter_applied are true or false")
    if not isinstance(out["metrics"], dict) or not all(
        isinstance(k, str) and type(v) in (int, float) and math.isfinite(v) for k, v in out["metrics"].items()
    ):
        raise ValueError("metrics must map names to finite numbers")
    if not isinstance(out["why"], str):
        raise ValueError("why must be text")
    if not out["finite"]:
        out["score"] = 0.0  # a run that ends in a non-finite state scores 0, as a broken run rule
    return out


def main(entry, ctx_path, out_path):
    t0 = time.time()
    try:
        ctx = json.load(open(ctx_path))
        from pathlib import Path

        sys.path.insert(0, str(Path(entry).parent))
        spec = importlib.util.spec_from_file_location("package_task", entry)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not callable(getattr(mod, "run", None)):
            raise ValueError("the entry file defines no run(ctx)")
        res = dict(checked(mod.run(ctx)), seconds=round(time.time() - t0, 2))
    except Exception as e:
        res = dict(
            error=f"{type(e).__name__}: {str(e)[:500]}",
            trace=traceback.format_exc()[-2000:],
            seconds=round(time.time() - t0, 2),
        )
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    return 0 if "error" not in res else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
