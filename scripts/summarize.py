#!/usr/bin/env python3
"""Print one table of a job's scores: a row per scene, and the mean over the scenes that were graded.

    python3 scripts/summarize.py jobs/<your job>

Reads each trial's verifier/reward.json. A scene with no reward (still running, or the grader failed) is listed as such and
left out of the mean. Needs only the Python standard library.
"""

import glob, json, os, sys

COLS = (
    ("reward", "reward"),
    ("appearance", "reward_appearance"),
    ("geometry", "reward_geometry"),
    ("functionality", "reward_functionality"),
)


def rows(job):
    out = []
    for trial in sorted(glob.glob(os.path.join(job, "*", ""))):
        name = os.path.basename(trial.rstrip("/")).split("__")[0]
        path = os.path.join(trial, "verifier", "reward.json")
        if not os.path.isfile(path):
            if os.path.isdir(os.path.join(trial, "verifier")) or os.path.isfile(os.path.join(trial, "result.json")):
                out.append((name, None))
            continue
        with open(path) as f:
            out.append((name, json.load(f)))
    return out


def main():
    if len(sys.argv) != 2 or not os.path.isdir(sys.argv[1]):
        sys.exit(__doc__)
    found = rows(sys.argv[1])
    if not found:
        sys.exit(f"no trials under {sys.argv[1]}")
    width = max(len(n) for n, _ in found + [("mean", None)])
    print(f"{'scene':<{width}}  " + "  ".join(f"{c:>13}" for c, _ in COLS))
    graded = [r for _, r in found if r and r.get("reward") is not None]
    for name, r in found:
        if r is None or r.get("reward") is None:
            print(
                f"{name:<{width}}  no scores: still running, or the grader failed (see this trial's verifier/ folder)"
            )
        else:
            print(f"{name:<{width}}  " + "  ".join(f"{r.get(k, float('nan')):>13.1f}" for _, k in COLS))
    if graded:
        mean = [sum(r.get(k, 0.0) for r in graded) / len(graded) for _, k in COLS]
        print(
            f"{'mean':<{width}}  "
            + "  ".join(f"{m:>13.1f}" for m in mean)
            + f"   ({len(graded)} of {len(found)} graded)"
        )


if __name__ == "__main__":
    main()
