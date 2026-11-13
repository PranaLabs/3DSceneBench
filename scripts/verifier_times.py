#!/usr/bin/env python
"""How long each verifier takes, from scoring reports (python -m sbv.evaluate ... --out REPORT.json; every report
records its own timings). Prints one row per verifier number or scoring part: the median and slowest seconds of one run
(one repeat of a unit, or one part of scoring a copy) and how many runs were measured. docs/reference/VERIFIERS.md
quotes this table.
Usage: python scripts/verifier_times.py REPORT.json [REPORT.json ...]
"""

import json, sys
from collections import defaultdict
import numpy as np


def times(paths):
    t = defaultdict(list)
    for p in paths:
        with open(p) as f:
            r = json.load(f)
        num = r.get("verifiers", {})
        for k, v in (r.get("seconds_by_part") or {}).items():
            t[f"part: {k}"].append(v)
        for k, v in ((r.get("geometry") or {}).get("seconds") or {}).items():
            t[k].append(v)
        for k, v in ((r.get("appearance") or {}).get("seconds") or {}).items():
            t[f"A: {k}"].append(v)
        for uid, reps in (r.get("units") or {}).items():  # init units are timed together, as part F2.1
            t[num.get(uid, "?")] += [x["seconds"] for x in reps if "seconds" in x]
    return {k: v for k, v in t.items() if v}


if __name__ == "__main__":
    print("| Verifier or part | Median s | Slowest s | Runs |\n|---|---|---|---|")
    for k, v in sorted(times(sys.argv[1:]).items()):
        print(f"| {k} | {np.median(v):.2f} | {np.max(v):.2f} | {len(v)} |")
