"""Position gate for pairing: without it, a copy object pairs with a reference object of its category wherever it
stands, so an inventory alone earns at-rest units by filling counts. This writes into a case, once and before any copy
is scored, a distance per reference object from its own size: gate = max(FLOOR, FACTOR x its box diagonal), by default
max(0.3 m, the diagonal). That is the reach the joints step and the place step of correspond() already use, with a floor
so that a cup 25 cm off is still the cup. sbv.common.case.correspond then pairs each slot by its word only within its
gate, for every readout (G1 to G3, parameter errors, F2 to F4) and at every instruction level. A case without gates gets
the default gate, max(0.3 m, the diagonal), when it is loaded (sbv/common/case.py gate()). --matcher geometric writes
case["pairing"]["matcher"]: correspond() then pairs by one geometric matching over all objects, the word only a
tie-break (sbv/common/case.py correspond_geometric). Also, the G2 pose bar grows with the object: position_tol = max(the
case's bar, POSE_FACTOR x the box diagonal), at most POSE_CAP.
Usage: python -m sbv.pair_gate CASE [--factor 1.0] [--floor 0.3] [--matcher geometric] [--pose-factor 0.25]"""

import argparse
import numpy as np
from sbv.common import case as K

FACTOR, FLOOR = 1.0, 0.3
POSE_FACTOR = 0.25  # G2 pose: position_tol = max(the case's bar, POSE_FACTOR x the box diagonal); 0 keeps the fixed bar
POSE_CAP = 0.5  # metres: a long counter still has to stand within half a metre


def set_gates(case, factor=FACTOR, floor=FLOOR, pose_factor=POSE_FACTOR, matcher=None):
    matcher = matcher or case.get("pairing", {}).get("matcher")  # a second run keeps the case's matcher
    for o in case["objects"]:
        o["gate"] = round(max(floor, factor * float(np.linalg.norm(o["dims"]))), 3)
    case["pairing"] = dict(gate="max(floor, factor x the box diagonal)", factor=factor, floor=floor)
    if matcher:
        case["pairing"]["matcher"] = matcher
    if pose_factor:  # a fixed 0.10 m bar over a 2 m counter and a 6 cm cup alike passed 14 of 377 right-sized objects
        diag = {o["slot"]: float(np.linalg.norm(o["dims"])) for o in case["objects"]}
        for r in case.get("geometry", {}).get("pose_size", []):
            r["position_tol"] = round(max(r["position_tol"], min(POSE_CAP, pose_factor * diag[r["slot"]])), 3)
        case["pairing"]["pose_tol"] = f"max(fixed bar, {pose_factor} x the box diagonal, at most {POSE_CAP} m)"
    return case


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case")
    ap.add_argument("--factor", type=float, default=FACTOR)
    ap.add_argument("--floor", type=float, default=FLOOR)
    ap.add_argument("--matcher", choices=["word", "geometric"])
    ap.add_argument("--pose-factor", type=float, default=POSE_FACTOR)
    a = ap.parse_args()
    c = K.load(a.case)
    K.save(
        set_gates(
            {k: v for k, v in c.items() if not k.startswith("_") and k != "case_hash"},
            a.factor,
            a.floor,
            a.pose_factor,
            a.matcher,
        ),
        a.case,
    )
    print("gates written:", a.case)
