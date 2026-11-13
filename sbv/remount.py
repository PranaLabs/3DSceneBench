"""Wall mounts for a case calibrated before wall mounts were recorded, without calibrating it again. Calibration now
mounts every fixed reference object that has no support path and touches the building or a fixed object
(g3_layout.mounts), and exempts from placement only what still fails; a case calibrated before keeps its old exempt
list, so a copy object paired to an exempt slot is never checked, and a full calibration takes over 30 minutes on a room
like pool013. This reruns only that step, on the case's own reference room with the case's placement spec, and writes
back the three fields calibration writes: placement.mounts (the case's own or --mounts first, then the ones found),
placement.exempt and reference_fails.placement. Nothing else in the case changes; its hash does.
Usage: python -m sbv.remount CASE [--out NEW_CASE] [--mounts MOUNTS_JSON]"""

import argparse, json
from sbv.common import case as K
from sbv.common.scene import Scene
from sbv.geometry import g3_layout as Lay


def remount(case, scene, mounts=None):
    """The case's placement spec as calibration would write it for its reference room (scene): the given mounts
    (default: the case's own) and then the ones found, and exempt = the reference objects that still fail placement with
    them."""
    place = {k: v for k, v in case["geometry"]["placement"].items() if k != "exempt"}
    place["mounts"] = list(place.get("mounts", []) if mounts is None else mounts)
    scene.reset(scene.q0)
    place["mounts"] = Lay.mounts(place, scene)
    rows = Lay.placement(place, case["objects"], scene, {n: n for n in scene.objects})["objects"]
    place["exempt"] = sorted(o["object"] for o in rows if not o["passed"])
    return place


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case")
    ap.add_argument("--out", help="write here (default: the case itself)")
    ap.add_argument("--mounts", help="JSON [{a: slot, b: world or slot}] instead of the case's own")
    a = ap.parse_args()
    case = K.load(a.case)
    scene = Scene(K.scene_xml(K.reference_path(case)), case["physics"], case["vocabulary"], probes=True, textures=False)
    place = remount(case, scene, json.load(open(a.mounts)) if a.mounts else None)
    with open(a.case) as f:
        raw = json.load(f)  # as written: load() rewrites words to their main word
    before = raw["geometry"]["placement"].get("exempt", [])
    raw["geometry"]["placement"].update(mounts=place["mounts"], exempt=place["exempt"])
    raw["geometry"].setdefault("reference_fails", {})["placement"] = place["exempt"]
    K.save(raw, a.out or a.case)
    print(
        f"{len(place['mounts'])} mounts; exempt {len(before)} -> {len(place['exempt'])}"
        + (f": {', '.join(place['exempt'])}" if place["exempt"] else "")
    )
