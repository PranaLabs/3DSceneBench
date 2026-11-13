#!/usr/bin/env python
"""Give a frozen case its extra pictures without calibrating the room again.
The input and held-out cameras stay exactly as they are: the extras are chosen apart from them
(sbv/appearance/extra.py), from the room as calibrate builds it (the settled room, the case's bounds), or taken from the
room's own extra_cameras.json, checked as at calibration. A visible scope the case already has gains the objects the
extras show.
Usage: python -m sbv.add_extra_cameras CASE ROOM_DIR [--out OUT]
"""

import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repository root
from sbv.common import case as K  # noqa: E402
from sbv.common.scene import Scene  # noqa: E402
from sbv.appearance import extra as Extra  # noqa: E402


def add(case, room):
    """The case with its extra cameras; ValueError when the room cannot have them."""
    scene = Scene(K.scene_xml(room), case["physics"], probes=True)
    scene.settle()
    case["appearance"] = Extra.add(case["appearance"], scene, case["room"]["bounds"], K.scene_xml(room))
    if (case.get("scope") or {}).get("visible") is not None:
        case["scope"]["visible"] = Extra.visible(case, case["scope"]["visible"])
    return case


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case")
    ap.add_argument("room")
    ap.add_argument("--out", help="write here instead of over CASE")
    a = ap.parse_args(argv)
    with open(a.case) as f:
        case = json.load(f)  # as written: K.load would rewrite category words
    if case.get("schema") != K.SCHEMA:
        raise SystemExit(f"{a.case}: expected schema {K.SCHEMA}, found {case.get('schema')}")
    try:
        add(case, a.room)
    except ValueError as e:
        raise SystemExit(f"{a.room}: {e}")
    K.save(case, a.out or a.case)
    x = case["appearance"]["extra_protocol"]
    print(
        f"{a.out or a.case}: {len(case['appearance']['extra_cameras'])} extra pictures "
        f"({len(x['adds'])} objects added, {len(x['unseen'])} unseen), case {case['case_hash']}"
    )


if __name__ == "__main__":
    main()
