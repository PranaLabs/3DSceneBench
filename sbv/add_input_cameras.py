#!/usr/bin/env python
"""Give a frozen case the input cameras of the 8- and 12-picture conditions without calibrating the room again. Input
cameras are not seeded, so they are rebuilt from the room as calibrate builds them (the settled room, the case's
bounds, 1.6 m up). The case's input_0-7 must come back unchanged, so the 1, 2, 4 and 8 picture conditions stay exactly
as they were; otherwise nothing is written and the cameras that moved are named. input_8-11 are rebuilt (aimed at what
the first eight miss, views.cover). A room that brings its own cameras (input_cameras.json and heldout_cameras.json,
sbv.appearance.heldout.SUPPLIED) takes its twelve from that file instead, checked as at calibration.
Usage: python -m sbv.add_input_cameras CASE ROOM_DIR [--out OUT]
"""

import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repository root
from sbv.common import case as K  # noqa: E402
from sbv.common.scene import Scene  # noqa: E402
from sbv.appearance import views as App  # noqa: E402


def add(case, room):
    """The case with all input cameras and the current picture conditions, or SystemExit naming the moved cameras."""
    app = case["appearance"]
    scene = Scene(K.scene_xml(room), case["physics"], probes=True)
    scene.settle()
    from sbv.appearance import heldout as H

    own = H.supplied(K.scene_xml(room))
    if own:  # the room's own cameras: twelve in its file, or regenerate them there
        cams = H.validate(scene, case["room"]["bounds"], *own, resolution=app["resolution"])["input_cameras"]
        if len(cams) != 12:
            raise SystemExit(
                f"{room}: {H.INPUTS_FILE} has {len(cams)} cameras; choose twelve there (the harness's input-views --count 12)"
            )
    else:
        cams = App.cameras(scene, case["room"]["bounds"], height=1.6, res=app["resolution"], kind="input")
    new = {c["name"]: c for c in cams}
    moved = [
        c["name"] for c in app.get("input_cameras", []) if int(c["name"].split("_")[1]) < 8 and new.get(c["name"]) != c
    ]
    if moved:
        raise SystemExit(
            f"{room}: the case's input cameras {', '.join(moved)} come back different: calibrate the room again"
        )
    app["input_cameras"], app["conditions"] = cams, App.CONDITIONS
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
    K.save(add(case, a.room), a.out or a.case)
    print(f"{a.out or a.case}: {len(case['appearance']['input_cameras'])} input cameras, case {case['case_hash']}")


if __name__ == "__main__":
    main()
