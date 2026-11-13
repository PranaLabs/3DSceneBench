"""python -m sbv.speed COMMAND ...

  profile ROOM CASE [--out DIR] [--procs N] [--sample-s S]          where the room's robot verification time goes
  layer1  ROOM CASE [--out DIR] [--procs N] [--copies DIR ...] [--edits] [--nudges | --nudges-only]
          [--nudge-draws 0.05:1,...] [--nudge-ids ID,...] [--reuse-plain layer1.json]
                                                                     the engine's speed modes against the plain engine
  recheck OUT                                                        re-decide a layer1.json from its records

ROOM is a scene folder (or scene.xml), CASE its calibrated case.json. Outputs go to --out (default ./speed_<room>).
"""

import argparse, os, sys


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m sbv.speed", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("profile", "layer1"):
        p = sub.add_parser(name)
        p.add_argument("room")
        p.add_argument("case")
        p.add_argument("--out")
        p.add_argument("--procs", type=int, default=4)
        if name == "profile":
            p.add_argument("--sample-s", type=float, default=0.5)
        if name == "layer1":
            p.add_argument(
                "--copies", nargs="*", default=[], help="copy scene folders (the kept settle replayed bit for bit)"
            )
            p.add_argument("--edits", action="store_true", help="also compare edit 0 (mass x1.3, friction +0.15)")
            p.add_argument(
                "--reuse-plain", help="layer1.json whose plain-engine records to reuse (the plain code path unchanged)"
            )
            p.add_argument("--nudges", action="store_true", help="also check calibrate's nudge draws plain vs frozen")
            p.add_argument(
                "--nudges-only", action="store_true", help="only the nudge check, added to an existing layer1.json"
            )
            p.add_argument(
                "--nudge-draws", default="0.05:1", help="sigma:seed,... (calibrate draws seeds 1-5 at 0.05 and 0.1)"
            )
            p.add_argument(
                "--nudge-ids", help="comma-separated task ids (default: every task whose plain nominal run passes)"
            )
    p = sub.add_parser("recheck")
    p.add_argument("out")
    a = ap.parse_args(argv)
    if a.cmd == "recheck":
        from sbv.speed.layer1 import recheck

        recheck(a.out)
        return 0
    out = a.out or f"speed_{os.path.basename(os.path.abspath(a.room).rstrip('/'))}"
    if a.cmd == "profile":
        from sbv.speed.profile import profile

        profile(a.room, a.case, out, a.procs, a.sample_s)
    elif a.cmd == "layer1":
        from sbv.speed.layer1 import layer1, nudge_check

        if not a.nudges_only:
            layer1(a.room, a.case, out, a.procs, a.copies, a.edits, reuse_plain=a.reuse_plain)
        if a.nudges or a.nudges_only:
            draws = tuple((float(x.split(":")[0]), int(x.split(":")[1])) for x in a.nudge_draws.split(","))
            nudge_check(a.room, a.case, out, a.procs, draws, ids=a.nudge_ids.split(",") if a.nudge_ids else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
