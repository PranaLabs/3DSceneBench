#!/usr/bin/env python3
"""time_left: how much of the task's time budget is left. In every task container as /usr/local/bin/time_left.

The run stops after the budget's minutes of wall-clock work, or at its output-token limit, whichever comes first. This
counts the minutes from the container's start, which is a little before the agent starts, and the run's own clock also
leaves out time lost to model-API outages, so the time it shows as left is never more than the time really left.
Output tokens are counted outside the container, by the agent's harness: this tool cannot show them.
Usage: time_left [--seconds]   (--seconds: only the whole seconds left, for scripts)"""

import json, os, sys, time

BUDGET = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "budget.json"
)  # {"minutes": ..., "output_tokens": ...}


def container_start(proc="/proc"):
    """The container's start, in seconds since the epoch: the start of its first process (boot time + its start
    ticks)."""
    with open(os.path.join(proc, "stat")) as f:
        boot = next(int(l.split()[1]) for l in f if l.startswith("btime "))
    with open(os.path.join(proc, "1", "stat")) as f:
        after_name = f.read().rsplit(")", 1)[1].split()
    return boot + int(after_name[19]) / os.sysconf(
        "SC_CLK_TCK"
    )  # field 22 of /proc/<pid>/stat: start time in clock ticks


def clock(seconds):
    s = int(seconds)
    return f"{s // 3600} h {s % 3600 // 60:02d} min" if s >= 3600 else f"{s // 60} min {s % 60:02d} s"


def left(start, now, minutes):
    """(seconds used, seconds left) of a budget of minutes, counted from start."""
    used = max(0.0, now - start)
    return used, max(0.0, minutes * 60 - used)


def report(start, now, minutes, tokens=None):
    used, rest = left(start, now, minutes)
    lines = [f"Time used: {clock(used)} of {clock(minutes * 60)}.", f"Time left: at least {clock(rest)}."]
    if tokens:
        lines.append(
            f"The run also stops at {tokens:,} output tokens (your reasoning included); this tool cannot count those."
        )
    return "\n".join(lines)


def main(argv=None, budget=BUDGET, proc="/proc", now=None):
    argv = sys.argv[1:] if argv is None else argv
    with open(budget) as f:
        b = json.load(f)
    start, now = container_start(proc), time.time() if now is None else now
    if "--seconds" in argv:
        print(int(left(start, now, b["minutes"])[1]))
    else:
        print(report(start, now, b["minutes"], b.get("output_tokens")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
