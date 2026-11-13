#!/usr/bin/env python
"""Read saved report.json, self.json or check.json without importing the grader.

Usage: python scripts/report_diagnostics.py OUT/check.json OUT/self.json REPORT.json
Warnings do not change grading or qualification. Exit 0 means the supplied JSON
objects were read; exit 2 means an input could not be read as a JSON object.
"""

import argparse
import json
import math
from pathlib import Path

SIMILARITIES = ("siglip2", "dinov3")


def missing_similarities(mean):
    mean = mean if isinstance(mean, dict) else {}
    return [k for k in SIMILARITIES if type(mean.get(k)) not in (int, float) or not math.isfinite(mean[k])]


def diagnostics(report):
    lines = [f"Recorded status: {report.get('status', 'not recorded')}"]
    if "ready" in report:
        lines.append(f"Recorded ready: {json.dumps(report['ready'])}")
    if report.get("failure"):
        lines.append(f"Recorded failure: {report['failure']}")
    if report.get("incomplete"):
        lines.append(
            f"Run incomplete: {json.dumps(report['incomplete'])}. Inspect worker logs before rerunning grading."
        )
    if "self" in report or "ready" in report:
        if not isinstance(report.get("self"), dict):
            lines.append(
                "Qualification self-score details are not recorded in a usable form; cannot determine appearance completeness. Inspect self.json and qualification logs."
            )
            return lines
        missing = missing_similarities(report["self"])
        if missing:
            lines.append(
                "Qualification appearance incomplete: missing or non-finite "
                + ", ".join(missing)
                + ". Inspect self.json for the renderer/metric status before treating qualification as full appearance evidence."
            )
        else:
            lines.append(
                "Both scored appearance similarities are recorded in qualification; this alone does not validate the rest of the run."
            )
        return lines

    units = report.get("units")
    for uid, repeats in (units if isinstance(units, dict) else {}).items():
        if not isinstance(repeats, list):
            continue  # qualification unit counts are not repeat records
        for index, row in enumerate(repeats, 1):
            if isinstance(row, dict) and row.get("grader_error"):
                lines.append(f"Grader error in {uid}, repeat {index}: {row.get('error', 'error detail not recorded')}")

    appearance = report.get("appearance")
    if not isinstance(appearance, dict) or not appearance:
        gate = report.get("F1")
        if isinstance(gate, dict) and gate.get("passed") is False:
            lines.append(
                "Appearance is not recorded; a failed F1 gate can skip appearance. Missing optional diagnostics alone do not change scoring."
            )
        else:
            lines.append(
                "Appearance is not recorded. This report alone cannot show whether appearance was requested. "
                "If a full appearance grade was expected, inspect the grading command and logs, then rerun with appearance enabled. "
                "Missing optional diagnostics alone do not change scoring."
            )
    elif appearance.get("copy_render_failed") is True:
        lines.append(
            "copy_render_failed is recorded; existing reward handling assigns zero appearance. "
            "This flag does not distinguish submission-asset problems from runtime/process failure. "
            "Inspect copy-render logs before attributing the failure; no cause is established by the flag alone."
        )
        if appearance.get("status"):
            lines.append(f"Recorded appearance detail: {appearance['status']}")
    elif appearance.get("status"):
        status = appearance["status"]
        source = "reference render" if str(status).startswith("reference ") else "renderer/metric"
        lines.append(
            f"Appearance incomplete ({source} status): {status}. "
            "Inspect renderer/metric runtime logs and rerun in the configured grading environment. "
            "A standalone numeric result does not establish a complete appearance grade; Harbor reward conversion handles this status separately."
        )
    else:
        missing = missing_similarities(appearance.get("mean") or {})
        if missing:
            lines.append(
                "Appearance incomplete: missing or non-finite "
                + ", ".join(missing)
                + ". Inspect saved metrics and renderer logs before comparing this result with a full grade."
            )
        else:
            lines.append(
                "Both scored appearance similarities are recorded; this alone does not validate the rest of the run."
            )
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args(argv)
    print("Read-only diagnostic: saved scores, verdicts, receipts and grader exit codes are unchanged.")
    print("Diagnostic exit 0 means readable reports, not a grading pass.")
    failed = False
    for path in args.reports:
        print(f"\n{path}:")
        try:
            report = json.loads(path.read_text())
            if not isinstance(report, dict):
                raise TypeError("expected a JSON object")
        except (OSError, ValueError, TypeError) as error:
            print(f"Cannot inspect report: {error}. No grading conclusion is available.")
            failed = True
            continue
        for line in diagnostics(report):
            print(line)
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
