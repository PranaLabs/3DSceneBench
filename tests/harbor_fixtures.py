"""Explicit synthetic visual evidence for exporter-only tests, never release inputs.

No metric models are loaded by these tests. Only local metric discovery is stubbed;
room dependencies, verifier version and evidence-file hashes use the real pipeline.
"""

import json
from pathlib import Path
from unittest import mock

from sbv import pipeline


def metric_environment():
    return mock.patch(
        "sbv.appearance.cache_identity.identity",
        return_value={"weights": {"exporter-test-fixture": "synthetic"}, "runtime": {"fixture": True}},
    )


def receipt(work, room="room", settings=None):
    """Record the actual fixture files plus a clearly synthetic complete self-render report."""
    rd = Path(work) / "rooms" / room
    out = Path(work) / "results" / room
    report = json.loads((out / "self.json").read_text()) if (out / "self.json").is_file() else {}
    report.update(appearance={"mean": {"siglip2": 1.0, "dinov3": 1.0}}, fixture="synthetic visual evidence")
    (out / "self.json").write_text(json.dumps(report))
    with metric_environment():
        record = dict(
            ready=True,
            status="passed",
            inputs=pipeline.identity(str(rd), settings or {}),
            version=pipeline.version(),
            self={"siglip2": 1.0, "dinov3": 1.0},
            artifacts=pipeline.result_files(out),
        )
    (out / "check.json").write_text(json.dumps(record))
    return record
