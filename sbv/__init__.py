"""3DSceneBench verifiers: appearance (A), geometry (G1 to G3) and
functionality (F1 to F4), with common holding what all three share. docs/reference/VERIFIERS.md numbers every
verifier."""

import os

# The verifier version: the public release tag. It changes whenever a change can move a score, so two results are
# comparable only at the same version. Every reward.json carries it as a number (Harbor takes only numbers there);
# report.json carries it beside the hash of the code that ran.
VERSION = "1.0"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the repository root
