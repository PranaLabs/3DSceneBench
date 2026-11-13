"""Tolerance policies: a room's task judge may widen the response tolerance of chosen calibration units. A room-building
tool may write the judge's choice as selected-tolerances.json and passes it on as
calibrate.build(tolerances=...). This verifier reads such a file and accepts an empty policy, which changes nothing.
It does not apply per-unit tolerances yet: a policy that names anything is refused, never silently ignored, so no
case is ever built under bars the verifier did not use.
"""

import json


def load(path):
    """The policy in a JSON file: an object, with an optional "units" object of unit id -> settings."""
    with open(path) as f:
        policy = json.load(f)
    if not isinstance(policy, dict):
        raise ValueError(f"{path}: a tolerance policy is a JSON object")
    if not isinstance(policy.get("units", {}), dict):
        raise ValueError(f'{path}: "units" must map unit ids to settings')
    return policy


def empty(policy):
    """True for no policy, {} or {"units": {}}: nothing to apply."""
    return not policy or not any(policy.values())


def require_supported(policy):
    """Refuse a policy this verifier cannot apply, before any calibration work starts."""
    if not empty(policy):
        named = sorted(policy.get("units") or {}) or sorted(policy)
        raise ValueError("this verifier does not apply tolerance policies yet; the policy names: " + ", ".join(named))
