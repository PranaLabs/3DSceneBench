"""Opt-in cache of a room's whole-room gravity rollouts: the F1 gate's stepping, the settled start and each init record.
Calibrating a room and then scoring it against itself, or calibrating it again after
editing only its declarations, repeats them exactly. Set SBV_GRAVITY_CACHE to a private, persistent directory; unset
or `off` skips key generation and I/O, as before.

The key hashes the compiled model itself, every array but the pictures (flat_textures changes only those), so it covers
every file the model was compiled from, wherever it lies, without serializing it; plus the engine options, the start
state, the physics settings, this package's code, the library versions and the machine type. A value is stored as JSON
only when it reads back exactly (same repr); unreadable or unwritable storage is a miss, never a failure."""

import hashlib, json, os, platform
from pathlib import Path
import mujoco, numpy as np
from sbv.common.calibration_jobs import encoded

ROOT = Path(__file__).resolve().parents[1]  # the code the rollouts run
SKIP = ("_", "tex", "ntex", "paths", "npaths", "nbuffer")  # pictures and file names: never physics
_CODE = {}


def key(scene, name, extra=None):
    from .code_dependencies import native

    code = native("gravity", root=ROOT)
    m = scene.m
    h = hashlib.sha256(
        encoded(
            [name, extra, scene.phys, code, mujoco.__version__, np.__version__, platform.machine(), platform.system()]
        ).encode()
    )
    for f in sorted(dir(m)):
        v = getattr(m, f) if not f.startswith(SKIP) else None
        if isinstance(v, np.ndarray):
            v = np.ascontiguousarray(v)
            h.update(f"{f} {v.dtype.str} {v.shape}".encode())
            h.update(v)
        elif isinstance(v, (int, float, bytes)):
            h.update(f"{f}={v!r}".encode())
    for s in (m.opt, m.stat):
        h.update(encoded({f: getattr(s, f) for f in dir(s) if not f.startswith("_")}).encode())
    return h.hexdigest()


def cached(scene, name, compute, extra=None, keep=lambda value: True):
    """compute(), or what it returned before for this model, start and code. A value is stored only if keep(value)."""
    folder = os.environ.get("SBV_GRAVITY_CACHE", "off")
    if not folder or folder == "off":
        return compute()
    path = Path(folder) / f"{name}-{key(scene, name, extra)}.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        pass
    value = compute()
    try:
        text = json.dumps(
            value, allow_nan=False, default=lambda x: x.item()
        )  # numpy scalars, if they read back the same
        if keep(value) and repr(json.loads(text)) == repr(value):
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            try:
                tmp.write_text(text)
                os.replace(tmp, path)  # atomic: a reader never sees half a file
            finally:
                tmp.unlink(missing_ok=True)
    except (OSError, ValueError, TypeError, AttributeError):
        pass  # a value JSON cannot hold exactly, or failing storage
    return value
