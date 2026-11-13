"""Private, content-addressed qualification evidence. Live grading does not use this."""

import copy
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from .calibration_jobs import encoded, read, atomic
from .dependencies import tree, sha, published, reuse_enabled


def artifacts(root, names):
    root = Path(root)
    result = {}
    for name in names:
        p = root / name
        if p.is_file():
            result[name] = sha(p)
        elif p.is_dir():
            result.update({str(Path(name) / k): v for k, v in tree(p).items()})
    return result


def _copy(source, dest, names):
    for name in names:
        a = Path(source) / name
        b = Path(dest) / name
        if b.is_dir():
            shutil.rmtree(b)
        elif b.exists():
            b.unlink()
        b.parent.mkdir(parents=True, exist_ok=True)
        if a.is_dir():
            shutil.copytree(a, b, ignore=shutil.ignore_patterns("__pycache__"))
        elif a.is_file():
            shutil.copy2(a, b)


def cached(out, name, identity, compute, names=(), accept=lambda r: r is not None, root=None, resume=True):
    """Retain original results/artifacts and elapsed times; recompute only a missing phase.

    identity may return None when a dependency (e.g. metric weights) is unknown:
    execute without caching rather than manufacture a reusable pass.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    key = identity()
    cache = Path(root) if root else out / ".phase-cache"
    if key is None or not resume or not reuse_enabled():
        return compute()
    key = json.loads(encoded(key))
    digest = hashlib.sha256(encoded(key).encode()).hexdigest()
    dest = cache / name / digest
    manifest = dest / "evidence.json"
    saved = read(manifest)
    if (
        saved
        and saved.get("identity") == json.loads(encoded(key))
        and artifacts(dest / "files", names) == saved.get("files")
    ):
        if saved.get("result_hash") == hashlib.sha256(encoded(saved.get("result")).encode()).hexdigest() and accept(
            saved["result"]
        ):
            _copy(dest / "files", out, names)
            atomic(cache / (name + "-current.json"), dict(reused=True, original=str(manifest)))
            return copy.deepcopy(saved["result"])
    result = compute()
    if not accept(result) or identity() != key:
        return result
    # Results with NaN/inf are never committed as evidence.
    try:
        body = encoded(result)
    except (ValueError, TypeError):
        return result
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=digest + ".", dir=dest.parent))
    try:
        (temp / "files").mkdir()
        _copy(out, temp / "files", names)
        atomic(
            temp / "evidence.json",
            dict(
                identity=key,
                result=result,
                files=artifacts(temp / "files", names),
                result_hash=hashlib.sha256(body.encode()).hexdigest(),
                origin=str(out),
                completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            ),
        )
        if dest.exists():
            shutil.rmtree(dest)
        temp.rename(dest)
        published(manifest, check=lambda: identity() == key)
        atomic(cache / (name + "-current.json"), dict(reused=False, original=str(manifest)))
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return result
