"""Optional private reuse of completed calibration trials; scoring/oracles always execute.

SBV_PACKAGE_CACHE stores the actual result and complete original run artifacts. Keys include the runtime input
closure, room-local controller data, entry code, native code, engine identity and trial parameters.
Staging paths are not identities. Errors, timeouts and non-finite runs are never stored.
Controllers receive a physical scene plus their declared input files.
"""

import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from sbv.common.dependencies import runtime, environment, PHYSICS_ENV, SCOPES, sha, operation, published, reuse_enabled


def identity(entry, engine, ctx, timeout, threads, python, code_room=None, purpose="calibration"):
    from .input_bundle import identity as inputs
    from sbv.common.code_dependencies import native

    room = Path(ctx["scene_dir"]).resolve()
    code_room = Path(code_room or room).resolve()
    files = inputs(room, entry, ctx.get("files", ()), code_room)
    return dict(
        schema="package-calibration-reuse/v4",
        inputs=files,
        native=native("package", [code_room / p for p in files["controller"]]),
        engine=engine,
        launcher_sha256=sha(python) if Path(python).is_file() else None,
        runtime=runtime(python, packages=SCOPES["nonrigid" if engine == "genesis" else "package"]),
        environment=environment(PHYSICS_ENV),
        objects=ctx["objects"],
        params=ctx["params"],
        seed=ctx["seed"],
        threads=threads,
        timeout_s=timeout,
        phase=purpose,
    )


def artifacts(folder):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(Path(folder).rglob("*")) if p.is_file()}


@operation
def completed(entry, engine, ctx, work, timeout, threads, python, execute, code_room=None, purpose="calibration"):
    root = os.environ.get("SBV_PACKAGE_CACHE")
    if not root or root == "off" or not reuse_enabled():
        return execute()
    ident = json.loads(json.dumps(identity(entry, engine, ctx, timeout, threads, python, code_room, purpose)))
    key = hashlib.sha256(json.dumps(ident, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    dest = Path(root).resolve() / key
    manifest = dest / "provenance.json"
    job = dest / "run"
    try:
        saved = json.loads(manifest.read_text())
        if saved["identity"] == ident and artifacts(job) == saved["artifacts"]:
            from .runner import checked

            result = json.loads((job / "result.json").read_text())
            checked(result)
            if result["finite"]:
                shutil.rmtree(work, ignore_errors=True)
                shutil.copytree(job, work)
                Path(work, "calibration-reuse.json").write_text(
                    json.dumps(
                        dict(
                            cache_key=key,
                            original=saved["origin"],
                            original_completed_at=saved["completed_at"],
                            current_context=ctx,
                            scope="Reused measured calibration result and artifacts; not a new execution or oracle.",
                        ),
                        indent=1,
                    )
                )
                return result
    except (OSError, ValueError, KeyError, TypeError):
        pass
    result = execute()
    if "error" in result or not result.get("finite", False):
        return result
    if identity(entry, engine, ctx, timeout, threads, python, code_room, purpose) != ident:
        return dict(error="task calibration inputs changed during execution; original artifacts retained")
    from .runner import checked

    checked(result)
    Path(root).mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=key + ".", dir=root))
    try:
        shutil.copytree(work, temp / "run")
        (temp / "provenance.json").write_text(
            json.dumps(
                dict(
                    identity=ident,
                    origin=dict(entry=str(entry), context=ctx, work=work),
                    completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    artifacts=artifacts(temp / "run"),
                ),
                indent=1,
            )
        )
        if dest.exists():
            shutil.rmtree(dest)
        temp.rename(dest)
        published(
            manifest, check=lambda: identity(entry, engine, ctx, timeout, threads, python, code_room, purpose) == ident
        )
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return result
