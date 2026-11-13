"""Rigid-job dependencies, separate from the complete qualification fingerprint.

Package controllers are executed by package_tasks, not calibrate.build. Full qualification
still uses calibration_jobs.identity, including every room file and all verifier code.
"""

import ast
import hashlib
import json
from pathlib import Path
import shutil

from .calibration_jobs import identity as complete_identity, encoded, read, atomic, Journal


def source_identity(root, package=False):
    """Rigid code only; normalize the calibration identity import, not execution code."""
    result = {}
    for p in sorted(Path(root).rglob("*.py")):
        rel = p.relative_to(root).as_posix()
        if (
            rel == "common/calibration_inputs.py"
            or (rel.startswith("package_tasks/") and not package)
            or rel == "package_tasks/calibration_cache.py"
        ):
            continue
        tree = ast.parse(p.read_bytes())
        if rel == "calibrate.py":
            # Both imports select checkpoint metadata; all physical build functions remain hashed.
            tree.body = [
                n
                for n in tree.body
                if not (isinstance(n, ast.ImportFrom) and n.module == "sbv.common.calibration_inputs")
            ]
            for n in tree.body:
                if isinstance(n, ast.ImportFrom) and n.module == "sbv.common.calibration_jobs":
                    n.names = [a for a in n.names if a.name != "identity"]
        result[rel] = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
    return result


def room_identity(room):
    """Ignore declared package Python code and package parameters, retain other inputs conservatively.

    Native files and asset dependencies are never excluded, even if a controller shares
    their directory. Unknown files stay dependencies. Malformed declarations fail closed.
    """
    from .calibration_jobs import asset_dependencies

    folder = Path(room).resolve()
    if folder.is_file():
        folder = folder.parent
    spec_path = folder / "tasks.json"
    spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}
    if not isinstance(spec, dict):
        raise ValueError("tasks.json must be an object")
    protected = {Path(p).resolve() for p in asset_dependencies(folder)}
    protected |= {
        folder / n
        for n in (
            "physics.json",
            "mounts.json",
            "functions.json",
            "nonrigid.json",
            "soft_liquid.json",
            "input_cameras.json",
            "heldout_cameras.json",
            "extra_cameras.json",
        )
    }
    ignored = set()
    for task in spec.get("package", []):
        entry = task.get("entry") if isinstance(task, dict) else None
        if not isinstance(entry, str):
            raise ValueError("package task needs an entry")
        path = (folder / entry).resolve()
        if not path.is_relative_to(folder) or path.suffix != ".py" or not path.is_file():
            raise ValueError("package entry must be a Python file inside the room")
        ignored.update([path] if path.parent == folder else path.parent.rglob("*.py"))
    result = {}
    for p in sorted(folder.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        if p == spec_path or (p.resolve() not in protected and (p.resolve() in ignored or p.suffix.lower() == ".md")):
            continue
        result[p.relative_to(folder).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    # Omitted tasks and an empty native-task mapping have identical rigid behavior.
    native = {k: v for k, v in spec.items() if k != "package"}
    if native:
        result["<native tasks>"] = hashlib.sha256(encoded(native).encode()).hexdigest()
    return result


def rigid_identity(room, settings):
    from .dependencies import runtime, environment, PHYSICS_ENV, SCOPES, sha
    from .scene_inputs import physical
    from .code_dependencies import native
    from .calibration_jobs import word_meanings

    folder = Path(room).resolve()
    folder = folder if folder.is_dir() else folder.parent
    declarations = {}
    for name in ("mounts.json", "functions.json"):
        if (folder / name).is_file():
            declarations[name] = sha(folder / name)
    spec = json.loads((folder / "tasks.json").read_text()) if (folder / "tasks.json").is_file() else {}
    tasks = {k: v for k, v in spec.items() if k != "package"}
    options = dict(settings)
    for name in ("tasks", "mounts", "functions"):
        if isinstance(options.get(name), (str, Path)):
            options[name] = json.loads(Path(options[name]).read_text())
    if isinstance(options.get("tasks"), dict):
        options["tasks"] = {k: v for k, v in options["tasks"].items() if k != "package"}
    from .dependencies import model_files
    from sbv.functionality.f4_robot import rby1

    robot_xml = Path(rby1.ROBOT_DIR) / "rby1a_1.2.xml"
    robot = (
        model_files(robot_xml)
        if robot_xml.is_file() and (settings.get("robot_tasks", 0) or settings.get("push_tasks", 0) or tasks)
        else None
    )
    return dict(
        schema=4,
        model=physical(room),
        declarations=declarations,
        tasks=tasks,
        robot=robot,
        words=word_meanings(room),
        code=native("rigid"),
        runtime=runtime(packages=SCOPES["rigid"]),
        environment=environment(PHYSICS_ENV),
        settings=options,
    )


def import_legacy(room, settings, checkpoint, source_snapshot):
    """Import completed v1 records only with exact old inputs and provably unchanged rigid code.

    source_snapshot is the retained original sbv Python tree, not a claimed revision string.
    No measurements or successful outcomes are synthesized; failed and incomplete records
    are not imported. Original elapsed times, results and their hashes are preserved.
    """
    snapshot = Path(source_snapshot).resolve()
    current = Path(__file__).resolve().parents[1]
    if source_identity(snapshot) != source_identity(current):
        raise ValueError("legacy rigid execution code differs")
    h = hashlib.sha256()
    for p in sorted(snapshot.rglob("*.py")):
        h.update(str(p.relative_to(snapshot)).encode())
        h.update(p.read_bytes())
    old = complete_identity(room, settings)
    old["code"] = h.hexdigest()
    origin = Path(checkpoint) / hashlib.sha256(encoded(old).encode()).hexdigest()[:24]
    if read(origin / "manifest.json") != old:
        raise ValueError("legacy inputs, settings or runtime do not match the retained checkpoint")
    copied = 0
    with Journal(checkpoint, rigid_identity(room, settings)) as dest:
        for phase in ("prepare", "objects", "events", "robot"):
            for p in sorted((origin / phase).glob("*.json")):
                record = read(p)
                if not record or record.get("status") != "completed" or "result" not in record:
                    continue
                if record.get("result_hash") != hashlib.sha256(encoded(record["result"]).encode()).hexdigest():
                    raise ValueError("corrupt legacy result: " + str(p))
                target = dest.root / phase / p.name
                if target.exists():
                    if read(target) != record:
                        raise ValueError("destination record conflicts: " + str(target))
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, target)
                copied += 1
        atomic(
            dest.root / "legacy-reuse.json",
            dict(
                origin=str(origin),
                source_snapshot=str(snapshot),
                original_identity=old,
                records_copied=copied,
                scope="Reused original completed calibration records, not new executions or oracle passes.",
            ),
        )
    return dict(records_copied=copied, origin=str(origin), destination=str(dest.root))
