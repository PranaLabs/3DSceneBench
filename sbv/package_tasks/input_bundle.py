"""A package task's inputs: its controller folder, the scene as given, and declared data files.

The cache key (identity) leaves appearance out: the scene's physical view, without colours, textures, lights or
cameras. The run itself always gets the scene unchanged, so live grading scores exactly what an entrant submitted;
that is why a task's result must not depend on appearance or on files it does not declare."""

import json
import os
from pathlib import Path
import shutil


def files(room, extra=()):
    from sbv.common.dependencies import sha
    from .schema import inside
    from sbv.nonrigid.schema import side_file

    root = Path(room)
    names = set(extra)
    side = Path(side_file(root))
    if (
        side.is_file()
    ):  # the side file is the copy's: an entry the non-rigid verifiers drop (a bad mesh path, invalid JSON)
        names.add(side.name)  # must not stop every package task, rigid ones included
        try:
            objects = json.loads(side.read_text()).get("objects", [])
        except (UnicodeDecodeError, ValueError, AttributeError, RecursionError):
            objects = []  # RecursionError: nesting too deep for the JSON reader
        shapes = [o.get("shape") for o in (objects if isinstance(objects, list) else []) if isinstance(o, dict)]
        names.update(
            f
            for f in (s.get("file") for s in shapes if isinstance(s, dict))
            if isinstance(f, str) and f and inside(str(root), f) is not None
        )
    for name in names:
        if inside(str(root), name) is None:
            raise ValueError("task input must be a regular file inside the package: " + str(name))
    return {name: sha(root / name) for name in sorted(names)}


def identity(room, entry, extra=(), code_room=None):
    from sbv.common.scene_inputs import physical
    from sbv.common.dependencies import sha
    from .verify import code_files

    root = Path(code_room or room).resolve()
    rel = os.path.relpath(Path(entry).resolve(), root)
    return dict(
        model=physical(room),
        files=files(room, extra),
        controller={p: sha(root / p) for p in code_files(str(root), rel)},
    )


def prepare(entry, ctx, work):
    """The controller's code copied into the run's folder; the scene is passed as it is, never rewritten, in
    calibration, self-checks and live grading alike. Declared files must exist inside the package."""
    from .verify import code_files

    room = Path(ctx["scene_dir"])
    root = Path(work)
    entry = Path(entry)
    files(room, ctx.get("files", ()))
    # Entry folders are the existing code dependency boundary. Root entries are standalone.
    code_root = entry.parent
    target = root / "controller"
    target.mkdir(parents=True, exist_ok=True)
    if entry.parent == room or len(Path(ctx.get("entry", "nested/entry.py")).parts) == 1:
        names = [entry.name]
    else:
        names = [
            os.path.relpath(p, entry.parent)
            for p in entry.parent.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        ]
    for name in names:
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(code_root / name, dest)
    return str(target / entry.name), dict(ctx)
