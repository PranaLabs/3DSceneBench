"""The verifier code each cached stage depends on: every module its entry points import, followed through the whole
package, plus the code they start as separate processes and the data files they read. A cached result is reused only
while all of these files are unchanged.

The import graph is read from the source (any `import sbv...` or `from sbv... import`, at module level or inside a
function), so a module added to a stage's code path joins its dependencies without anyone listing it. What an import
does not show, or shows wrongly, is listed here, each with its reason:
- SUBPROCESS: code started by name in a new process (`python -m`, `-c`, a script path), and the stages that run it;
- DATA: files read at run time;
- BOOKKEEPING: the cache's own modules, which compute keys and store evidence but never a measurement: neither a
  dependency nor followed;
- SHALLOW: modules that are dependencies, but whose imports inside functions only serve cache keys: only their
  module-level imports are followed;
- IGNORE: an import a stage never executes, with why.
tests/test_code_dependencies.py fails when a `python -m`, `-c` or script path names a module SUBPROCESS does not list,
or a module reads a file beside the code that DATA does not list. Package controllers are arbitrary Python and may
import anything: they depend on the whole package."""

import ast
from functools import lru_cache
from pathlib import Path
from .dependencies import sha

ROOT = Path(__file__).resolve().parents[1]
BOOKKEEPING = frozenset(
    {
        "common/dependencies.py",
        "common/code_dependencies.py",
        "common/phase_cache.py",
        "common/calibration_inputs.py",
        "common/scene_inputs.py",
        "appearance/cache_identity.py",
        "package_tasks/calibration_cache.py",
        "qualification.py",
        "pipeline.py",
    }
)
SUBPROCESS = {  # file -> ((the file it starts by name, the stages that run it, or None for all), ...)
    "appearance/views.py": (
        ("appearance/render.py", ("render", "appearance")),  # -m sbv.appearance.render
        ("appearance/metrics.py", ("appearance",)),
    ),  # TORCH_PY sbv/appearance/metrics.py
    "nonrigid/verify.py": (("nonrigid/world.py", None),),  # -c "from sbv.nonrigid.world import run"
    "package_tasks/verify.py": (("package_tasks/runner.py", None),),  # -m sbv.package_tasks.runner
}
SHALLOW = frozenset(
    {
        "common/calibration_jobs.py",  # jobs and journals; its function imports build cache identities
        "common/gravity_cache.py",
    }
)  # stores rollouts; its function import computes the key
IGNORE = {
    # views.task_objects reads tasks.json to aim cameras; the cameras chosen are frozen in the case, which every render
    # and appearance key hashes, and tasks.py's own imports (the robot planner) never run for it.
    "render": {("appearance/views.py", "functionality/f4_robot/tasks.py")},
    "appearance": {("appearance/views.py", "functionality/f4_robot/tasks.py")},
    # The rigid self-score calls evaluate without renders: views only scores appearance, in its own cached phase.
    "rigid_score": {("evaluate.py", "appearance/views.py")},
}
DATA = {"common/scene.py": ("common/categories.txt",)}  # the category words and aliases
STAGES = {  # each cached computation's entry points
    "rigid": ("calibrate.py",),  # calibrate.build (qualification.calibration)
    "calibration_case": ("calibrate.py", "appearance/heldout.py", "appearance/views.py"),
    "rigid_score": ("evaluate.py",),  # the reference room's rigid self-score
    "gravity": (
        "common/scene.py",
        "functionality/f1_gate.py",
        "functionality/f2_physical.py",
        "functionality/f4_robot/plan.py",
        "common/gravity_cache.py",
    ),  # the rollouts gravity_cache keeps
    "resources": ("calibrate.py", "functionality/f1_gate.py", "common/scene.py"),  # calibrate._resources
    "nonrigid_calibration": ("nonrigid/verify.py",),
    "nonrigid": ("nonrigid/verify.py",),
    "package": ("package_tasks/verify.py",),
    "render": ("appearance/views.py", "appearance/render.py"),  # reference and input renders
    "appearance": ("appearance/views.py",),
}


def _module(name, root):
    """The file of `sbv.a.b` (a module or a package's __init__), or None."""
    parts = name.split(".")[1:]
    for path in (root.joinpath(*parts).with_suffix(".py"), root.joinpath(*parts, "__init__.py")):
        if parts and path.is_file():
            return path
    return None


def _nodes(tree, shallow):
    """Every node, or with shallow=True every node outside function bodies."""
    todo = [tree]
    while todo:
        node = todo.pop()
        yield node
        if not (shallow and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))):
            todo += list(ast.iter_child_nodes(node))


def imports(path, root, shallow=False):
    """The sbv modules one file imports, anywhere in it (shallow: at module level only)."""
    package = path.relative_to(root).with_suffix("").parts[:-1]
    found = set()
    for node in _nodes(ast.parse(path.read_bytes()), shallow):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names if a.name == "sbv" or a.name.startswith("sbv.")]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)] if node.level > 1 else package
                module = ".".join(("sbv",) + tuple(base) + ((node.module,) if node.module else ()))
            elif node.module and (node.module == "sbv" or node.module.startswith("sbv.")):
                module = node.module
            else:
                continue
            names = [module] + [module + "." + a.name for a in node.names]
        else:
            continue
        for name in names:
            parts = name.split(".")
            for k in range(1, len(parts) + 1):  # sbv.a.b also runs sbv/__init__ and sbv/a/__init__
                f = _module(".".join(parts[:k]), root)
                if f:
                    found.add(f)
    return found


@lru_cache(maxsize=None)
def _closure(stage, root):
    root = Path(root)
    ignore = IGNORE.get(stage, set())
    seen, todo = set(), [root / e for e in STAGES[stage]]
    while todo:
        p = todo.pop()
        rel = p.relative_to(root).as_posix()
        if p in seen or not p.is_file() or rel in BOOKKEEPING:
            continue
        seen.add(p)
        nxt = [root / f for f, stages in SUBPROCESS.get(rel, ()) if stages is None or stage in stages]
        nxt += imports(p, root, shallow=rel in SHALLOW)
        todo += [q for q in nxt if (rel, q.relative_to(root).as_posix()) not in ignore]
    files = {p.relative_to(root).as_posix() for p in seen}
    return frozenset(files | {d for f in files for d in DATA.get(f, ()) if (root / d).is_file()})


def files(stage, root=ROOT):
    """The files a stage depends on, relative to the sbv package."""
    return sorted(_closure(stage, Path(root).resolve()))


def native(stage, controller_files=(), root=ROOT):
    root = Path(root)
    if controller_files:
        # Package controllers are Python programs, not a restricted DSL. A controller
        # may dynamically import any verifier module; do not guess its read set.
        paths = {
            p.relative_to(root).as_posix()
            for p in root.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        }
    else:
        paths = files(stage, root)
    return {p: sha(root / p) for p in sorted(paths)}
