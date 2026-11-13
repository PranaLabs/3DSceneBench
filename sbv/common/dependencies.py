"""Content manifests shared by verifier caches. No simulation or cache policy here."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from threading import RLock

_SESSION = ContextVar("verifier_dependency_session", default=None)
SCOPES = {
    "rigid": ("numpy", "scipy", "mujoco", "trimesh"),
    "package": ("numpy", "scipy", "mujoco", "trimesh"),
    "nonrigid": ("numpy", "scipy", "mujoco", "trimesh", "genesis-world", "torch", "taichi", "gstaichi"),
    "render": ("numpy", "mujoco", "trimesh", "bpy"),
    "appearance": ("numpy", "mujoco", "trimesh", "bpy", "torch", "transformers", "Pillow", "lpips"),
}


def use_session(state):
    _SESSION.set(state)


def current_session():
    return _SESSION.get()


def memo(key, compute):
    state = _SESSION.get()
    if state is None:
        return compute()
    with state["lock"]:
        if key not in state["memo"]:
            state["memo"][key] = compute()
        return state["memo"][key]


def defer(cleanup):
    state = _SESSION.get()
    if state is None:
        cleanup()
    else:
        state["cleanup"].append(cleanup)


def published(path, check=None, invalidate=None):
    state = _SESSION.get()
    if state is not None:
        if check is None:
            runtimes = dict(state["runtimes"])
            check = lambda: all(runtime(p, packages=s) == v for (p, s), v in runtimes.items())
        state["receipts"].append((check, invalidate or (lambda: Path(path).unlink(missing_ok=True))))


@contextmanager
def session():
    if _SESSION.get() is not None:
        yield
        return
    state = dict(memo={}, runtimes={}, receipts=[], cleanup=[], reuse=True, lock=RLock())
    token = _SESSION.set(state)
    try:
        yield
    finally:
        try:
            # Recheck each newly published claim with one shared fresh snapshot.
            # An unrelated metric/runtime change must not discard physical evidence.
            state["memo"] = {}
            bad = False
            for check, invalidate in list(state["receipts"]):
                try:
                    valid = check()
                except Exception:
                    valid = False
                if not valid:
                    invalidate()
                    bad = True
            if bad:
                raise RuntimeError("execution dependencies changed during the operation; measured artifacts retained")
        finally:
            try:
                for cleanup in state["cleanup"]:
                    cleanup()
            finally:
                _SESSION.reset(token)


def reuse_enabled():
    state = _SESSION.get()
    return state is None or state.get("reuse", True)


def operation(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with session():
            return fn(*args, **kwargs)

    return wrapped


PACKAGES = ("numpy", "scipy", "mujoco", "trimesh", "genesis-world", "bpy", "torch", "transformers", "Pillow")
PHYSICS_ENV = (
    "SBV_SOLVER_THREADS",
    "SBV_ROBOT_FREEZE",
    "SBV_ROBOT_DIR",
    "QD_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "GS_BACKEND",
    "PYTHONPATH",
)
CHECK_ENV = PHYSICS_ENV + (
    "SBV_LIGHT",
    "SBV_RENDER_DEVICE",
    "SBV_CYCLES_BACKEND",
    "SBV_METRIC_DEVICE",
    "SBV_NONRIGID_PYTHON",
    "SBV_SOFT_PYTHON",
    "SBV_TORCH_PYTHON",
    "NONRIGID_THREADS",
    "NONRIGID_VIDEO",
)  # SBV_SOFT_PYTHON.


def sha(path):
    p = Path(path)
    s = p.stat()

    def compute():
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()

    return memo(("file", s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns), compute)


def environment(names=CHECK_ENV):
    return {k: os.environ.get(k) for k in names}


def runtime(python=None, packages=None, fresh=False):
    """Query the selected interpreter, including wrapper-selected container runtimes.

    Do not memoize by executable path: an environment may be upgraded in place.
    Distribution versions are combined with the interpreter's actual binary hash.
    """
    packages = tuple(PACKAGES if packages is None else packages)
    key = ("runtime", str(python) if python is not None else None, packages, tuple(environment().items()))
    state = _SESSION.get()
    if state is not None and not fresh:
        result = memo(key, lambda: runtime(python, packages=packages, fresh=True))
        state["runtimes"][(python, packages)] = result
        return result
    if python is not None:
        code = (
            "import hashlib,importlib.metadata as m,json,platform,sys\nv={}\n"
            f"for name in {packages!r}:\n"
            " try:v[name]=m.version(name)\n except m.PackageNotFoundError:v[name]=None\n"
            "h=hashlib.sha256()\nwith open(sys.executable,'rb') as f:\n"
            " for b in iter(lambda:f.read(1048576),b''):h.update(b)\n"
            "print(json.dumps(dict(python=sys.version,executable_sha256=h.hexdigest(),"
            "system=platform.system(),machine=platform.machine(),versions=v)))"
        )
        result = json.loads(subprocess.check_output([str(python), "-c", code], text=True, timeout=90))
    else:
        versions = {}
        for name in packages:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        loaded = {
            n: getattr(sys.modules.get(n), "__version__", versions[n])
            for n in ("numpy", "scipy", "mujoco", "trimesh")
            if n in versions
        }
        result = dict(
            loaded_versions=loaded,
            python=sys.version,
            executable_sha256=sha(sys.executable),
            system=platform.system(),
            machine=platform.machine(),
            versions=versions,
        )
    return result


def tree(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return {}
    # Following directory symlinks implicitly would hide dependencies and can loop.
    if any(p.is_symlink() and p.is_dir() for p in folder.rglob("*")):
        raise ValueError("dependency directories must not be symlinks: " + str(folder))
    return {
        p.relative_to(folder).as_posix(): sha(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }


def model_files(xml):
    from .calibration_jobs import asset_dependencies

    xml = Path(xml).resolve()
    return {os.path.relpath(p, xml.parent): h for p, h in asset_dependencies(xml).items()}


def xml_tree(xml):
    """Expand includes for declarations without compiling meshes or running physics."""
    import xml.etree.ElementTree as ET

    main = Path(xml).resolve()

    def read(path, active):
        if path in active:
            raise ValueError("cyclic MJCF include: " + str(path))
        root = ET.parse(path).getroot()

        def expand(parent):
            for child in list(parent):
                if child.tag != "include":
                    expand(child)
                    continue
                candidates = (path.parent / child.get("file", ""), main.parent / child.get("file", ""))
                target = next((p.resolve() for p in candidates if p.is_file()), None)
                if target is None:
                    raise ValueError("missing MJCF include in " + str(path))
                included = read(target, active | {path})
                i = list(parent).index(child)
                parent.remove(child)
                for n in reversed(list(included)):
                    parent.insert(i, n)

        expand(root)
        return root

    return read(main, set())


def render_identity():
    from .code_dependencies import native

    return dict(
        code=native("render"),
        runtime=runtime(packages=SCOPES["render"]),
        environment=environment(("SBV_LIGHT", "SBV_RENDER_DEVICE", "SBV_CYCLES_BACKEND")),
    )
