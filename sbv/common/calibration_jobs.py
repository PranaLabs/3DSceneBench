"""Observable, resumable calibration jobs. Only completed jobs are reusable.

A timeout aborts the run (and its worker pool); it never excludes a physical test.
Records are atomic, and a file lock prevents two runs sharing a checkpoint.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from concurrent.futures import wait, FIRST_COMPLETED

from sbv.common.workers import Workers

_ACTIVE = None


def encoded(value):
    return json.dumps(
        value,
        sort_keys=True,
        allow_nan=False,
        default=lambda x: os.fspath(x)
        if isinstance(x, os.PathLike)
        else x.tolist()
        if hasattr(x, "tolist")
        else list(x)
        if isinstance(x, set)
        else float(x),
    )


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temp.write_text(encoded(value))
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def note(stage, **details):
    """Worker-local current subtest, visible even during a long object job."""
    if _ACTIVE is not None:
        path, record = _ACTIVE
        record.update(stage=stage, stage_started=time.time(), **details)
        atomic(path, record)


class CalibrationIncomplete(RuntimeError):
    def __init__(self, status, job, message):
        self.status, self.job = status, job
        super().__init__(f"{status}: {job}: {message}")


def _execute(fn, arg, path, key, name, phase):
    global _ACTIVE
    started = time.time()
    record = dict(
        key=key,
        job=name,
        phase=phase,
        status="running",
        started=started,
        pid=os.getpid(),
        stage="starting",
        stage_started=started,
    )
    _ACTIVE = (path, record)
    atomic(path, record)
    try:
        result = fn(arg)
        record.update(
            status="completed", result=result, result_hash=hashlib.sha256(encoded(result).encode()).hexdigest()
        )
        return result
    except BaseException as e:
        record.update(status="error", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        record.update(seconds=time.time() - started, finished=time.time())
        atomic(path, record)
        _ACTIVE = None


class Journal:
    def __init__(self, directory, signature, log=print, timeout=3600, resume=True, heartbeat=30):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("job timeout must be finite and positive")
        self.timeout, self.resume, self.heartbeat = timeout, resume, heartbeat

        def say(message):  # progress must reach a log file or docker logs while jobs run:
            log(message)
            sys.stdout.flush()  # print to a file or pipe is held in a buffer until a pool starts

        self.log = say
        self.signature = json.loads(encoded(signature))
        self.root = Path(directory) / hashlib.sha256(encoded(signature).encode()).hexdigest()[:24]
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = None
        self.created = set()

    def __enter__(self):
        import fcntl

        self._lock = open(self.root.parent / "lock", "a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock.close()
            self._lock = None
            raise RuntimeError(f"calibration checkpoint is already in use: {self.root}")
        try:
            old = read(self.root / "manifest.json")
            if old is not None and old != self.signature:
                raise ValueError("checkpoint identity mismatch")
            atomic(self.root / "manifest.json", self.signature)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        self.log(f"calibration checkpoint: {self.root} (resume={self.resume}, timeout={self.timeout:g}s/job)")
        return self

    def __exit__(self, *args):
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def invalidate(self, created_only=False):
        """Do not reuse a mixture of inputs if files changed during execution."""
        for path in self.created if created_only else self.root.glob("*/*.json"):
            saved = read(path)
            if saved is not None:
                saved.update(status="invalidated", error="calibration inputs changed during execution")
                atomic(path, saved)

    def jobs(self, phase, fn, jobs, procs=1, initializer=None, initargs=(), scene=None, key_args=None):
        """Run at most procs jobs at once; persist completion out of order, return input order.

        The deadline includes worker setup. Queued work is bounded by the worker
        count, so hundreds of waiting jobs do not consume a deadline in the queue.
        """
        if procs < 1:
            raise ValueError("procs must be positive")
        pending, results, paths = [], {}, {}
        for name, arg in jobs:
            if name in paths:
                raise ValueError(f"duplicate calibration job {name}")
            key = hashlib.sha256(encoded([phase, name, key_args(arg) if key_args else arg]).encode()).hexdigest()
            path = self.root / phase / (key + ".json")
            paths[name] = path
            saved = read(path)
            if (
                self.resume
                and saved
                and saved.get("key") == key
                and saved.get("status") == "completed"
                and "result" in saved
                and saved.get("result_hash") == hashlib.sha256(encoded(saved["result"]).encode()).hexdigest()
            ):
                results[name] = saved["result"]
            else:
                pending.append((name, arg, path, key))
        total = len(paths)

        def summary(status):
            atomic(
                self.root / (phase + "-summary.json"),
                dict(
                    phase=phase,
                    status=status,
                    completed=len(results),
                    total=total,
                    updated=time.time(),
                    jobs={n: str(p.relative_to(self.root)) for n, p in paths.items()},
                ),
            )

        summary("running")
        self.log(f"{phase}: {len(results)}/{total} completed (reused); {len(pending)} to run")
        if not pending:
            summary("completed")
            return [results[name] for name, _ in jobs]
        active = {}
        todo = iter(pending)
        seen = {}
        failed = set()
        last = time.monotonic()
        try:
            with Workers(procs, initializer, initargs, scene) as pool:
                procs = pool._max_workers  # fewer if fewer fit in memory: a queued job's deadline never runs

                def submit():
                    item = next(todo, None)
                    if item is None:
                        return
                    name, arg, path, key = item
                    # Do not leave stale running timestamps from a prior attempt.
                    self.created.add(path)
                    atomic(path, dict(key=key, job=name, phase=phase, status="queued", queued=time.time()))
                    f = pool.submit(_execute, fn, arg, str(path), key, name, phase)
                    active[f] = (name, path, time.monotonic())

                for _ in range(min(procs, len(pending))):
                    submit()
                while active:
                    done, _ = wait(active, timeout=min(0.2, self.timeout), return_when=FIRST_COMPLETED)
                    # Harvest every completed sibling before handling a timeout or error.
                    failure = None
                    for f in done:
                        name, path, submitted = active.pop(f)
                        try:
                            results[name] = f.result()
                        except BaseException as e:
                            failed.add(name)
                            failure = failure or e
                            continue
                        saved = read(path) or {}
                        self.log(f"{phase}: {len(results)}/{total} completed: {name} ({saved.get('seconds', 0):.2f}s)")
                    if done:
                        summary("running")
                    if failure is not None:
                        raise failure
                    for f, (name, path, submitted) in active.items():
                        saved = read(path) or {}
                        stage = saved.get("stage", "worker setup")
                        if seen.get(name) != stage:
                            self.log(f"{phase}: running {name}: {stage}")
                            seen[name] = stage
                        if time.monotonic() - submitted >= self.timeout:
                            raise CalibrationIncomplete(
                                "timeout", name, f"exceeded {self.timeout:g}s including setup; stage={stage}"
                            )
                    if time.monotonic() - last >= self.heartbeat:
                        self.log(
                            f"{phase}: {len(results)}/{total} completed; "
                            + ", ".join(
                                f"{name}: {(time.monotonic() - submitted):.0f}s"
                                for name, _, submitted in active.values()
                            )
                        )
                        last = time.monotonic()
                    for _ in done:
                        submit()
        except BaseException as e:
            # Workers.__exit__ terminates and joins first, so it cannot overwrite these records.
            for name, arg, path, key in pending:
                saved = read(path) or dict(key=key, job=name, phase=phase)
                if saved.get("status") == "completed":
                    continue
                status = (
                    "timeout"
                    if isinstance(e, CalibrationIncomplete) and e.job == name
                    else "error"
                    if name in failed or saved.get("status") == "error"
                    else "interrupted"
                    if saved.get("status") in ("running", "queued")
                    else "not_run"
                )
                saved.update(status=status, error=saved.get("error", f"{type(e).__name__}: {e}"), finished=time.time())
                if "started" in saved:
                    saved["seconds"] = time.time() - saved["started"]
                atomic(path, saved)
            self.log(f"{phase}: incomplete: {type(e).__name__}: {e}")
            summary(
                e.status
                if isinstance(e, CalibrationIncomplete)
                else "interrupted"
                if isinstance(e, KeyboardInterrupt)
                else "error"
            )
            raise
        summary("completed")
        return [results[name] for name, _ in jobs]


def word_meanings(room):
    """Each category word of the room's objects (<category>__<n> bodies on worldbody) -> [the main word it is scored as,
    its topic]: all a room's check takes from the shared word list, sbv/common/categories.txt, which is not code and so
    not in the code hash. A word of the room that changes meaning checks the room again; words added to the list do not
    (the list only grows, sbv/words.py)."""
    import xml.etree.ElementTree as ET
    from sbv.common import case as K
    from sbv.common.scene import main_word, topic
    from sbv.common.dependencies import xml_tree

    try:
        bodies = xml_tree(K.scene_xml(str(room))).findall("./worldbody/body")
    except (ET.ParseError, OSError):
        return None
    words = sorted({b.get("name", "").split("__")[0].lower() for b in bodies if "__" in b.get("name", "")})
    return {w: [main_word(w), topic(w)] for w in words}


def asset_dependencies(room):
    """Hash external MJCF includes and file assets too (including symlink targets).

    Candidate bases cover the main XML, included XMLs and compiler asset dirs.
    Hashing an extra existing candidate is conservative: it only forces a rerun.
    """
    import xml.etree.ElementTree as ET
    from sbv.common import case as K

    main = Path(K.scene_xml(str(room))).resolve()
    documents, todo, refs, directories = {}, [main], [], set()
    while todo:
        path = todo.pop()
        if path in documents:
            continue
        root = ET.parse(path).getroot()
        documents[path] = root
        for compiler in root.iter("compiler"):
            for name in ("assetdir", "meshdir", "texturedir"):
                if compiler.get(name):
                    directories.add(compiler.get(name))
        for element in root.iter():
            file = element.get("file")
            if element.tag == "texture":
                refs.extend(
                    element.get(k)
                    for k in ("fileup", "filedown", "filefront", "fileback", "fileleft", "fileright")
                    if element.get(k)
                )
            if not file:
                continue
            if element.tag == "include":
                candidates = [path.parent / file, main.parent / file]
                found = [p.resolve() for p in candidates if p.is_file()]
                if not found:
                    raise ValueError(f"cannot fingerprint include {file!r} in {path}")
                todo.extend(found)
            else:
                refs.append(file)
    bases = {p.parent for p in documents}
    bases |= {base / d for base in list(bases) for d in directories}
    files = set(documents)
    for file in refs:
        found = {(base / file).resolve() for base in bases if (base / file).is_file()}
        if not found:
            raise ValueError(f"cannot fingerprint file asset {file!r} in {main}")
        files.update(found)
    from .dependencies import sha

    return {str(p): sha(p) for p in sorted(files)}


def identity(room, settings, *, auxiliary=True):
    """Content, not mtime or branch name. Runtime/hardware matters to physics."""
    import mujoco
    from sbv.common import case as K
    from sbv.functionality.f4_robot import rby1
    from sbv.common.dependencies import runtime, environment, sha, tree

    root = Path(__file__).resolve().parents[1]
    from .dependencies import memo

    sources = sorted(root.rglob("*.py"))

    def code_hash():
        h = hashlib.sha256()
        for p in sources:
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
        return h.hexdigest()

    code = memo(("full-native-source", tuple((str(p), sha(p)) for p in sources)), code_hash)
    folder = Path(room).resolve()
    folder = folder if folder.is_dir() else folder.parent
    overrides = {}
    for name in ("mounts", "functions", "tasks"):
        p = settings.get(name)
        if isinstance(p, (str, os.PathLike)):
            overrides[name] = sha(p)
    runtimes = {"native": runtime()}
    for name in (
        ("SBV_NONRIGID_PYTHON", "SBV_SOFT_PYTHON", "SBV_TORCH_PYTHON") if auxiliary else ()
    ):  # SBV_SOFT_PYTHON: the old name
        if os.environ.get(name):
            runtimes[name] = runtime(os.environ[name])
    metrics = root.parent / ".venv-metrics/bin/python"
    if auxiliary and not os.environ.get("SBV_TORCH_PYTHON") and metrics.is_file():
        runtimes["metrics"] = runtime(metrics)
    from sbv.appearance.cache_identity import identity as metric_identity
    from sbv.appearance.views import TORCH_PY

    weights = metric_identity(TORCH_PY) if auxiliary else None
    return dict(
        weights=weights,
        schema=5,
        room=str(Path(room).resolve()),
        files=tree(folder),
        assets=asset_dependencies(room),
        words=word_meanings(room),
        code=code,
        engine=mujoco.__version__,
        machine=platform.machine(),
        system=platform.system(),
        robot=K.fingerprint(rby1.ROBOT_DIR) if os.path.isdir(rby1.ROBOT_DIR) else None,
        settings=settings,
        override_files=overrides,
        runtimes=runtimes,
        environment=environment(),
    )
