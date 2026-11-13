"""Worker processes for calibrate and evaluate. multiprocessing.Pool restarts a worker the kernel kills (out of memory,
say) and waits forever for the job it held, so a room too big for the machine hung until Harbor's timeout. Workers
breaks at the first death instead: every job not yet finished raises BrokenProcessPool, and why() names the cause. A
broken pool that reaches the end of the with block becomes WorkerDied. A watcher thread makes sure the pool also ends
when Python's own manager thread would wait forever (see _watch).

A pool told the Scene its initializer builds with scene.make compiles that Scene once, here, and its workers fork from
this process sharing the model (Linux; SBV_SHARE_MODEL=0 turns it off). fit() then sizes the pool to the memory free."""

import multiprocessing as mp, os, signal, sys, threading
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool  # noqa: F401  (raised to the callers)
from sbv.common import scene as SC


class WorkerDied(RuntimeError):
    """A worker process died, so its jobs never finished: a failure of the machine, not a verdict on the room."""


def fit(procs, scene, avail=None):
    """At most procs workers: as many as fit in SBV_MEM_FRACTION (default 0.8) of the memory available now
    (MemAvailable), each costing its own MjData (data buffer plus arena), since the model is shared."""
    if avail is None:
        try:
            with open("/proc/meminfo") as f:
                avail = next(int(l.split()[1]) for l in f if l.startswith("MemAvailable:")) * 1024
        except (OSError, StopIteration, ValueError):
            return procs
    each = scene.d.nbuffer + scene.m.narena
    n = max(1, int(avail * float(os.environ.get("SBV_MEM_FRACTION", "0.8")) // each))
    if n < procs:
        print(
            f"sbv: {n} workers of {each / 2**30:.2f} GB fit in the {avail / 2**30:.1f} GB available; using {n}, not {procs}",
            file=sys.stderr,
        )
    return min(procs, n)


def _threads(threads, initializer, initargs):
    os.environ["SBV_SOLVER_THREADS"] = threads  # this worker only; the parent keeps its own
    if initializer is not None:
        initializer(*initargs)


class Workers(ProcessPoolExecutor):
    def __init__(self, procs, initializer, initargs, scene=None):
        """scene: (args, kwargs) of the Scene the initializer builds with scene.make, to compile once and share."""
        threads = os.environ.get(
            "SBV_WORKER_SOLVER_THREADS"
        )  # solver threads per worker (sbv/common/scene.py solver_threads)
        if threads is not None:
            if int(threads) < 1:
                raise ValueError("SBV_WORKER_SOLVER_THREADS must be a positive integer")
            initializer, initargs = _threads, (str(int(threads)), initializer, initargs)
        fork = scene is not None and sys.platform == "linux" and os.environ.get("SBV_SHARE_MODEL") != "0"
        self.shared = SC.share(*scene) if fork else None
        try:
            if fork:
                procs = fit(procs, SC._SHARED[self.shared][0])
            super().__init__(
                procs, mp_context=mp.get_context("fork") if fork else None, initializer=initializer, initargs=initargs
            )
        except BaseException:  # no pool: release the compile now, not never
            if self.shared is not None:
                SC.unshare(self.shared)
            raise
        self.workers, self.results = self._processes, self._result_queue  # shutdown() drops both
        self.closing, self.done, self.watcher = threading.Lock(), threading.Event(), None

    def shutdown(self, wait=True, *, cancel_futures=False):
        """As ProcessPoolExecutor.shutdown, but the result queue is closed here, under the lock the watcher closes it
        with."""
        manager = self._executor_manager_thread
        try:
            super().shutdown(
                wait=False, cancel_futures=cancel_futures
            )  # tells the manager to stop; waits for nothing, closes nothing
            if wait:
                if manager is not None:
                    manager.join()
                with self.closing:
                    self.results.close()
                self.done.set()
                if self.watcher is not None:
                    self.watcher.join()
        finally:
            if self.shared is not None:
                SC.unshare(self.shared)
                self.shared = None

    def submit(self, fn, *args):
        try:
            f = super().submit(fn, *args)
        except Exception as e:  # a worker died while the jobs were still being handed out
            if not (isinstance(e, BrokenProcessPool) or self._broken):
                raise
            f = Future()
            f.set_exception(
                BrokenProcessPool(str(e))
            )  # OSError, ValueError: starting the next spawned worker hit a pipe the pool had just closed
        if self.watcher is None:  # after the first submit, which forks every worker when forking
            self.watcher = threading.Thread(target=self._watch, daemon=True, name="sbv-workers-watcher")
            self.watcher.start()
        return f

    def _watch(self):
        """Every 0.1 s until shutdown: once a worker has died or the pool broke, mark the pool broken, close this
        process's end of the result pipe and stop every worker still running, those listed after the death too; at
        shutdown, wait for them. Python 3.11's manager thread waits forever in two cases (both reproduced on a loaded
        Mac): a worker started (spawn) while the manager was stopping the others is never stopped, so the manager waits
        for it to exit; and a worker killed half-way through sending a result leaves the manager waiting for the rest,
        until no process holds the pipe open for writing."""
        while True:
            stop = self.done.wait(0.1)
            procs = list(self.workers.values())
            if self._broken or any(p.exitcode not in (None, 0) for p in procs):
                self._broken = self._broken or "a worker process died"  # submit refuses new jobs from now on
                with self.closing:
                    self.results._writer.close()
                for p in procs:
                    if p.exitcode is None:
                        p.terminate()
                    if stop:
                        p.join()
            if stop:
                return

    def __exit__(self, kind, *rest):
        if kind is not None:  # an error in this process: stop the workers now, as Pool did
            for p in list(self.workers.values()):
                p.terminate()
        self.shutdown(cancel_futures=True)  # waits for the workers, so their exit codes are known
        if kind is not None and issubclass(kind, BrokenProcessPool):
            raise WorkerDied(self.why()) from None
        return False

    def why(self):
        """The cause of a broken pool, from the exit codes. SIGTERM is the pool stopping the others after the death."""
        codes = sorted({p.exitcode for p in self.workers.values() if p.exitcode not in (None, -signal.SIGTERM)})
        if -signal.SIGKILL in codes:
            return "a verifier worker was killed with SIGKILL (exit code -9), most likely by the kernel because the machine ran out of memory"
        return f"a verifier worker stopped abruptly (exit code {', '.join(map(str, codes)) or 'unknown'}; negative: killed by that signal)"
