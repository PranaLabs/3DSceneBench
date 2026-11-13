"""Workers always end (sbv/common/workers.py): a worker killed at each of the two moments where Python 3.11's process
pool waited forever, forced to happen every time, REPEATS times. An alarm fails a pool still running after 60 s (before
the fix the process then still did not exit: the stuck thread is not a daemon)."""

import multiprocessing as mp, os, shutil, signal, struct, sys, tempfile, time, unittest
from types import SimpleNamespace
from unittest.mock import patch
from sbv.common import workers as W
from tests import rooms

REPEATS = 5


def _init():
    pass


def _job(i):
    return i


def _half_sent(_):
    """Die as the kernel would kill a worker half-way through sending a 1 MB result: the length and the first 100 bytes
    written, the result queue's write lock still held."""
    q = sys._getframe(1).f_locals["result_queue"]  # of concurrent.futures.process._process_worker
    q._wlock.acquire()
    os.write(q._writer.fileno(), struct.pack("!i", 1 << 20) + bytes(100))
    os.kill(os.getpid(), signal.SIGKILL)


class _Late(dict):
    """The pool's list of workers, forcing the order that left a worker running forever: the second worker is started
    (spawn) before the first is killed, and listed only after the broken pool stopped the workers it knew."""

    pool = None

    def __setitem__(self, pid, p):
        if self:
            first = next(iter(self.values()))
            os.kill(first.pid, signal.SIGKILL)
            first.join()
            while not self.pool._broken:
                time.sleep(0.01)
            time.sleep(0.2)  # by now the manager has stopped the workers it knew
        super().__setitem__(pid, p)


class Deadline:
    def __init__(self, s):
        self.s = s

    def __enter__(self):
        def hung(*_):
            raise TimeoutError(f"still running {self.s} s after a worker was killed")

        self.old = signal.signal(signal.SIGALRM, hung)
        signal.alarm(self.s)

    def __exit__(self, *_):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self.old)


class KilledWorker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="sbv_workers_")
        cls.xml = rooms.write(os.path.join(cls.tmp, "ref"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def pool(self, start):
        """Workers started one by one as jobs come (spawn, as on a Mac), or all forked at the first job sharing the
        model (as on Linux), whatever this machine's default."""
        ctx = mp.get_context(start)
        with (
            patch.object(mp, "get_context", lambda *_: ctx),
            patch.object(W, "sys", SimpleNamespace(platform="linux", stderr=sys.stderr)),
            patch.dict(os.environ, {"SBV_SHARE_MODEL": "1"}),
        ):
            return W.Workers(4, _init, (), ((self.xml,), dict(probes=True)) if start == "fork" else None)

    def ends(self, pool, jobs):
        """The killed worker's job lost, every job ended, the cause named, no worker left running, the compile
        released."""
        with Deadline(60), pool:
            fs = [pool.submit(fn, i) for i, fn in enumerate(jobs)]
            self.assertIsInstance(fs[0].exception(), W.BrokenProcessPool)
            [f.exception() for f in fs]  # every job ends
        self.assertIn("SIGKILL", pool.why())
        self.assertEqual([p.pid for p in pool.workers.values() if p.exitcode is None], [])
        self.assertEqual(W.SC._SHARED, {})

    def test_a_worker_started_while_the_pool_breaks_is_stopped(self):
        for _ in range(REPEATS):
            pool = self.pool("spawn")
            late = pool._processes = pool.workers = _Late()
            late.pool = pool
            self.ends(pool, [_job] * 4)

    def test_a_worker_killed_while_sending_its_result(self):
        for start in ("spawn", "fork"):
            for _ in range(REPEATS):
                with self.subTest(start=start):
                    self.ends(self.pool(start), [_half_sent] + [_job] * 7)


if __name__ == "__main__":
    unittest.main()
