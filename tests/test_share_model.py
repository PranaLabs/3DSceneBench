"""Workers share the Scene their parent compiled before forking (sbv/common/workers.py, scene.make), and the case and
the self-score are byte for byte those of workers that compile their own (SBV_SHARE_MODEL=0)."""

import json, os, shutil, sys, tempfile, unittest
from unittest.mock import patch
from sbv import calibrate, evaluate
from sbv.common import case as K
from sbv.common import scene as SC
from sbv.common.workers import Workers, fit
from tests import rooms

W = None


def _init(xml):
    global W
    W = SC.make(xml, probes=True)


def _job(_):
    return getattr(W, "inherited", False)


def timeless(x):
    """x as JSON, without its wall-clock fields."""
    drop = lambda x: (
        {k: drop(v) for k, v in x.items() if k not in ("seconds", "seconds_by_part")}
        if isinstance(x, dict)
        else [drop(v) for v in x]
        if isinstance(x, list)
        else x
    )
    return json.dumps(
        drop(json.loads(json.dumps(x, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))),
        sort_keys=True,
    )


@unittest.skipUnless(sys.platform == "linux", "the model is shared only where the workers fork")
class ShareModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="sbv_share_")
        cls.xml = rooms.write(os.path.join(cls.tmp, "ref"))
        cls.ref = os.path.dirname(cls.xml)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_workers_take_the_parent_compile_and_it_is_released(self):
        for env, inherited in (("1", True), ("0", False)):
            with (
                patch.dict(os.environ, {"SBV_SHARE_MODEL": env}),
                Workers(2, _init, (self.xml,), ((self.xml,), dict(probes=True))) as pool,
            ):
                self.assertEqual(set(pool.map(_job, range(4))), {inherited})
            self.assertEqual(SC._SHARED, {})

    def test_an_edited_file_is_compiled_again(self):
        xml = rooms.write(os.path.join(self.tmp, "edited"))
        k = SC.share((xml,), dict(probes=True))
        old = SC._SHARED[k][0]
        with open(xml, "a") as f:
            f.write("<!-- edited -->\n")
        self.assertIsNot(SC._SHARED[SC.share((xml,), dict(probes=True))][0], old)  # never the compile of the old file
        SC.unshare(k)
        SC.unshare(k)
        self.assertEqual(SC._SHARED, {})

    def test_the_shared_compile_never_gets_a_thread_pool(self):
        seen, real = [], SC.solver_threads  # a pool's threads do not exist in the forked workers (SIGSEGV)

        def spy(d):
            seen.append(os.environ.get("SBV_SOLVER_THREADS"))
            return real(d)

        with patch.dict(os.environ, {"SBV_SOLVER_THREADS": "8"}), patch.object(SC, "solver_threads", spy):
            k = SC.share((self.xml,), dict(probes=True))
            self.assertEqual(seen, ["1"])
            self.assertEqual(os.environ["SBV_SOLVER_THREADS"], "8")
            SC.unshare(k)
        self.assertEqual(SC._SHARED, {})

    def test_fit_counts_one_mjdata_per_worker(self):
        s = SC.Scene(self.xml, probes=True)
        each = s.d.nbuffer + s.m.narena
        self.assertEqual(fit(32, s, avail=each * 10), 8)  # SBV_MEM_FRACTION 0.8 of 10 MjData
        self.assertEqual(fit(4, s, avail=each * 100), 4)
        self.assertEqual(fit(4, s, avail=0), 1)

    def test_case_and_self_score_unchanged(self):
        out = {}
        for env in ("0", "1"):
            with patch.dict(os.environ, {"SBV_SHARE_MODEL": env}):
                case = calibrate.build(
                    self.ref,
                    procs=4,
                    draws=1,
                    need=1,
                    jitter=1,
                    log=lambda *a: None,
                    robot_tasks=0,
                    push_tasks=0,
                    camera_seed=7,
                )
                case["calibration"].pop("seconds")
                case["reference"] = os.path.relpath(self.xml, self.tmp)
                path = os.path.join(self.tmp, f"case{env}.json")
                K.save(case, path)  # case_hash: of everything but the clock
                out[env] = (
                    timeless(case),
                    timeless(evaluate.evaluate(K.load(path), self.ref, procs=4, log=lambda *a: None)),
                )
        self.assertEqual(out["1"][0], out["0"][0])
        self.assertEqual(out["1"][1], out["0"][1])


if __name__ == "__main__":
    unittest.main()
