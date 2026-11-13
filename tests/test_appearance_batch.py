import importlib.util
import os
import tempfile
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

HAS_TORCH = all(importlib.util.find_spec(n) for n in ("torch", "transformers", "PIL"))
if HAS_TORCH:
    import numpy as np
    from PIL import Image
    from sbv.appearance import metrics as M


@unittest.skipUnless(HAS_TORCH, "run with the metrics environment")
class AppearanceBatch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.jobs = []
        for i in range(3):
            root = Path(self.tmp.name) / str(i)
            ref = root / "ref"
            gen = root / "gen"
            ref.mkdir(parents=True)
            gen.mkdir()
            for k in range(i + 1):
                Image.new("RGB", (16, 16), (30, 50 + k, 90)).save(ref / f"view_{k}.png")
                Image.new("RGB", (16, 16), (30 + i, 50 + k, 90)).save(gen / f"view_{k}.png")
            self.jobs.append(dict(ref=str(ref), gen=str(gen)))

    def metric(self, pairs, device):
        return [float(np.abs(np.asarray(a, float) - np.asarray(b, float)).mean()) for a, b in pairs]

    def test_same_reports_single_model_load_per_metric(self):
        with ExitStack() as stack:
            fns = [
                stack.enter_context(patch.object(M, n, side_effect=self.metric))
                for n in ("siglip", "dinov3", "lpips_vgg")
            ]
            separate = [M.main(j["ref"], j["gen"], "cpu") for j in self.jobs]
            for f in fns:
                f.reset_mock()
            together = M.score_many(self.jobs, "cpu")
            self.assertEqual(separate, together)
            for f in fns:
                self.assertEqual(f.call_count, 1)

    def test_broken_image_and_inference_failure_do_not_poison_other_jobs(self):
        bad = Path(self.jobs[1]["gen"]) / "view_0.png"
        bad.write_bytes(b"bad image")

        def sometimes(pairs, device):
            if any(b.getpixel((0, 0))[0] == 32 for _, b in pairs):
                raise ValueError("bad pair")
            return self.metric(pairs, device)

        with ExitStack() as stack:
            for n in ("siglip", "dinov3", "lpips_vgg"):
                stack.enter_context(patch.object(M, n, side_effect=sometimes))
            a, b, c = M.score_many(self.jobs, "cpu")
        self.assertEqual(a["mean"]["siglip2"], 0.0)
        self.assertIn("status", b)
        self.assertEqual(c["unavailable"]["siglip2"], "ValueError: bad pair")

    def test_empty_batch_and_no_common_views(self):
        self.assertEqual(M.score_many([], "cpu"), [])
        for p in Path(self.jobs[0]["gen"]).glob("*.png"):
            p.unlink()
        with ExitStack() as stack:
            for n in ("siglip", "dinov3", "lpips_vgg"):
                stack.enter_context(patch.object(M, n, side_effect=self.metric))
            out = M.score_many(self.jobs[:1], "cpu")[0]
        self.assertEqual(out["views"], [])
        self.assertIsNone(out["mean"]["siglip2"])

    @unittest.skipUnless(os.environ.get("SBV_TEST_REAL_METRICS") == "1", "opt-in: needs cached model weights")
    def test_real_models_cached_and_batched_match_uncached(self):
        jobs = self.jobs[:2]
        # Use normal evaluation-size images. Different pixels exercise non-identity scores.
        rng = np.random.default_rng(10)
        for job in jobs:
            for p in Path(job["ref"]).glob("*.png"):
                a = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
                Image.fromarray(a).save(p)
                Image.fromarray(np.roll(a, 5, axis=0)).save(Path(job["gen"]) / p.name)
        with patch.dict(os.environ, SBV_FEATURE_CACHE="off"):
            start = time.perf_counter()
            original = [M.main(j["ref"], j["gen"], "cpu") for j in jobs]
            uncached_seconds = time.perf_counter() - start
        with patch.dict(os.environ, SBV_FEATURE_CACHE=str(Path(self.tmp.name) / "cache")):
            start = time.perf_counter()
            cold = M.score_many(jobs, "cpu")
            cold_seconds = time.perf_counter() - start
            self.assertGreater(len(list((Path(self.tmp.name) / "cache").glob("*.npy"))), 0)
            start = time.perf_counter()
            warm = M.score_many(jobs, "cpu")
            warm_seconds = time.perf_counter() - start
        for old, a, b in zip(original, cold, warm):
            self.assertEqual(old["per_view"], a["per_view"])
            self.assertEqual(a["per_view"], b["per_view"])
            self.assertEqual(old["mean"], a["mean"])
            self.assertEqual(a["mean"], b["mean"])
            self.assertEqual(set(old["unavailable"]), set(a["unavailable"]))
        self.assertIsNotNone(original[0]["mean"]["siglip2"])
        self.assertIsNotNone(original[0]["mean"]["lpips_vgg"])
        print("Real metric coverage:", original[0]["mean"], "unavailable:", list(original[0]["unavailable"]))
        print(
            "Appearance seconds:",
            dict(uncached_single=uncached_seconds, cold_batch=cold_seconds, warm_batch=warm_seconds),
        )
