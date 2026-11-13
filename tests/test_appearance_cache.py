import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

HAS_TORCH = all(importlib.util.find_spec(n) for n in ("torch", "transformers", "PIL"))
if HAS_TORCH:
    import torch
    from PIL import Image
    from sbv.appearance import metrics as M


@unittest.skipUnless(HAS_TORCH, "run with the metrics environment")
class AppearanceCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = patch.dict(os.environ, SBV_FEATURE_CACHE=self.tmp.name)
        env.start()
        self.addCleanup(env.stop)
        self.model = SimpleNamespace(
            config=SimpleNamespace(_commit_hash="revision1", to_dict=lambda: {"model_type": "test"})
        )
        self.image = Image.new("RGB", (16, 16), (120, 50, 20))
        self.encode = Mock(return_value=torch.tensor([[0.3, 0.5, 0.7]], dtype=torch.float32))

    def cache(self, device="cpu", prep="p1"):
        return M.ReferenceFeatures(self.model, device, prep)

    def test_exact_embeddings_and_repeated_views(self):
        a = self.cache().get(self.image, self.encode)
        b = self.cache().get(self.image.copy(), self.encode)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(self.encode.call_count, 1)
        self.assertEqual(M.cosine(a, a * 0.5), M.cosine(b, a * 0.5))

    def test_resolved_revision_without_config_private_field(self):
        del self.model.config._commit_hash
        revision = "a" * 40
        with patch("huggingface_hub.hf_hub_download", return_value=f"/cache/model/snapshots/{revision}/config.json"):
            self.assertEqual(M.weight_revision("model"), revision)
        with patch("huggingface_hub.hf_hub_download", return_value="/local/config.json"):
            self.assertIsNone(M.weight_revision("model"))
        for _ in range(2):
            M.ReferenceFeatures(self.model, "cpu", "p1", revision).get(self.image, self.encode)
        self.assertEqual(self.encode.call_count, 1)

    def test_image_model_preprocessing_and_device_invalidation(self):
        self.cache().get(self.image, self.encode)
        changed = self.image.copy()
        changed.putpixel((0, 0), (1, 2, 3))
        self.cache().get(changed, self.encode)
        self.cache(prep="p2").get(self.image, self.encode)
        self.cache(device=torch.device("cpu:0")).get(self.image, self.encode)
        self.model.config._commit_hash = "revision2"
        self.cache().get(self.image, self.encode)
        self.assertEqual(self.encode.call_count, 5)

    def test_corruption_readonly_disabled_and_unknown_revision(self):
        self.cache().get(self.image, self.encode)
        for p in Path(self.tmp.name).glob("*.npy"):
            p.write_bytes(b"broken")
        self.cache().get(self.image, self.encode)
        self.assertEqual(self.encode.call_count, 2)
        blocked = Path(self.tmp.name) / "blocked"
        blocked.touch()
        for value in (str(blocked), "off"):
            with patch.dict(os.environ, SBV_FEATURE_CACHE=value):
                self.cache().get(self.image, self.encode)
        self.model.config._commit_hash = None
        self.cache().get(self.image, self.encode)
        self.assertEqual(self.encode.call_count, 5)
