#!/usr/bin/env python
"""The three Appearance metrics on matching reference and generated renders (doc, Appearance: detailed metrics).
Runs in the PyTorch environment (.venv-metrics, requirements/metrics.txt), not in the MuJoCo one.
  SigLIP-2 So400m/16 NaFlex: the checkpoint's image processor with max_num_patches=1024, pooled image features,
    L2-normalized, cosine similarity (higher is better).
  DINOv3 ViT-L/16: 512 x 512 without cropping, RGB in [0, 1] with ImageNet normalization, the final normalized CLS
    token, cosine similarity (higher is better). The checkpoint is gated: accept Meta's licence on Hugging Face and log
    in.
  LPIPS-VGG v0.1: images scaled to [-1, 1], distance (lower is better).
Usage: .venv-metrics/bin/python sbv/appearance/metrics.py REF_DIR GEN_DIR --out metrics.json [--device cpu]
Views are the PNG files present in both folders; the room score is the mean over views.
"""

import argparse, hashlib, json, os, tempfile
from pathlib import Path
import PIL, transformers
import numpy as np, torch
from PIL import Image

SIGLIP, DINO = "google/siglip2-so400m-patch16-naflex", "facebook/dinov3-vitl16-pretrain-lvd1689m"
MEAN, STD = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def cosine(a, b):
    a, b = torch.nn.functional.normalize(a.float(), dim=-1), torch.nn.functional.normalize(b.float(), dim=-1)
    return float((a * b).sum())


def weight_revision(model_id):
    # Resolve before loading so both the model and cache use this exact snapshot.
    # Transformers 5 drops config._commit_hash; reading that field alone disables
    # every cache entry on the pinned runtime.
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(model_id, "config.json"))
    revision = path.parent.name
    return (
        revision
        if path.parent.parent.name == "snapshots"
        and len(revision) == 40
        and all(c in "0123456789abcdef" for c in revision)
        else None
    )


class ReferenceFeatures:
    """Cache only reference embeddings in a private evaluator-owned directory.

    Unknown weight revisions bypass the cache. Image content, model revision,
    preprocessing, metric source, device and library versions all invalidate it.
    LPIPS stays on its original paired-image path.
    """

    def __init__(self, model, device, preprocessing, revision=None):
        self.device = device
        self.folder = os.environ.get("SBV_FEATURE_CACHE", os.path.expanduser("~/.cache/scenebench/features"))
        revision = revision or getattr(model.config, "_commit_hash", None)
        self.enabled = bool(revision and self.folder and self.folder != "off")
        self.signature = json.dumps(
            [
                model.config.to_dict(),
                revision,
                str(device),
                preprocessing,
                torch.__version__,
                transformers.__version__,
                np.__version__,
                PIL.__version__,
            ],
            sort_keys=True,
        )

    def get(self, img, compute):
        if not self.enabled:
            return compute(img)
        h = hashlib.sha256(self.signature.encode())
        h.update(Path(__file__).read_bytes())
        h.update(str((img.mode, img.size)).encode())
        h.update(img.tobytes())
        path = Path(self.folder) / (h.hexdigest() + ".npy")
        try:
            a = np.load(path, allow_pickle=False)
            if a.dtype == np.float32 and a.ndim == 2 and a.shape[0] == 1 and np.isfinite(a).all():
                return torch.from_numpy(a).to(self.device)
        except (OSError, ValueError, EOFError):
            pass
        result = compute(img)
        tmp = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
                tmp = f.name
                np.save(f, result.detach().cpu().numpy(), allow_pickle=False)
            os.replace(tmp, path)
        except OSError:
            pass
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)
        return result


def siglip(pairs, device):
    from transformers import AutoImageProcessor, AutoModel

    revision = weight_revision(SIGLIP)
    proc = AutoImageProcessor.from_pretrained(SIGLIP, revision=revision)
    model = AutoModel.from_pretrained(SIGLIP, revision=revision).to(device).eval()
    cache = ReferenceFeatures(model, device, [proc.to_dict(), {"max_num_patches": 1024}], revision)

    def emb(img):
        x = proc(images=[img], max_num_patches=1024, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.get_image_features(**x)
        return getattr(out, "pooler_output", out)

    return [cosine(cache.get(a, emb), emb(b)) for a, b in pairs]


def dinov3(pairs, device):
    from transformers import AutoModel

    revision = weight_revision(DINO)
    model = AutoModel.from_pretrained(DINO, revision=revision).to(device).eval()
    cache = ReferenceFeatures(model, device, [MEAN.tolist(), STD.tolist()], revision)

    def emb(img):
        x = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)[None]
        x = ((x - MEAN) / STD).to(device)
        with torch.no_grad():
            out = model(pixel_values=x)
        return out.pooler_output

    return [cosine(cache.get(a, emb), emb(b)) for a, b in pairs]


def lpips_vgg(pairs, device):
    import lpips

    net = lpips.LPIPS(net="vgg", version="0.1", lpips=True, verbose=False).to(device).eval()
    t = lambda img: (torch.from_numpy(np.asarray(img, np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]).to(device)
    with torch.no_grad():
        return [float(net(t(a), t(b))) for a, b in pairs]


def images(ref, gen):
    names = sorted(f for f in os.listdir(ref) if f.endswith(".png") and os.path.exists(os.path.join(gen, f)))

    def rgb(path):
        with Image.open(path) as img:
            return img.convert("RGB")

    pairs = [(rgb(os.path.join(ref, n)), rgb(os.path.join(gen, n))) for n in names]
    return names, pairs


def score_many(jobs, device):
    """Load each model once for a batch of renders; keep the original per-image math.

    Model functions release their weights before the next metric loads. If a
    batch metric fails, retry its jobs separately to isolate the failing input.
    Callers bound batch size; decoded images, not all three models, stay resident.
    """
    batches, reports = [], []
    for job in jobs:
        try:
            names, pairs = images(job["ref"], job["gen"])
            report = dict(views=[n[:-4] for n in names], per_view={}, mean={}, unavailable={})
        except Exception as e:
            pairs = None
            report = dict(status=f"{type(e).__name__}: {str(e)[:300]}")
        batches.append(pairs)
        reports.append(report)
    active = [(r, p) for r, p in zip(reports, batches) if p is not None]
    for key, fn in (("siglip2", siglip), ("dinov3", dinov3), ("lpips_vgg", lpips_vgg)):

        def record(out, vals):
            out["per_view"][key] = [round(v, 5) for v in vals]
            out["mean"][key] = round(float(np.mean(vals)), 5) if vals else None

        if not active:
            continue
        try:
            vals = fn([pair for _, pairs in active for pair in pairs], device)
            start = 0
            for out, pairs in active:
                record(out, vals[start : start + len(pairs)])
                start += len(pairs)
        except Exception as batch_error:
            for out, pairs in active:
                try:
                    if len(active) == 1:
                        raise batch_error
                    record(out, fn(pairs, device))
                except Exception as e:
                    out["mean"][key] = None
                    out["unavailable"][key] = f"{type(e).__name__}: {str(e)[:300]}"
    return reports


def main(ref, gen, device):
    return score_many([dict(ref=ref, gen=gen)], device)[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", nargs="?")
    ap.add_argument("gen", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--jobs", help="JSON list of {ref, gen, out}; reuse models across these render pairs")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.jobs:
        if a.ref or a.gen or a.out:
            ap.error("--jobs cannot be combined with ref/gen/--out")
        with open(a.jobs) as f:
            jobs = json.load(f)
    else:
        if not (a.ref and a.gen and a.out):
            ap.error("ref, gen and --out are required without --jobs")
        jobs = [dict(ref=a.ref, gen=a.gen, out=a.out)]
    for job, res in zip(jobs, score_many(jobs, a.device)):
        with open(job["out"], "w") as f:
            json.dump(res, f, indent=1)
        print(json.dumps(res.get("mean")), json.dumps(res.get("unavailable", res.get("status")))[:300])
