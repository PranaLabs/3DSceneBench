"""Identify locally available metric weights without loading models or using the network."""

import importlib.util
import os
from pathlib import Path
import json
import subprocess


def local_weights():
    from huggingface_hub import hf_hub_download
    from sbv.common.dependencies import sha

    result = {}
    for repo in ("google/siglip2-so400m-patch16-naflex", "facebook/dinov3-vitl16-pretrain-lvd1689m"):
        config = Path(hf_hub_download(repo, "config.json", local_files_only=True))
        folder = config.parent
        if not any(p.suffix in (".bin", ".safetensors") for p in folder.iterdir()):
            return None
        result[repo] = {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()}
    spec = importlib.util.find_spec("lpips")
    if spec is None:
        return None
    lpips = Path(spec.origin).parent / "weights/v0.1/vgg.pth"
    home = Path(
        os.environ.get("TORCH_HOME") or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "torch"
    )
    vgg = home / "hub/checkpoints/vgg16-397923af.pth"
    if not lpips.is_file() or not vgg.is_file():
        return None
    result["lpips_vgg"] = {"lpips": sha(lpips), "vgg": sha(vgg)}
    return result


def identity(python):
    from sbv.common.dependencies import memo, sha, runtime, SCOPES

    if not Path(python).is_file():
        return None
    root = str(Path(__file__).resolve().parents[2])
    code = (
        "import sys,json;sys.path.insert(0,"
        + repr(root)
        + ");from sbv.appearance.cache_identity import local_weights;print(json.dumps(local_weights()))"
    )

    def query():
        try:
            weights = json.loads(
                subprocess.check_output([python, "-c", code], text=True, stderr=subprocess.PIPE, timeout=120)
            )
            return (
                None
                if weights is None
                else dict(weights=weights, runtime=runtime(python, packages=SCOPES["appearance"]))
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            return None

    return memo(
        (
            "metric-weights",
            sha(python),
            tuple((k, os.environ.get(k)) for k in ("HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME")),
        ),
        query,
    )
