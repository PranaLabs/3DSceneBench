#!/usr/bin/env bash
# Build the private grader base (docker/grader.Dockerfile) on top of the verifier image. The weights of SigLIP-2, DINOv3
# and VGG come from this machine's caches, so run the appearance metrics once first. Only the model folders are copied,
# never the Hugging Face login. Usage: docker/build_grader.sh [TAG] [BASE]   (default 3dscenebench-grader:dev on 3dscenebench:dev)
set -euo pipefail
cd "$(dirname "$0")/.."
HUB=${HF_HOME:-$HOME/.cache/huggingface}/hub; TORCH=${TORCH_HOME:-$HOME/.cache/torch}
CTX=$(mktemp -d); trap 'rm -rf "$CTX"' EXIT
for m in google--siglip2-so400m-patch16-naflex facebook--dinov3-vitl16-pretrain-lvd1689m; do   # as in sbv/appearance/metrics.py
    d="$HUB/models--$m"; [ -d "$d/snapshots" ] || { echo "no $d: run the appearance metrics once first" >&2; exit 1; }
    mkdir -p "$CTX/huggingface/hub/models--$m"; cp -R "$d/refs" "$CTX/huggingface/hub/models--$m/"
    cp -RL "$d/snapshots" "$CTX/huggingface/hub/models--$m/"            # real files instead of links into blobs/
    [ -d "$d/.no_exist" ] && cp -R "$d/.no_exist" "$CTX/huggingface/hub/models--$m/"
done
mkdir -p "$CTX/torch/hub/checkpoints"; cp "$TORCH/hub/checkpoints/vgg16-397923af.pth" "$CTX/torch/hub/checkpoints/"
docker build --platform linux/amd64 --build-arg BASE="${2:-3dscenebench:dev}" -f docker/grader.Dockerfile -t "${1:-3dscenebench-grader:dev}" "$CTX"
