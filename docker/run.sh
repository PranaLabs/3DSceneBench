#!/usr/bin/env bash
# Run the verifier image on a work folder: WORK/rooms and WORK/submissions in, WORK/results out (layout in sbv/pipeline.py).
# Usage: docker/run.sh WORK [command ...]    e.g.  docker/run.sh ~/scenebench          (the whole run)
#                                                  docker/run.sh ~/scenebench selftest
# Model weights stay between runs in ~/.cache/huggingface and ~/.cache/torch (SigLIP-2 4.5 GB, DINOv3 1.2 GB, VGG 0.5 GB).
# DINOv3 is gated: a Hugging Face login on this machine (hf auth login, stored in ~/.cache/huggingface) is carried in by
# that mount, or export HF_TOKEN; the account must have accepted Meta's licence.
set -euo pipefail
WORK=$(cd "$1" && pwd); shift
mkdir -p "$HOME/.cache/huggingface" "$HOME/.cache/torch"
exec docker run --rm --platform linux/amd64 --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$WORK":/work -v "$HOME/.cache/huggingface":/cache/huggingface \
    -v "$HOME/.cache/torch":/cache/torch -e HF_TOKEN "${SBV_IMAGE:-3dscenebench:dev}" "${@:-run}"
