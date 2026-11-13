# The private grader base: the verifier image plus the image models' weights, because Harbor runs graders with no
# network. Build it with docker/build_grader.sh, which gathers the weights from this machine's caches. DINOv3's weights
# are gated by Meta's licence: keep this image private (this machine, or a private registry).
ARG BASE=3dscenebench:dev
FROM --platform=linux/amd64 ${BASE}
COPY huggingface /cache/huggingface
COPY torch /cache/torch
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
