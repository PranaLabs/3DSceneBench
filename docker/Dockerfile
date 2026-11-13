# The 3DSceneBench verifiers in one image: physics, robot tasks, renders, image metrics and the fixed pipeline
# (sbv/pipeline.py). Built for linux/amd64, because Blender's Python module has no Linux ARM build; on an Apple Silicon Mac
# it runs under emulation, natively on an x86 Linux server. Build from the repository root with docker/build.sh.
FROM --platform=linux/amd64 python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libegl1 libgomp1 libsm6 libxext6 libxfixes3 \
        libxi6 libxkbcommon0 libxrender1 libxxf86vm1 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/scenebench

# Physics, robot and renders in .venv; the image metrics in .venv-metrics (Blender's module needs NumPy 1.x); Genesis in .venv-nonrigid
COPY requirements requirements
RUN python -m venv .venv && .venv/bin/pip install --no-cache-dir -r requirements/core.txt -r requirements/render.txt
RUN python -m venv .venv-metrics \
    && .venv-metrics/bin/pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchvision==0.29.1 \
    && .venv-metrics/bin/pip install --no-cache-dir -r requirements/metrics.txt

# Genesis for the non-rigid verifiers in .venv-nonrigid (Genesis needs NumPy 2). Genesis cloth remeshes with pymeshlab,
# whose plugins need libOpenGL.so.0 (libopengl0): without it a cloth run crashes in the grader
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libopengl0 && rm -rf /var/lib/apt/lists/* \
    && python -m venv .venv-nonrigid \
    && .venv-nonrigid/bin/pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \
    && .venv-nonrigid/bin/pip install --no-cache-dir -r requirements/nonrigid.txt

# The robot: MuJoCo Menagerie's RB-Y1 at the tested commit, where sbv/functionality/f4_robot/rby1.py looks for it
COPY scripts/get_robot.sh scripts/
RUN scripts/get_robot.sh && rm -rf data/menagerie/.git

COPY . .

# Model weights are not in the image: they download on first use into /cache (docker/run.sh mounts it so they stay).
# DINOv3 is gated: the account behind the mounted Hugging Face login, or HF_TOKEN, must have accepted Meta's licence.
ENV HF_HOME=/cache/huggingface TORCH_HOME=/cache/torch PYTHONUNBUFFERED=1 \
    SBV_NONRIGID_PYTHON=/opt/scenebench/.venv-nonrigid/bin/python GS_BACKEND=cpu PYGLET_HEADLESS=true PYOPENGL_PLATFORM=egl
ENTRYPOINT ["/opt/scenebench/.venv/bin/python", "-m", "sbv.pipeline"]
CMD ["run", "--work", "/work"]
