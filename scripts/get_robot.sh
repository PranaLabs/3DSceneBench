#!/usr/bin/env bash
# Download the RB-Y1 robot model (MuJoCo Menagerie, Google DeepMind, Apache-2.0) at the commit the verifiers were
# tested with, into data/menagerie/ at the repository root (121 MB). Usage: scripts/get_robot.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=data/menagerie; COMMIT=4d038b3
if [ -d "$DEST/rainbow_robotics_rby1" ]; then echo "already present: $DEST/rainbow_robotics_rby1"; exit 0; fi
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git "$DEST"
cd "$DEST" && git sparse-checkout set rainbow_robotics_rby1 && git checkout -q "$COMMIT"
echo "RB-Y1 model at $(pwd)/rainbow_robotics_rby1 (commit $COMMIT)"
