#!/usr/bin/env bash
# Build the verifier image from the repository root. Usage: docker/build.sh [TAG]   (default 3dscenebench:dev)
set -euo pipefail
cd "$(dirname "$0")/.."
exec docker build --platform linux/amd64 -f docker/Dockerfile -t "${1:-3dscenebench:dev}" .
