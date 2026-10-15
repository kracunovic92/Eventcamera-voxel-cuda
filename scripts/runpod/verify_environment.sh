#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

for tool in nvidia-smi nvcc cmake ninja python3 git c++; do
    command -v "$tool" >/dev/null || { printf 'Missing required tool: %s\n' "$tool" >&2; exit 1; }
done

nvidia-smi
nvcc --version
cmake --version
ninja --version
python3 --version
python3 -c 'import numpy; print("NumPy", numpy.__version__)'
git --version
c++ --version
git rev-parse HEAD
git status --porcelain
