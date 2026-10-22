#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

for tool in nvidia-smi nvcc cmake ninja git c++; do
    command -v "$tool" >/dev/null || { printf 'Missing required tool: %s\n' "$tool" >&2; exit 1; }
done

nvidia-smi
nvcc --version
cmake --version
ninja --version
git --version
c++ --version
git rev-parse HEAD
git status --porcelain
