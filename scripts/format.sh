#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
find apps benchmarks include src tests -type f \
    \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \) \
    -print0 | xargs -0 -r clang-format-19 -i
