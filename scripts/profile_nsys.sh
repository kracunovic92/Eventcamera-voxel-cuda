#!/usr/bin/env bash
set -euo pipefail

if ! command -v nsys >/dev/null 2>&1; then
    echo "nsys is unavailable; install Nsight Systems on the GPU host." >&2
    exit 2
fi
if (( $# < 2 )); then
    echo "Usage: $0 <output-prefix> <recording.hdf5> [experiment options]" >&2
    exit 2
fi
profile_prefix=$1
recording=$2
shift 2
mkdir -p "$(dirname "$profile_prefix")"
nsys profile --trace=cuda,nvtx,osrt --sample=none --output="${profile_prefix}" \
    ./build/cuda-profile/event_voxel_experiment --input "${recording}" \
    "$@"
