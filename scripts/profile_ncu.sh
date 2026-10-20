#!/usr/bin/env bash
set -euo pipefail

if ! command -v ncu >/dev/null 2>&1; then
    echo "ncu is unavailable; install Nsight Compute on the GPU host." >&2
    exit 2
fi
if (( $# < 2 )); then
    echo "Usage: $0 <output-prefix> <recording.hdf5> [experiment options]" >&2
    exit 2
fi
profile_prefix=$1
recording=$2
shift 2
ncu --set full --target-processes all --export="${profile_prefix}" \
    ./build/cuda-profile/event_voxel_experiment --source hdf5 --input "${recording}" \
    --mode kernel --max-windows 1 --repeat 1 --warmup 1 "$@"
