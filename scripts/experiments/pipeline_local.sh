#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--smoke-only" ) ]]; then
    printf 'Usage: %s [--smoke-only]\n' "$0" >&2
    exit 2
fi

smoke_only=false
if [[ $# -eq 1 ]]; then
    smoke_only=true
fi

[[ -x .venv/bin/python ]] || { printf 'Run make setup first.\n' >&2; exit 1; }
if [[ "$smoke_only" == false && -n "$(git status --porcelain --untracked-files=normal)" ]]; then
    printf 'A full local run requires a clean Git working tree.\n' >&2
    exit 1
fi

bash scripts/setup_external.sh rpg
if [[ "$smoke_only" == true ]]; then
    cuda_config=configs/cuda_benchmark_smoke.json
    cpu_config=configs/benchmark_smoke.json
    make data-prepare DATASET=80_balls WINDOW_ID=first_100000 WINDOW_MAX_EVENTS=100000
    for bins in 2 5; do
        make verify-release DATASET=80_balls WINDOW_ID=first_100000 BINS="$bins"
    done
else
    cuda_config=configs/cuda_benchmark_matrix.json
    cpu_config=configs/benchmark_matrix.json
    make verify-all
    make data-prepare DATASET=sparklers WINDOW_ID=full
fi

output_root="${EVENT_VOXEL_LOCAL_RESULT_ROOT:-results/experiments/local}"
mkdir -p "$output_root"
session_root="$(mktemp -d "$output_root/session-XXXXXXXX")"
run_path_file="$(mktemp)"
trap 'rm -f "$run_path_file"' EXIT
matrix_args=()
summary_args=()
if [[ "$smoke_only" == true ]]; then
    matrix_args=(--allow-dirty)
    summary_args=(--include-dirty)
fi

.venv/bin/python scripts/experiments/capture_real.py \
    --config "$cuda_config" \
    --dataset-root data/processed \
    --cpu-executable build/cpu-benchmark-release/event_voxel_run_cpu \
    --output-root "$session_root/outputs" \
    --run-path-file "$run_path_file"
.venv/bin/python scripts/experiments/validate_outputs.py capture "$(cat "$run_path_file")" \
    --dataset-root data/processed

.venv/bin/python tools/run_benchmark_matrix.py \
    --config "$cpu_config" \
    --executable build/cpu-benchmark-release/event_voxel_bench_cpu \
    --output-root "$session_root/benchmarks" \
    "${matrix_args[@]}"
.venv/bin/python tools/summarize_benchmarks.py \
    --runs "$session_root/benchmarks" \
    --output "$session_root/cpu_summary.json" \
    "${summary_args[@]}"
.venv/bin/python scripts/experiments/package_datasets.py \
    --config "$cuda_config" \
    --output "$session_root/datasets.tar.gz"

printf 'Local experiment session: %s\n' "$session_root"
printf 'Transfer this verified dataset archive to RunPod: %s\n' "$session_root/datasets.tar.gz"
