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

output_root="${EVENT_VOXEL_RESULT_ROOT:-/workspace/experiment-results}"
dataset_root="${EVENT_VOXEL_DATA_ROOT:-/workspace/benchmark-datasets}"
mkdir -p "$output_root"
session_root="$(mktemp -d "$output_root/session-XXXXXXXX")"
temporary_dir="$(mktemp -d)"
trap 'rm -rf "$temporary_dir"' EXIT

archive_results() {
    local archive
    archive="$(mktemp --suffix=.tar.gz "/workspace/event-voxel-results-$(git rev-parse --short HEAD)-XXXXXXXX")"
    tar -C "$session_root" -czf "$archive" .
    printf 'Result archive: %s\n' "$archive"
    sha256sum "$archive"
}

./scripts/runpod/verify_environment.sh | tee "$temporary_dir/environment.txt"

if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
    printf 'Official RunPod run requires a clean Git working tree.\n' >&2
    exit 1
fi

dataset_args=()
if [[ -n "${EVENT_VOXEL_DATA_ARCHIVE:-}" ]]; then
    dataset_args=(--archive "$EVENT_VOXEL_DATA_ARCHIVE")
elif [[ -n "${EVENT_VOXEL_DATA_URL:-}" ]]; then
    dataset_args=(--url "$EVENT_VOXEL_DATA_URL")
fi
python3 scripts/runpod/prepare_datasets.py \
    --config configs/cuda_benchmark_smoke.json \
    --config configs/cuda_benchmark_matrix.json \
    --destination "$dataset_root" \
    "${dataset_args[@]}"

cmake --preset cuda-release
cmake --build --preset cuda-release --parallel
ctest --preset cuda-release --output-on-failure | tee "$temporary_dir/correctness.txt"
./build/cuda-release/tests/event_voxel_tests '[cuda]' | tee -a "$temporary_dir/correctness.txt"
python3 scripts/benchmark/capture_real.py \
    --config configs/cuda_benchmark_smoke.json \
    --dataset-root "$dataset_root" \
    --cpu-executable build/cuda-release/event_voxel_run_cpu \
    --cuda-executable build/cuda-release/event_voxel_run_cuda \
    --output-root "$session_root/data/outputs/smoke" \
    --run-path-file "$temporary_dir/smoke_capture_path"
python3 scripts/benchmark/compare_outputs.py "$(cat "$temporary_dir/smoke_capture_path")" \
    | tee -a "$temporary_dir/correctness.txt"

python3 scripts/benchmark/run_matrix.py \
    --config configs/cuda_benchmark_smoke.json \
    --executable build/cuda-release/event_voxel_bench_cuda \
    --dataset-root "$dataset_root" \
    --output-root "$session_root/smoke" \
    --correctness-file "$temporary_dir/correctness.txt" \
    --environment-file "$temporary_dir/environment.txt" \
    --run-path-file "$temporary_dir/smoke_path"
python3 scripts/benchmark/summarize_results.py "$(cat "$temporary_dir/smoke_path")/measurements.csv"

if [[ "$smoke_only" == true ]]; then
    printf 'Smoke: %s\n' "$(cat "$temporary_dir/smoke_path")"
    archive_results
    exit 0
fi

python3 scripts/benchmark/capture_real.py \
    --config configs/cuda_benchmark_matrix.json \
    --dataset-root "$dataset_root" \
    --cpu-executable build/cuda-release/event_voxel_run_cpu \
    --cuda-executable build/cuda-release/event_voxel_run_cuda \
    --output-root "$session_root/data/outputs/full" \
    --run-path-file "$temporary_dir/full_capture_path"
python3 scripts/benchmark/compare_outputs.py "$(cat "$temporary_dir/full_capture_path")" \
    | tee -a "$temporary_dir/correctness.txt"

python3 scripts/benchmark/run_matrix.py \
    --config configs/cuda_benchmark_matrix.json \
    --executable build/cuda-release/event_voxel_bench_cuda \
    --dataset-root "$dataset_root" \
    --output-root "$session_root/full" \
    --correctness-file "$temporary_dir/correctness.txt" \
    --environment-file "$temporary_dir/environment.txt" \
    --run-path-file "$temporary_dir/full_path"
python3 scripts/benchmark/summarize_results.py "$(cat "$temporary_dir/full_path")/measurements.csv"

printf 'Smoke: %s\nFull: %s\n' "$(cat "$temporary_dir/smoke_path")" "$(cat "$temporary_dir/full_path")"
archive_results
