#!/usr/bin/env bash
set -euo pipefail

window_us_values="${1:-50000,100000,200000,500000}"
bin_values="${2:-2,5,10}"
block_size="${3:-256}"
repetitions="${4:-1}"
if [[ ! "$window_us_values" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] ||
   [[ ! "$bin_values" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] ||
   [[ ! "$block_size" =~ ^[1-9][0-9]*$ ]] ||
   [[ ! "$repetitions" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Window durations, bins, block size, and repetitions must be positive integers\n' >&2
    exit 2
fi
IFS=, read -r -a windows <<< "$window_us_values"
IFS=, read -r -a bins_list <<< "$bin_values"
for bins in "${bins_list[@]}"; do
    if (( bins < 2 )); then
        printf 'Bins must be at least 2\n' >&2
        exit 2
    fi
done
if (( block_size != 128 && block_size != 256 && block_size != 512 )); then
    printf 'CUDA block size must be 128, 256, or 512\n' >&2
    exit 2
fi

repo_root=/workspace/Eventcamera-voxel-cuda
cd "$repo_root"
raw_dir="$repo_root/data/raw"
shopt -s nullglob
recordings=("$raw_dir"/*.hdf5)
if (( ${#recordings[@]} == 0 )); then
    printf 'No HDF5 recordings found in %s\n' "$raw_dir" >&2
    exit 1
fi
for executable in build/local-release/event_voxel_experiment build/cuda-release/event_voxel_experiment; do
    if [[ ! -x "$executable" ]]; then
        printf 'Missing executable: %s (build the CPU and CUDA presets first)\n' "$executable" >&2
        exit 1
    fi
done

results_parent=/workspace/experiment-results
mkdir -p "$results_parent"
batch_dir="$(mktemp -d "$results_parent/batch-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
commit="$(git rev-parse HEAD)"
{
    printf 'git_commit=%s\n' "$commit"
    printf 'window_us_values=%s\nbin_values=%s\nblock_size=%s\nrepetitions=%s\n' "$window_us_values" "$bin_values" "$block_size" "$repetitions"
    printf 'recordings=%s\n' "${#recordings[@]}"
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$batch_dir/batch-info.txt"
printf 'recording\twindow_us\tbins\trepetition\tbackend\tstatus\n' > "$batch_dir/runs.tsv"
printf 'Results: %s\n' "$batch_dir"

run_one() {
    local recording="$1" window_us="$2" bins="$3" repetition="$4" backend="$5"
    local name="${recording##*/}"
    local log="$batch_dir/${name%.hdf5}-w${window_us}us-b${bins}-rep${repetition}-${backend}.log"
    local -a command
    if [[ "$backend" == cpu ]]; then
        command=(build/local-release/event_voxel_experiment)
    else
        command=(build/cuda-release/event_voxel_experiment)
    fi
    command+=(--input "$recording" --window-us "$window_us" --bins "$bins" --results-root "$batch_dir")
    if [[ "$backend" != cpu ]]; then
        command+=(--strategy "$backend" --block-size "$block_size")
    fi
    printf 'Running %s, %s us, %s bins, repetition %s, %s\n' "$name" "$window_us" "$bins" "$repetition" "$backend"
    if "${command[@]}" > "$log" 2>&1; then
        printf '%s\t%s\t%s\t%s\t%s\tok\n' "$name" "$window_us" "$bins" "$repetition" "$backend" >> "$batch_dir/runs.tsv"
    else
        printf '%s\t%s\t%s\t%s\t%s\tfailed\n' "$name" "$window_us" "$bins" "$repetition" "$backend" >> "$batch_dir/runs.tsv"
        cat "$log" >&2
        printf 'Benchmark failed; partial results remain in %s\n' "$batch_dir" >&2
        exit 1
    fi
}

for recording in "${recordings[@]}"; do
    for window_us in "${windows[@]}"; do
        for bins in "${bins_list[@]}"; do
            for (( repetition=1; repetition<=repetitions; repetition++ )); do
                run_one "$recording" "$window_us" "$bins" "$repetition" cpu
                for strategy in event-scatter pixel-gather sort-reduce; do
                    run_one "$recording" "$window_us" "$bins" "$repetition" "$strategy"
                done
            done
        done
    done
done
printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$batch_dir/batch-info.txt"
printf 'Completed %d benchmark runs in %s\n' "$(( ${#recordings[@]} * ${#windows[@]} * ${#bins_list[@]} * repetitions * 4 ))" "$batch_dir"
