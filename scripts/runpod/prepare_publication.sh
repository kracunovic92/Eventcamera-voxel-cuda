#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if (( $# != 1 )) || [[ ! "$1" =~ ^batch-[A-Za-z0-9-]+$ ]]; then
    printf 'Usage: %s batch-NAME\n' "$0" >&2
    exit 2
fi
batch="$1"
source_dir="$repo_root/results/runpod/$batch"
published_dir="$repo_root/results/published/runpod/$batch"
if [[ ! -d "$source_dir" ]]; then
    printf 'Downloaded batch not found: %s\n' "$source_dir" >&2
    exit 1
fi
if [[ -e "$published_dir" ]]; then
    printf 'Publication already exists: %s\n' "$published_dir" >&2
    exit 1
fi

python3 "$repo_root/scripts/runpod/summarize_batch.py" "$source_dir"
mkdir -p "$published_dir"
cp "$source_dir/batch-info.txt" "$source_dir/runs.tsv" \
   "$source_dir/comparison.csv" "$source_dir/stage_summary.csv" "$published_dir/"
# Keep every per-window CSV and JSON in a compact archive; logs stay local.
tar -czf "$published_dir/raw-results.tar.gz" --exclude='*.log' \
    -C "$repo_root/results/runpod" "$batch"
(
    cd "$published_dir"
    sha256sum raw-results.tar.gz > SHA256SUMS
)
{
    printf '# RunPod benchmark batch %s\n\n' "$batch"
    printf 'Source revision: %s\n\n' "$(sed -n 's/^git_commit=//p' "$source_dir/batch-info.txt")"
    cat <<'EOF'
- `comparison.csv`: one row per recording, window duration, bin count, and backend.
- `stage_summary.csv`: stage timings and transferred byte counts for each run.
- `raw-results.tar.gz`: per-window `windows.csv` and `samples.csv`, plus run metadata and summaries.
- `SHA256SUMS`: checksum for the archive; verify with `sha256sum -c SHA256SUMS`.

`strategy` covers grid clear, preprocessing, and final CUDA kernel execution.
`h2d-events` and `d2h-grid` are host-to-device and device-to-host copies.
`processing-total` covers buffer preparation and the CUDA pipeline; `run_wall_ms`
also includes GPU setup, input preparation, CSV writing, and loop overhead.
`other_run_wall_ms` is the residual after measured input preparation and
processing; it does not isolate those remaining activities. Stage intervals
overlap, so do not
sum every row in `stage_summary.csv`. See `docs/streaming-experiments.md`.
EOF
} > "$published_dir/README.md"
printf 'Prepared tracked publication: %s\n' "$published_dir"
