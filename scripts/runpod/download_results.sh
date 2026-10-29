#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/connection.sh"

local_results="$runpod_repo_root/results/runpod"
mkdir -p "$local_results"
if (( $# > 1 )); then
    printf 'Usage: %s [batch-name]\n' "$0" >&2
    exit 2
fi
if (( $# == 1 )); then
    batch="$1"
    if [[ ! "$batch" =~ ^batch-[A-Za-z0-9-]+$ ]]; then
        printf 'Invalid batch name: %s\n' "$batch" >&2
        exit 2
    fi
    printf 'Downloading %s to %s\n' "$batch" "$local_results"
    scp "${runpod_scp_options[@]}" -r \
        "$RUNPOD_SSH_TARGET:$runpod_remote_results/$batch" "$local_results/"
else
    printf 'Downloading all pod results to %s\n' "$local_results"
    scp "${runpod_scp_options[@]}" -r \
        "$RUNPOD_SSH_TARGET:$runpod_remote_results/." "$local_results/"
fi
printf 'Download complete: %s\n' "$local_results"
