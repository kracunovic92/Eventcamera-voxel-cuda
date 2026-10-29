#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/connection.sh"

raw_dir="$runpod_repo_root/data/raw"
shopt -s nullglob dotglob
entries=("$raw_dir"/*)
if (( ${#entries[@]} == 0 )); then
    printf 'No files found in %s\n' "$raw_dir" >&2
    exit 1
fi

printf 'Uploading %d top-level item(s) from %s\n' "${#entries[@]}" "$raw_dir"
ssh "${runpod_ssh_options[@]}" "$RUNPOD_SSH_TARGET" "mkdir -p '$runpod_remote_repo/data/raw'"
scp "${runpod_scp_options[@]}" -r "${entries[@]}" "$RUNPOD_SSH_TARGET:$runpod_remote_repo/data/raw/"
printf 'Upload complete: %s/data/raw/\n' "$runpod_remote_repo"
