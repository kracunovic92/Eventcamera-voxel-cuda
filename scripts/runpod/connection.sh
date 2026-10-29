#!/usr/bin/env bash

runpod_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runpod_remote_repo=/workspace/Eventcamera-voxel-cuda
runpod_remote_results=/workspace/experiment-results

: "${RUNPOD_SSH_TARGET:?Set RUNPOD_SSH_TARGET to the SSH over exposed TCP target, for example root@213.173.99.11}"
: "${RUNPOD_SSH_PORT:?Set RUNPOD_SSH_PORT to the current SSH over exposed TCP port from RunPod Connect}"
RUNPOD_SSH_KEY="${RUNPOD_SSH_KEY:-$HOME/.ssh/id_rsa}"

if [[ ! "$RUNPOD_SSH_TARGET" =~ ^[a-z_][a-z0-9_-]*@[A-Za-z0-9.-]+$ ]]; then
    printf 'Invalid RUNPOD_SSH_TARGET: expected user@hostname or user@IPv4-address\n' >&2
    exit 2
fi
if [[ ! "$RUNPOD_SSH_PORT" =~ ^[0-9]+$ ]] || (( 10#$RUNPOD_SSH_PORT < 1 || 10#$RUNPOD_SSH_PORT > 65535 )); then
    printf 'Invalid RUNPOD_SSH_PORT: expected 1–65535\n' >&2
    exit 2
fi
if [[ ! -f "$RUNPOD_SSH_KEY" ]]; then
    printf 'SSH private key not found: %s\n' "$RUNPOD_SSH_KEY" >&2
    exit 2
fi

runpod_ssh_options=(-o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -i "$RUNPOD_SSH_KEY" -p "$RUNPOD_SSH_PORT")
runpod_scp_options=(-o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -i "$RUNPOD_SSH_KEY" -P "$RUNPOD_SSH_PORT")
if [[ -n "${RUNPOD_KNOWN_HOSTS:-}" ]]; then
    runpod_ssh_options+=(-o StrictHostKeyChecking=yes -o "UserKnownHostsFile=$RUNPOD_KNOWN_HOSTS")
    runpod_scp_options+=(-o StrictHostKeyChecking=yes -o "UserKnownHostsFile=$RUNPOD_KNOWN_HOSTS")
fi
