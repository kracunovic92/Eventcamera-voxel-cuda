#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PUBLIC_KEY:-}" ]]; then
    mkdir -p /root/.ssh /run/sshd
    chmod 700 /root/.ssh
    printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    ssh-keygen -A
    /usr/sbin/sshd
else
    printf 'PUBLIC_KEY is unset; SSH/SCP access is unavailable.\n' >&2
fi

exec sleep infinity
