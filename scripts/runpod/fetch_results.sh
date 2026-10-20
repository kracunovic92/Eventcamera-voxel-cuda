#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <pod-host> <ssh-port> <private-key> <remote-archive> [local-directory] [rclone-destination]" >&2
    echo "Example destination: mydrive:event-voxel-results" >&2
    exit 2
}

(( $# >= 4 && $# <= 6 )) || usage
host=$1
port=$2
key=$3
remote_archive=$4
local_directory=${5:-results/downloads}
rclone_destination=${6:-}

[[ $host =~ ^[A-Za-z0-9.-]+$ && $port =~ ^[0-9]+$ ]] || usage
port_number=$((10#$port))
(( port_number >= 1 && port_number <= 65535 )) || usage
[[ -f $key ]] || { echo "SSH key does not exist: $key" >&2; exit 2; }
[[ $remote_archive =~ ^/workspace/event-voxel-results-[A-Za-z0-9._-]+\.tar\.gz$ ]] || usage
for tool in ssh scp sha256sum mktemp; do
    command -v "$tool" >/dev/null || { echo "Missing required tool: $tool" >&2; exit 2; }
done
if [[ -n $rclone_destination ]]; then
    command -v rclone >/dev/null || { echo "Install rclone before using a remote destination" >&2; exit 2; }
fi

remote_line=$(ssh -o BatchMode=yes -i "$key" -p "$port" "root@$host" \
    sha256sum "$remote_archive")
expected=${remote_line%% *}
[[ $expected =~ ^[a-fA-F0-9]{64}$ ]] || { echo "Invalid remote SHA-256 response" >&2; exit 1; }

mkdir -p "$local_directory"
filename=${remote_archive##*/}
local_archive=$local_directory/$filename
if [[ -f $local_archive ]]; then
    actual=$(sha256sum "$local_archive")
    actual=${actual%% *}
    [[ $actual == "$expected" ]] || {
        echo "Existing local archive has a different hash: $local_archive" >&2
        exit 1
    }
else
    temporary=$(mktemp "$local_directory/.${filename}.XXXXXX")
    trap 'rm -f "$temporary"' EXIT
    scp -o BatchMode=yes -i "$key" -P "$port" \
        "root@$host:$remote_archive" "$temporary"
    actual=$(sha256sum "$temporary")
    actual=${actual%% *}
    [[ $actual == "$expected" ]] || { echo "Downloaded archive SHA-256 mismatch" >&2; exit 1; }
    mv "$temporary" "$local_archive"
    trap - EXIT
fi

printf '%s  %s\n' "$expected" "$filename" > "$local_archive.sha256"
echo "Verified local archive: $local_archive"
if [[ -n $rclone_destination ]]; then
    rclone copyto "$local_archive" "${rclone_destination%/}/$filename"
    rclone copyto "$local_archive.sha256" "${rclone_destination%/}/$filename.sha256"
    echo "Copied archive and checksum to $rclone_destination"
fi
