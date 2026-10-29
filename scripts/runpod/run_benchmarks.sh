#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/connection.sh"

window_us_values=50000,100000,200000,500000
bin_values=2,5,10
block_size=256
repetitions=1

if (( $# > 0 )) && [[ "$1" != --* ]]; then
    # Keep the original positional interface (one duration in microseconds).
    if (( $# > 4 )); then
        printf 'Too many positional arguments\n' >&2
        exit 2
    fi
    window_us_values="$1"
    bin_values="${2:-5}"
    block_size="${3:-256}"
    repetitions="${4:-1}"
else
    while (( $# > 0 )); do
        option="$1"
        if (( $# < 2 )); then
            printf 'Missing value for %s\n' "$option" >&2
            exit 2
        fi
        value="$2"
        case "$option" in
            --window-ms)
                if [[ ! "$value" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]]; then
                    printf 'Expected comma-separated positive milliseconds for --window-ms\n' >&2
                    exit 2
                fi
                IFS=, read -r -a milliseconds <<< "$value"
                microseconds=()
                for ms in "${milliseconds[@]}"; do
                    microseconds+=("$((ms * 1000))")
                done
                window_us_values="$(IFS=,; printf '%s' "${microseconds[*]}")"
                ;;
            --bins) bin_values="$value" ;;
            --block-size) block_size="$value" ;;
            --repetitions) repetitions="$value" ;;
            *) printf 'Unknown option: %s\n' "$option" >&2; exit 2 ;;
        esac
        shift 2
    done
fi

if [[ ! "$window_us_values" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] ||
   [[ ! "$bin_values" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] ||
   [[ ! "$block_size" =~ ^[1-9][0-9]*$ ]] ||
   [[ ! "$repetitions" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Expected positive integer durations, bins, block size, and repetitions\n' >&2
    exit 2
fi

ssh "${runpod_ssh_options[@]}" "$RUNPOD_SSH_TARGET" \
    "bash -s -- $window_us_values $bin_values $block_size $repetitions" \
    < "$(dirname "${BASH_SOURCE[0]}")/benchmark_on_pod.sh"
