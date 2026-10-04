#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

check_repository() {
    local directory="$1" url="$2" revision="$3"
    if [[ ! -d "$directory/.git" ]]; then
        git clone "$url" "$directory"
        git -C "$directory" checkout --detach "$revision"
    fi
    if [[ "$(git -C "$directory" rev-parse HEAD)" != "$revision" ]]; then
        echo "Unexpected revision in $directory; expected $revision" >&2
        return 1
    fi
    if [[ -n "$(git -C "$directory" status --porcelain)" ]]; then
        echo "Local changes in $directory; use a clean pinned checkout" >&2
        return 1
    fi
}

case "${1:-all}" in
    ecf)
        check_repository external/hdf5_ecf \
            https://github.com/prophesee-ai/hdf5_ecf.git \
            8735ce7161a637cacc5f0affef48129a13ca5a3f
        ;;
    rpg)
        check_repository external/rpg_e2depth \
            https://github.com/uzh-rpg/rpg_e2depth.git \
            17d453c175d94a6e4e666b4dafb64b072eb2df5d
        ;;
    all)
        bash "$0" ecf
        bash "$0" rpg
        ;;
    *)
        echo "Usage: $0 [ecf|rpg|all]" >&2
        exit 2
        ;;
esac
