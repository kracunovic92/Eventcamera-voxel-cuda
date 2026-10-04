
#!/usr/bin/env bash

# Configure the Prophesee HDF5 ECF decoder.
#
# Usage:
#   source scripts/hdf5_env.sh
#
# Must be sourced because the environment variables need
# to remain available in the current terminal session.

# Prevent accidental execution instead of sourcing.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "ERROR: This script must be sourced." >&2
    echo "Usage: source scripts/hdf5_env.sh" >&2
    exit 1
fi

SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &&
        pwd -P
)" || return 1

PROJECT_ROOT="$(
    git -C "${SCRIPT_DIR}" rev-parse --show-toplevel
)" || {
    echo "ERROR: Cannot determine repository root." >&2
    return 1
}

ECF_PLUGIN_DIR="${PROJECT_ROOT}/build/hdf5_ecf"
ECF_PLUGIN="${ECF_PLUGIN_DIR}/libH5Zecf.so"

# Fail immediately if the plugin hasn't been compiled.
if [[ ! -f "${ECF_PLUGIN}" ]]; then
    echo "ERROR: HDF5 ECF plugin not found:" >&2
    echo "  ${ECF_PLUGIN}" >&2
    echo "Build external/hdf5_ecf first." >&2
    return 1
fi

# HDF5 searches this directory for external filters.
export HDF5_PLUGIN_PATH="${ECF_PLUGIN_DIR}"

# The ECF plugin may depend on another library
# located in the same build directory.
export LD_LIBRARY_PATH="${ECF_PLUGIN_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "Prophesee HDF5 environment configured."
echo "HDF5_PLUGIN_PATH=${HDF5_PLUGIN_PATH}"
