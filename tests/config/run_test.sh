#!/usr/bin/env bash

set -xe

if [ $# -ne 2 ]; then
  echo "ERROR: USAGE: ${0} <config_test-path> <config_files-path>"
  exit 1
fi

# Assign the base path from the argument
binary_path="$(realpath $1)"
config_files_path=$2

# Make temporary directory
TEMP_DIR="$(mktemp -d)"
function clean() {
    rm -r "${TEMP_DIR}"
}
trap clean EXIT

# Copy assets to temporary directory
cp "${config_files_path}/conf_test.conf" "${TEMP_DIR}"
cp "${config_files_path}/clvk.conf" "${TEMP_DIR}"

# Run test
pushd "${TEMP_DIR}"
CLVK_CONFIG_FILE="${TEMP_DIR}/conf_test.conf" \
CLVK_LOG_COLOUR=1 \
"${binary_path}/config_test"
popd
