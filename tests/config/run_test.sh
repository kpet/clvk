#!/usr/bin/env bash

set -xe

if [ $# -eq 0 ]; then
  echo "Error: Please path to the test binary path as a command-line argument."
  exit 1
fi

TEMP_DIR="$(mktemp -d)"
pushd "${TEMP_DIR}"

function clean() {
    rm -r "${TEMP_DIR}"
}
trap clean EXIT

# Assign the base path from the argument
binary_path=$1

cp "${binary_path}/clvk.conf" "${TEMP_DIR}"

# Run test
CLVK_CONFIG_FILE="${binary_path}/conf_test.conf" \
"${binary_path}/config_test"

popd
