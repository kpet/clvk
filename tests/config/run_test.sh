#!/bin/bash

set -xe

if [ $# -eq 0 ]; then
  echo "Error: Please path to the test binary path as a command-line argument."
  exit 1
fi

TEMP_DIR="$(mktemp -d)"
pushd "${TEMP_DIR}"

function clean() {
    rm -f "conf_test.conf"
    rm -f "clvk.conf"
}
##trap clean EXIT

# Assign the base path from the argument
binary_path=$1

# Create the first file with the base path
echo "cache_dir=testing" > "conf_test.conf"

# Create the second file with the base path
echo "cache_dir=failed" > "clvk.conf"
echo "compiler_temp_dir=not/overwritten/" > "clvk.conf"

# Run test
CLVK_CONFIG_FILE="${TEMP_DIR}/conf_test.conf" \
"${binary_path}/config_test"

popd
