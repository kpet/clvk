#!/usr/bin/env bash

set -xe

if [ $# -eq 0 ]; then
  echo "Error: Please append path to the test binary path as a command-line argument."
  exit 1
fi

TEMP_DIR="$(mktemp -d)"
pushd "${TEMP_DIR}"

function clean() {
    rm -r "${TEMP_DIR}"
}
trap clean EXIT

# Assign the base path from the argument
binary_path="$1"

cp "${binary_path}/clvk.conf" "${TEMP_DIR}"

# Assing an env var for testing
default_color_val=$(echo CLVK_LOG_COLOUR)
export CLVK_LOG_COLOUR=1

# Also get the default conf file in case a user has it set
default_cofig_file=$(echo CLVK_CONFIG_FILE)

# Run test
CLVK_CONFIG_FILE="${binary_path}/conf_test.conf" \
"${binary_path}/config_test"

# Restore old vals
export CLVK_LOG_COLOUR=${default_color_val}
export CLVK_CONFIG_FILE=${default_cofig_file}

popd
