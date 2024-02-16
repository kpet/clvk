#!/bin/bash

set -xe

# Check for a command-line argument
if [ $# -eq 0 ]; then
  echo "Error: Please provide a base path as a command-line argument."
  exit 1
fi

function clean() {
    rm -f "${base_path}/conf_test.conf"
    rm -f "${base_path}/clvk.conf"
}
trap clean EXIT

# Assign the base path from the argument
base_path=$1

# Create the first file with the base path
echo "cache_dir=testing" > "${base_path}/conf_test.conf"

# Create the second file with the base path
echo "cache_dir=failed" > "clvk.conf"
echo "compiler_temp_dir=not/overwritten/" > "clvk.conf"


export CLVK_CONFIG_FILE="${base_path}/conf_test.conf"

# Run test
"${base_path}/config_test"
