#!/bin/bash

# Check for a command-line argument
if [ $# -eq 0 ]; then
  echo "Error: Please provide a base path as a command-line argument."
  exit 1
fi

# Assign the base path from the argument
base_path=$1

# Create the first file with the base path
touch $base_path/conf_test.conf
echo "cache_dir=failed" >> $base_path/conf_test.conf

# Create the second file with the base path
touch $base_path/clvk.conf
echo "cache_dir=testing" >> $base_path/clvk.conf

export CLVK_CONFIG_FILE=$base_path/clvk.conf

# Run test
$base_path/config_test
