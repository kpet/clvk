#!/usr/bin/env bash

set -xe

[[ $# -eq 1 ]] || (echo "missing input trace file" && exit -1)

SCRIPT_DIR="$(dirname $(realpath "${BASH_SOURCE[0]}"))"
TRACE_FILE="$1"
OUTPUT_FILE="${SCRIPT_DIR}/output.txt"
EXPECTATION_SORTED=${SCRIPT_DIR}/expectation-sorted.txt

# Either it is in your path, or you need to define the environment variable
TRACE_PROCESSOR_SHELL=${TRACE_PROCESSOR_SHELL:-"trace_processor_shell"}

echo "SELECT name FROM slice WHERE slice.category='clvk'" \
    | "${TRACE_PROCESSOR_SHELL}" -q /dev/stdin "${TRACE_FILE}" \
    | sort \
    | uniq \
          > "${OUTPUT_FILE}"

# Also sort the expectation to make sure to apply the same sort algorithm to the output and the expectation.
sort "${SCRIPT_DIR}/expectation.txt" > "${EXPECTATION_SORTED}"

diff "${OUTPUT_FILE}" "${SCRIPT_DIR}/expectation.txt" "{EXPECTATION_SORTED}"

rm "${OUTPUT_FILE}" "${EXPECTATION_SORTED}"
