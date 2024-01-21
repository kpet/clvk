#!/usr/bin/env bash

set -xe


[[ $# -eq 2 ]] || (echo "[ERROR] USAGE: ${BASH_SOURCE[0]} <input_trace> <expectation_file>" && exit -1)

SCRIPT_DIR="$(dirname $(realpath "${BASH_SOURCE[0]}"))"
TRACE_FILE="$1"
EXPECTATION_FILE="$2"
OUTPUT_FILE="${SCRIPT_DIR}/output.txt"
EXPECTATION_SORTED="${EXPECTATION_FILE}.sorted"

# Either it is in your path, or you need to define the environment variable
TRACE_PROCESSOR_SHELL=${TRACE_PROCESSOR_SHELL:-"trace_processor_shell"}

function clean() {
    rm -f "${OUTPUT_FILE}" "${EXPECTATION_SORTED}"
}
trap clean EXIT

echo "SELECT name FROM slice WHERE slice.category='clvk'" \
    | "${TRACE_PROCESSOR_SHELL}" -q /dev/stdin "${TRACE_FILE}" \
    | sort \
    | uniq \
          > "${OUTPUT_FILE}"

# Also sort the expectation to make sure to apply the same sort algorithm to the output and the expectation.
sort "${EXPECTATION_FILE}" > "${EXPECTATION_SORTED}"

diff "${OUTPUT_FILE}" "${EXPECTATION_SORTED}"
