#!/usr/bin/env bash

set -xe

[[ $# -eq 1 ]] || (echo "missing input trace file" && exit -1)

SCRIPT_DIR="$(dirname $(realpath "${BASH_SOURCE[0]}"))"
TRACE_FILE="$1"
OUTPUT_FILE="${SCRIPT_DIR}/output.txt"

# Either it is in your path, or you need to define the environment variable
TRACE_PROCESSOR_SHELL=${TRACE_PROCESSOR_SHELL:-"trace_processor_shell"}

echo "SELECT name FROM slice WHERE slice.category='clvk'" \
    | "${TRACE_PROCESSOR_SHELL}" -q /dev/stdin "${TRACE_FILE}" \
    | sort \
    | uniq \
          > "${OUTPUT_FILE}"
diff "${OUTPUT_FILE}" "${SCRIPT_DIR}/expectation.txt"

rm "${OUTPUT_FILE}"
