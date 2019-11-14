#!/usr/bin/env bash

set -ex

if [ "${TRAVIS_OS_NAME}" != "windows" ]; then
    ccache --max-size=2G
    ccache --show-stats
fi

declare -A CMD_TIMEOUT_MAP
CMD_TIMEOUT_MAP[osx]=gtimeout
CMD_TIMEOUT_MAP[linux]=timeout
CMD_TIMEOUT="${CMD_TIMEOUT_MAP[${TRAVIS_OS_NAME}]}"

declare -A BUILD_TIMEOUT_MAP
BUILD_TIMEOUT_MAP[osx]=2000
BUILD_TIMEOUT_MAP[linux]=2400
BUILD_TIMEOUT="${BUILD_TIMEOUT_MAP[${TRAVIS_OS_NAME}]}"

BUILD_DIR=build

mkdir "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake --version
cmake -DCMAKE_BUILD_TYPE=Release -DCLVK_VULKAN_IMPLEMENTATION=talvos -DSPIRV_WERROR=OFF -DLLVM_TARGETS_TO_BUILD=ARM ${JOB_CMAKE_CONFIG} ..

set +e
${CMD_TIMEOUT} ${BUILD_TIMEOUT} make -j2
BUILD_STATUS=$?
set -e

if [ "${TRAVIS_OS_NAME}" != "windows" ]; then
    ccache --show-stats
fi

if [ ${BUILD_STATUS} -ne 0 ]; then
    exit ${BUILD_STATUS}
fi

./simple_test
./simple_test_static
./api_tests
