#!/usr/bin/env bash

set -ex

if [ "${TRAVIS_OS_NAME}" == "linux" ]; then
    sudo apt update
    sudo apt install ocl-icd-libopencl1
fi

if [ "${TRAVIS_OS_NAME}" == "osx" ]; then
    brew install ccache bash
    export PATH="/usr/local/opt/ccache/libexec:$PATH"
fi

if [ "${TRAVIS_OS_NAME}" == "windows" ]; then
    choco install make
fi

set +ex
