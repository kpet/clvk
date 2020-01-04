#!/usr/bin/env bash

set -ex

if [ "${TRAVIS_OS_NAME}" == "linux" ]; then
    sudo apt-get install -qq clang-format
fi

if [ "${TRAVIS_OS_NAME}" == "osx" ]; then
    brew install ccache bash
    export PATH="/usr/local/opt/ccache/libexec:$PATH"
fi

if [ "${TRAVIS_OS_NAME}" == "windows" ]; then
    choco install make
fi

set +ex
