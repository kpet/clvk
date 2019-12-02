#!/usr/bin/env bash

set -ex

if [ "${TRAVIS_OS_NAME}" == "linux" ]; then
    sudo apt-get install -qq g++-7 clang-format
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 90
fi

if [ "${TRAVIS_OS_NAME}" == "osx" ]; then
    brew install ccache bash
    export PATH="/usr/local/opt/ccache/libexec:$PATH"
fi

if [ "${TRAVIS_OS_NAME}" == "windows" ]; then
    choco install make
fi

set +ex
