#!/usr/bin/env bash

set -ex

cd external/clspv
./utils/fetch_sources.py --deps clang llvm
cd ../..

if [ "${TRAVIS_OS_NAME}" = "osx" ]; then
    cd external/OpenCL-Headers/
    ln -s CL OpenCL
    cd ../..
fi
