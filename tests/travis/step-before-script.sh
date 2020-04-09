#!/usr/bin/env bash

set -ex

cd external/clspv
./utils/fetch_sources.py --deps llvm
cd ../..

git clone https://github.com/talvos/talvos.git

if [ "${TRAVIS_OS_NAME}" = "osx" ]; then
    cd external/OpenCL-Headers/
    ln -s CL OpenCL
    cd ../..
fi
