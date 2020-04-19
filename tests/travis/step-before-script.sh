#!/usr/bin/env bash

set -ex

if [ "${JOB_CHECK_FORMAT}" -eq 1 ]; then
    exit 0
fi

cd external/clspv
./utils/fetch_sources.py --deps llvm
cd ../..

git clone https://github.com/talvos/talvos.git

if [ "${TRAVIS_OS_NAME}" = "osx" ]; then
    cd external/OpenCL-Headers/
    ln -s CL OpenCL
    cd ../..
fi
