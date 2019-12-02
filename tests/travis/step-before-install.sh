#!/usr/bin/env bash

set -ex

if [ "$TRAVIS_OS_NAME" = "linux" ]; then
    sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
    sudo add-apt-repository -y 'deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial main'
    wget -O - http://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
    sudo apt-get update -qq
fi
