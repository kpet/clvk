#!/usr/bin/env bash

set -ex

if [ "$TRAVIS_OS_NAME" = "linux" ]; then
    sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
    sudo apt-get update -qq
fi
