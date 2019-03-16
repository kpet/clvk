[![Build Status](https://travis-ci.com/kpet/clvk.svg?branch=master)](https://travis-ci.com/kpet/clvk)

# What is this?

This project is a prototype implementation of OpenCL 1.2 on top of Vulkan using
[clspv](https://github.com/google/clspv) as the compiler.

# Disclaimer

This experimental piece of software has been developed as a hobby project
by a single person.  This is *not production quality code*.  If your whole
computer crashes and burns, don't blame me! You've been warned.

# Limitations

* Only one device per CL context
* No support for images
* No support for out-of-order queues
* No support for device partitioning
* No support for native kernels
* All the limitations implied by the use of clspv
* ... and problably others

# Applications known to run

The following applications work at least partially:

* [clinfo](https://github.com/Oblomov/clinfo)
* [OpenCL conformance tests](https://github.com/KhronosGroup/OpenCL-CTS)
* [SHOC](https://github.com/vetter/shoc)
* [OCLToys](https://github.com/ignatenkobrain/ocltoys.git)
* Let others know whether your favourite application works or not, send a PR :)

TODO move to a separate doc file and list the status for all applications that
     have been tested

# Getting dependencies

clvk depends on the following external projects:

* [clspv](https://github.com/google/clspv) and its dependencies
* [OpenCL-Headers](https://github.com/KhronosGroup/OpenCL-Headers)
* [SPIRV-Headers](https://github.com/KhronosGroup/SPIRV-Headers)
* [SPIRV-Tools](https://github.com/KhronosGroup/SPIRV-Tools)

clvk also (obviously) depends on a Vulkan implementation. The build system
supports a number of options there (see [Building section](#building)).

To fetch all the dependencies needed to build and run clvk, please run:

```
git submodule update --init --recursive
./external/clspv/utils/fetch_sources.py --deps clang llvm
```

# Building

clvk uses CMake for its build system.

## Getting started

To build with the default configuration options, just use following:

```
mkdir -p build
cd build
cmake ../
make -j$(nproc)
```

## Other options

The build system allows a number of things to be configured.

### Vulkan implementation

You can select the Vulkan implementation that clvk will target with the
`VULKAN_IMPLEMENTATION` build system option. Two options are currently
supported:

* `-DVULKAN_IMPLEMENTATION=system` instructs the build system to use the
  Vulkan implementation provided by your system as detected by CMake. This
  is the default.

* `-DVULKAN_IMPLEMENTATION=talvos` instructs the build system to use
  [Talvos](https://github.com/talvos/talvos). Talvos emulates the
  Vulkan API and provides an interpreter for SPIR-V modules. You don't
  need Vulkan-compatible hardware and drivers to run clvk using Talvos.

### OpenCL conformance tests

Passing `-DBUILD_CONFORMANCE_TESTS=ON` will instruct CMake to build the
[OpenCL conformance tests](https://github.com/KhronosGroup/OpenCL-CTS).
This is _not expected to work out-of-the box_ at the moment.

### Clspv compilation

You can select the compilation style that clvk will use with Clspv via
the `CLVK_CLSPV_ONLINE_COMPILER` option. By default, Clspv is run in a
separate process.

* `-DCLVK_CLSPV_ONLINE_COMPILER=1` will cause clvk to compile kernels
in the same process via the Clspv C++ API.

# Using

To use clvk to run an OpenCL application, you just need to make sure
that the clvk libOpenCL.so shared library is picked up by the dynamic
linker and that clvk has access to the `clspv` binary. The following
ought to work on most Unix-like systems:

```
$ LD_LIBRARY_PATH=/path/to/build /path/to/application

# Running the included simple test
$ LD_LIBRARY_PATH=./build ./build/simple_test
```

If you wish to move the built library and `clspv` binary out of the build
tree, you will need to make sure that you provide clvk with a path
to the `clspv` binary via the `CVK_CLSPV_BIN` environment variable
(see [Environment variables](#environment-variables)).

# Environment variables

The behaviour of a few things in clvk can be controlled by environment
variables. Here's a quick guide:

* `CVK_LOG` controls the level of logging

   * 0: only print fatal messages (default)
   * 1: print errors as well
   * 2: print warnings as well
   * 3: print information messages as well
   * 4: print all debug messages

* `CVK_LOG_COLOUR` controls colour logging

   * 0: disabled
   * 1: enabled (default)

* `CVK_CLSPV_BIN` to provide a path to the clspv binary to use

* `CVK_VALIDATION_LAYERS` allows to enable Vulkan validation layers

   * 0: disabled (default)
   * 1: enabled

