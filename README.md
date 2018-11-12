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

# Supported applications

[Full list](doc/supported-applications.md)

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
`CLVK_VULKAN_IMPLEMENTATION` build system option. Two options are currently
supported:

* `-DCLVK_VULKAN_IMPLEMENTATION=system` instructs the build system to use the
  Vulkan implementation provided by your system as detected by CMake. This
  is the default.

* `-DCLVK_VULKAN_IMPLEMENTATION=talvos` instructs the build system to use
  [Talvos](https://github.com/talvos/talvos). Talvos emulates the
  Vulkan API and provides an interpreter for SPIR-V modules. You don't
  need Vulkan-compatible hardware and drivers to run clvk using Talvos.

### Tests

It is possible to disable the build of the tests by passing
`-DCLVK_BUILD_TESTS=OFF`.

### OpenCL conformance tests

Passing `-DCLVK_BUILD_CONFORMANCE_TESTS=ON` will instruct CMake to build the
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
to the `clspv` binary via the `CLVK_CLSPV_BIN` environment variable
(see [Environment variables](#environment-variables)).

# Environment variables

The behaviour of a few things in clvk can be controlled by environment
variables. Here's a quick guide:

* `CLVK_LOG` controls the level of logging

   * 0: only print fatal messages (default)
   * 1: print errors as well
   * 2: print warnings as well
   * 3: print information messages as well
   * 4: print all debug messages

* `CLVK_LOG_COLOUR` controls colour logging

   * 0: disabled
   * 1: enabled (default)

* `CLVK_CLSPV_BIN` to provide a path to the clspv binary to use

* `CLVK_VALIDATION_LAYERS` allows to enable Vulkan validation layers

   * 0: disabled (default)
   * 1: enabled

* `CLVK_CLSPV_OPTIONS` to provide additional options to pass to clspv

* `CLVK_QUEUE_PROFILING_USE_TIMESTAMP_QUERIES` to use timestamp queries to
  measure the `CL_PROFILING_COMMAND_{START,END}` profiling infos.
  WARNING: the values will not use the same time base as that used for
  `CL_PROFILING_COMMAND_{QUEUED,SUBMIT}` but this allows to get
  closer-to-the-execution timestamps. The two can be reconciled on devices
  that support `VK_EXT_calibrated_timestamps`
  (see [#110](https://github.com/kpet/clvk/issues/110)).

