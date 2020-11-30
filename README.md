[![Azure Pipelines Status](https://dev.azure.com/kpet/clvk/_apis/build/status/kpet.clvk?branchName=master)](https://dev.azure.com/kpet/clvk/_build/latest?definitionId=1&branchName=master)

# What is this?

This project is a prototype implementation of OpenCL 1.2 on top of Vulkan using
[clspv](https://github.com/google/clspv) as the compiler.

# Limitations

* Only one device per CL context
* No support for images with a `host_ptr`
* No support for out-of-order queues
* No support for device partitioning
* No support for native kernels
* All the limitations implied by the use of clspv
* ... and problably others

# Supported applications

[Full list](docs/supported-applications.md)

# Getting dependencies

clvk depends on the following external projects:

* [clspv](https://github.com/google/clspv) and its dependencies
* [OpenCL-Headers](https://github.com/KhronosGroup/OpenCL-Headers)
* [SPIRV-Headers](https://github.com/KhronosGroup/SPIRV-Headers)
* [SPIRV-Tools](https://github.com/KhronosGroup/SPIRV-Tools)
* [SPIRV-LLVM-Translator](https://github.com/KhronosGroup/SPIRV-LLVM-Translator)

clvk also (obviously) depends on a Vulkan implementation. The build system
supports a number of options there (see [Building section](#building)).

To fetch all the dependencies needed to build and run clvk, please run:

```
git submodule update --init --recursive
./external/clspv/utils/fetch_sources.py --deps llvm
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
  The path to Talvos's sources must be provided by the user using
  `CLVK_TALVOS_DIR`.

* `-DCLVK_VULKAN_IMPLEMENTATION=loader` enables building against a copy of the
  Vulkan Loader sources provided by the user using `CLVK_VULKAN_LOADER_DIR`.
  The configuration of the loader to target Vulkan ICDs is left to the user.

* `-DCLVK_VULKAN_IMPLEMENTATION=custom` instructs the build system to use the
  values provided by the user manually using `-DVulkan_INCLUDE_DIRS` and
  `-DVulkan_LIBRARIES`.

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

You can build clvk using an external Clspv source tree by setting
`-DCLSPV_SOURCE_DIR=/path/to/clspv/source/`.

# Using

## Via the OpenCL ICD Loader

clvk supports the `cl_khr_icd` OpenCL extension that makes it possible
to use the [OpenCL ICD Loader](https://github.com/KhronosGroup/OpenCL-ICD-Loader).

## Directly

To use clvk to run an OpenCL application, you just need to make sure that the
clvk shared library is picked up by the dynamic linker.

When clspv is not built into the shared library (which is currently the default),
you also need to make sure that clvk has access to the `clspv` binary. If you
wish to move the built library and `clspv` binary out of the build tree, you will
need to make sure that you provide clvk with a path to the `clspv` binary via the
`CLVK_CLSPV_BIN` environment variable
(see [Environment variables](#environment-variables)).

### Unix-like systems (Linux, macOS)

The following ought to work on Unix-like systems:

```
$ LD_LIBRARY_PATH=/path/to/build /path/to/application

# Running the included simple test
$ LD_LIBRARY_PATH=./build ./build/simple_test
```

### Windows

Copy `OpenCL.dll` into a system location or alongside the application executable
you want to run with clvk.

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
   * 1: enabled (default when the output is a terminal)

* `CLVK_LOG_DEST` controls where the logging output goes

   * `stderr`: logging goes to the standard error (default)
   * `stdout`: logging goes to the standard output
   * `file:<fname>`: logging goes to `<fname>`. The file will be created if it
     does not exist and will be truncated.

* `CLVK_CLSPV_BIN` to provide a path to the clspv binary to use

* `CLVK_LLVMSPIRV_BIN` to provide a path to the llvm-spirv binary to use

* `CLVK_VALIDATION_LAYERS` allows to enable Vulkan validation layers

   * 0: disabled (default)
   * 1: enabled

* `CLVK_CLSPV_OPTIONS` to provide additional options to pass to clspv

* `CLVK_QUEUE_PROFILING_USE_TIMESTAMP_QUERIES` to use timestamp queries to
  measure the `CL_PROFILING_COMMAND_{START,END}` profiling infos on devices
  that do not support `VK_EXT_calibrated_timestamps`.

   * 0: disabled (default)
   * 1: enabled

  WARNING: the values will not use the same time base as that used for
  `CL_PROFILING_COMMAND_{QUEUED,SUBMIT}` but this allows to get
  closer-to-the-execution timestamps.

* `CLVK_SKIP_SPIRV_CAPABILITY_CHECK` to avoid checking whether the Vulkan device
  supports all of the SPIR-V capabilities declared by each SPIR-V module.

* `CLVK_MAX_BATCH_SIZE` to control the maximum number of commands that can be
  recorded in a single command buffer.

* `CLVK_KEEP_TEMPORARIES` to keep temporary files created during program build,
  compilation and link operations.

   * 0: disabled (default)
   * 1: enabled

* `CLVK_CACHE_DIR` specifies a directory used for caching compiled program data
  between applications runs. The user is responsible for ensuring that this
  directory is not used concurrently by more than one application.
