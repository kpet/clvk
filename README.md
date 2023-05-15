# clvk [![CI badge](https://github.com/kpet/clvk/actions/workflows/presubmit.yml/badge.svg?branch=main)](https://github.com/kpet/clvk/actions/workflows/presubmit.yml?query=branch%3Amain++) [![Discord Shield](https://discordapp.com/api/guilds/1002628585250631681/widget.png?style=shield)](https://discord.gg/xsVdjmhFM9)

clvk is a prototype implementation of OpenCL 3.0 on top of Vulkan using
[clspv](https://github.com/google/clspv) as the compiler.

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

* `-DCLVK_VULKAN_IMPLEMENTATION=loader` enables building against a copy of the
  [Vulkan Loader](https://github.com/KhronosGroup/Vulkan-Loader) sources
  provided by the user using `CLVK_VULKAN_LOADER_DIR`.
  The configuration of the loader to target Vulkan ICDs is left to the user.

* `-DCLVK_VULKAN_IMPLEMENTATION=custom` instructs the build system to use the
  values provided by the user manually using `-DVulkan_INCLUDE_DIRS` and
  `-DVulkan_LIBRARIES`.

### Tests

It is possible to disable the build of the tests by passing
`-DCLVK_BUILD_TESTS=OFF`.

It is also possible to disable only the build of the tests linking with the
static OpenCL library by passing `-DCLVK_BUILD_STATIC_TESTS=OFF`.

By default, tests needing `gtest` are linked with the libraries coming from
llvm (through clspv).
It is possible to use other libraries by passing
`-DCLVK_GTEST_LIBRARIES=<lib1>;<lib2>` (semicolumn separated list).

### Assertions

Assertions can be controlled with the `CLVK_ENABLE_ASSERTIONS` build option.
They are enabled by default in Debug builds and disabled in other build types.

### OpenCL conformance tests

Passing `-DCLVK_BUILD_CONFORMANCE_TESTS=ON` will instruct CMake to build the
[OpenCL conformance tests](https://github.com/KhronosGroup/OpenCL-CTS).
This is _not expected to work out-of-the box_ at the moment.

It is also possible to build GL and GLES interroperability tests by passing
`-DCLVK_BUILD_CONFORMANCE_TESTS_GL_GLES_SUPPORTED=ON`.

### Clspv compilation

You can select the compilation style that clvk will use with Clspv via
the `CLVK_CLSPV_ONLINE_COMPILER` option. By default, Clspv is run in a
separate process.

* `-DCLVK_CLSPV_ONLINE_COMPILER=1` will cause clvk to compile kernels
in the same process via the Clspv C++ API.

You can build clvk using an external Clspv source tree by setting
`-DCLSPV_SOURCE_DIR=/path/to/clspv/source/`.

### SPIRV components

All needed SPIRV components are added to `clvk` using git submodules.
It is possible to disable the build of those component or to reuse already
existing sources:

#### SPIRV-Headers

`SPIRV_HEADERS_SOURCE_DIR` can be overriden to use another `SPIRV-Headers`
repository.

#### SPIRV-Tools

`SPIRV_TOOLS_SOURCE_DIR` can be overriden to use another `SPIRV-Tools`
repository.
You can also disable the build of `SPIRV-Tools` by setting
`-DCLVK_BUILD_SPIRV_TOOLS=OFF`.

#### SPIRV-LLVM-Translator

`LLVM_SPIRV_SOURCE` can be overriden to use another `SPIRV-LLVM-Translator`
repository.
Note that it is not used if the compiler support is disabled (enabled by
default).

## Building for Android

clvk can be built for Android using the
[Android NDK](https://developer.android.com/ndk) toolchain.

1. Download and extract the NDK toolchain to a directory (`/path/to/ndk`)
2. Pass the following options to CMake:
    - `-DCMAKE_TOOLCHAIN_FILE=/path/to/ndk/build/cmake/android.toolchain.cmake`
    - `-DANDROID_ABI=<ABI_FOR_THE_TARGET_DEVICE>`, most likely `arm64-v8a`
    - `-DCLVK_VULKAN_IMPLEMENTATION=loader`
    - `-DCLVK_VULKAN_LOADER_DIR=/path/to/Vulkan-Loader` (see above)
3. That should be it!

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
`CLVK_CLSPV_PATH` environment variable
(see [Environment variables](#environment-variables)).

### Unix-like systems (Linux, macOS)

The following ought to work on Unix-like systems:

```
$ LD_LIBRARY_PATH=/path/to/build /path/to/application

# Running the included simple test
$ LD_LIBRARY_PATH=./build ./build/simple_test
```

#### With perfetto traces

> Perfetto is a production-grade open-source stack for performance instrumentation and trace analysis. It offers services and libraries and for recording system-level and app-level traces, native + java heap profiling, a library for analyzing traces using SQL and a web-based UI to visualize and explore multi-GB traces.
>
> -- https://github.com/google/perfetto/tree/v29.0#perfetto---system-profiling-app-tracing-and-trace-analysis

Perfetto can be enabled by passing the following options to CMake:
   - `-DCLVK_PERFETTO_ENABLE=ON`
   - `-DCLVK_PERFETTO_SDK_DIR=/path/to/perfetto/sdk`

The perfetto SDK can be found in the [Perfetto Github repository](https://github.com/google/perfetto/tree/v29.0)

If you already have a perfetto library in your system, you still need to provide the path
to the SDK directory so the build system can find `perfetto.h`.
But you should also provide the following option to CMake:
   - `-DCLVK_PERFETTO_LIBRARY=<your_perfetto_library_name>`

By default, clvk will use Perfetto's `InProcess` backend, which means
that you just have to run your application to generate traces.
Environment variables can be used to control the maximum size of traces and what file they are saved to.

If you'd rather use Perfetto's `System` backend, pass the following option
to CMake:
   - `-DCLVK_PERFETTO_BACKEND=System`

Once traces have been generated, you can view them using the
[perfetto trace viewer](https://ui.perfetto.dev/).

### Windows

Copy `OpenCL.dll` into a system location or alongside the application executable
you want to run with clvk.

## Tuning clvk

clvk can be tuned to improve the performance of specific workloads or on specific platforms. While we try to have the default
parameters set at their best values for each platform, they can be
changed for specific applications. One of the best way to know whether something
can be improved is to use traces to understand what should be changed.

### Group size

clvk is grouping commands and waiting for a call to `clFlush` or any
blocking calls (`clFinish`, `clWaitForEvents`, etc.) to submit those groups for execution.

clvk's default group flushing behaviour can be controlled using the following two variables to flush groups as soon as a given number of commands have been grouped:
   - `CLVK_MAX_CMD_GROUP_SIZE`
   - `CLVK_MAX_FIRST_CMD_GROUP_SIZE`


### Batch size

clvk relies on vulkan to offload workoad to the GPU. As such, it is better to
batch OpenCL commands (translated into vulkan commands) into a vulkan command
buffer. But doing that may increase the latency to start running commands.

The size of those batches can be controlled using the following two variables:
   - `CLVK_MAX_CMD_BATCH_SIZE`
   - `CLVK_MAX_FIRST_CMD_BATCH_SIZE`


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

* `CLVK_CLSPV_PATH` to provide a path to the clspv binary to use

* `CLVK_LLVMSPIRV_BIN` to provide a path to the llvm-spirv binary to use

* `CLVK_VALIDATION_LAYERS` allows to enable Vulkan validation layers

   * 0: disabled (default)
   * 1: enabled

* `CLVK_CLSPV_OPTIONS` to provide additional options to pass to clspv

* `CLVK_CLSPV_NATIVE_BUILTINS` comma separated list of builtins that will use
  the native implementation by default

* `CLVK_CLSPV_LIBRARY_BUILTINS` comma separated list of builtins that will be
  forced to use the libclc implementation

* `CLVK_QUEUE_PROFILING_USE_TIMESTAMP_QUERIES` to use timestamp queries to
  measure the `CL_PROFILING_COMMAND_{START,END}` profiling infos on devices
  that do not support `VK_EXT_calibrated_timestamps`.

   * 0: disabled (default)
   * 1: enabled

  WARNING: the values will not use the same time base as that used for
  `CL_PROFILING_COMMAND_{QUEUED,SUBMIT}` but this allows to get
  closer-to-the-execution timestamps.

* `CLVK_SPIRV_VALIDATION` controls SPIR-V validation behaviour.

   * 0: skip validation
   * 1: warn when validation fails
   * 2: fail compilation and report an error when validation fails (default)

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

* `CLVK_COMPLIER_TEMP_DIR` specifies a directory used to create a temporary
  folder to store compiled program data used in a single run. This folder shall
  have write permission (default: current directory).

* `CLVK_MAX_CMD_GROUP_SIZE` specifies the maximum number of commands in a group.
  When a group reaches this number, it is automatically flushed.

* `CLVK_MAX_FIRST_CMD_GROUP_SIZE` specifies the maximum number of commands in a
  group when there is no group to be processed or being processed in the queue.

* `CLVK_MAX_CMD_BATCH_SIZE` specifies the maximum number of commands per batch.
  When this number is reached, the batch will be added to the current group of
  commands, and a new batch will be created.

* `CLVK_MAX_FIRST_CMD_BATCH_SIZE` specifies the maximum number of commands per
  batch when there is no batch to be processed or being processed in the queue.

* `CLVK_PERFETTO_TRACE_MAX_SIZE` specifies the maximum size (in kB) of traces
  generated by Perfetto. It only applies when using Perfetto with the
  `InProcess` backend.

* `CLVK_PERFETTO_TRACE_DEST` specifies the filename to use for the traces
  generated by Perfetto (default: `clvk.perfetto-trace`).

# Limitations

* Only one device per CL context
* No support for out-of-order queues
* No support for device partitioning
* No support for native kernels
* All the limitations implied by the use of clspv
* ... and problably others
