# clvk [![CI badge](https://github.com/kpet/clvk/actions/workflows/presubmit.yml/badge.svg?branch=main)](https://github.com/kpet/clvk/actions/workflows/presubmit.yml?query=branch%3Amain++) [![Discord Shield](https://discordapp.com/api/guilds/1002628585250631681/widget.png?style=shield)](https://discord.gg/xsVdjmhFM9)

clvk is a [conformant](https://www.khronos.org/conformance/adopters/conformant-products/opencl)
implementation of OpenCL 3.0 on top of Vulkan using
[clspv](https://github.com/google/clspv) as the compiler.

![OpenCL Logo](./docs/opencl-light.svg#gh-light-mode-only)
![OpenCL Logo](./docs/opencl-dark.svg#gh-dark-mode-only)

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

## Build options

The build system allows a number of things to be configured.

### Vulkan implementation

You can select the Vulkan implementation that clvk will target with the
`CLVK_VULKAN_IMPLEMENTATION` build system option. Two options are currently
supported:

* `-DCLVK_VULKAN_IMPLEMENTATION=system` instructs the build system to use the
  Vulkan implementation provided by your system as detected by CMake's
  `find_package(Vulkan)`. This is the default. You can use the
  [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) to provide headers and
  libraries to build against.

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

### Sanitizers

Support for [sanitizers](https://github.com/google/sanitizers) is integrated into
the build system:

* `CLVK_ENABLE_ASAN` can be used to enable
   [AddressSanitizer](https://clang.llvm.org/docs/AddressSanitizer.html).
* `CLVK_ENABLE_TSAN` can be used to enable
   [ThreadSanitizer](https://clang.llvm.org/docs/ThreadSanitizer.html).
* `CLVK_ENABLE_UBSAN` can be used to enable
   [UndefinedBehaviorSanitizer](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html).

## Building libclc manually

When compiling clvk on small machine (such as Raspberry pi) building libclc
manually can help. Here are the few steps to do it:

1. Build a host native clang compiler using the source pointed by clspv in `<clvk>/external/clspv/third_party/llvm`:
```
cmake -B <clang_host> -S <clvk>/external/clspv/third_party/llvm \
  -DLLVM_ENABLE_PORJECTS="clang" \
  -DLLVM_NATIVE_TARGET=1 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="<clang_host>/install"
cmake --build <clang_host> --target install
```
2. Build libclc using that compiler:
```
cmake -B <libclc> -S <clvk>/external/clspv/third_party/llvm/libclc \
  -DLLVM_CMAKE_DIR="<clang_host>/lib/cmake" \
  -DLIBCLC_DIR_TARGETS_TO_BUILD="clspv--;clspv64--"
cmake --build <libclc>
```
3. Pass the following options to CMake when compiling clvk:
  - `-DCLSPV_EXTERNAL_LIBCLC_DIR="<libclc>"`

## Building for Android

clvk can be built for Android using the
[Android NDK](https://developer.android.com/ndk) toolchain.

1. Download and extract the NDK toolchain to a directory (`/path/to/ndk`)
2. Pass the following options to CMake:
    - `-DCMAKE_TOOLCHAIN_FILE=/path/to/ndk/build/cmake/android.toolchain.cmake`
    - `-DANDROID_ABI=<ABI_FOR_THE_TARGET_DEVICE>`, most likely `arm64-v8a`
    - `-DVulkan_LIBRARY=/path/to/ndk/**/<api-level>/libvulkan.so`
3. That should be it!

When compiling clvk on a small machine (such as Raspberry pi), building libclc
might need to be done as a separate step between 1. and 2.
([Building libclc manually](#building-libclc-manually)).

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
(see [Configuration](#configuration)).

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
> -- https://github.com/google/perfetto/tree/v46.0#perfetto---system-profiling-app-tracing-and-trace-analysis

Perfetto can be enabled by passing the following options to CMake:
   - `-DCLVK_PERFETTO_ENABLE=ON`
   - `-DCLVK_PERFETTO_SDK_DIR=/path/to/perfetto/sdk`

The perfetto SDK can be found in the [Perfetto Github repository](https://github.com/google/perfetto/tree/v46.0)

If you already have a perfetto library in your system you should provide the
following option to CMake:
   - `-DCLVK_PERFETTO_LIBRARY=<your_perfetto_library_name>`
And for the headers, you need to define either:
   - `CLVK_PERFETTO_SDK_DIR` to point to perfetto's SDK (to find `perfetto.h`)
   - `CLVK_PERFETTO_INCLUDE_DIR` to point to perfetto's include directory: `<perfetto>/include` (to find `<perfetto>/include/perfetto/tracing.h`)

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

### Raspberry Pi

Make sure you have an up-to-date Mesa installed on your system.
At the time of writing (May 2023) RaspberryPi OS (Debian 11) did not, but Ubuntu
23.04 does have a compatible vulkan driver.

Install the prerequisites:
```
$ sudo apt install mesa-vulkan-drivers vulkan-tools libvulkan-dev git cmake clang clinfo
```

Check if your vulkan implementation has the `VK_KHR_storage_buffer_storage_class`
extension or supports Vulkan 1.1. Note that it's not enough if this is the case for
llvmpipe. You need v3dv to support this too. If it does not, your Mesa is too old.

To fetch the dependencies, do `python3 ./external/clspv/utils/fetch_sources.py`
because the python interpreter may not be found if not explicitly called as python3.

Building will take many hours on a rPi4.
Maybe you can skip some tool building, but building with default settings works, at least.

Once the libOpenCL.so library has been built, verify with:
```
LD_LIBRARY_PATH=/path/to/build clinfo
```

### With global timing information

Global timing information about API functions as well as some internal functions can be logged at the end of the execution.

To enable it, pass the following option to CMake:
  - `-DCLVK_ENABLE_TIMING=ON`

Here is an example of what to expect running `simple_test`:

```
[CLVK] 0.00 ms -> clReleaseContext (1 blocks, avg 0.001 ms)
[CLVK] 0.02 ms -> clReleaseProgram (1 blocks, avg 0.024 ms)
[CLVK] 0.01 ms -> clReleaseKernel (1 blocks, avg 0.008 ms)
[CLVK] 0.00 ms -> clReleaseCommandQueue (1 blocks, avg 0.005 ms)
[CLVK] 0.01 ms -> clReleaseMemObject (1 blocks, avg 0.007 ms)
[CLVK] 0.00 ms -> clEnqueueUnmapMemObject (1 blocks, avg 0.002 ms)
[CLVK] 0.03 ms -> clEnqueueMapBuffer (1 blocks, avg 0.034 ms)
[CLVK] 5.04 ms -> vkQueueWaitIdle (1 blocks, avg 5.043 ms)
[CLVK] 5.13 ms -> executor_wait (1 blocks, avg 5.126 ms)
[CLVK] 0.02 ms -> vkQueueSubmit (1 blocks, avg 0.022 ms)
[CLVK] 5.07 ms -> execute_cmd: CLVK_COMMAND_BATCH (3 blocks, avg 1.690 ms)
[CLVK] 5.09 ms -> execute_cmds (3 blocks, avg 1.698 ms)
[CLVK] 0.00 ms -> extract_cmds_required_by (3 blocks, avg 0.001 ms)
[CLVK] 0.00 ms -> enqueue_command (3 blocks, avg 0.001 ms)
[CLVK] 0.00 ms -> end_current_command_batch (1 blocks, avg 0.002 ms)
[CLVK] 0.12 ms -> flush_no_lock (4 blocks, avg 0.030 ms)
[CLVK] 5.20 ms -> clFinish (2 blocks, avg 2.602 ms)
[CLVK] 97.06 ms -> clEnqueueNDRangeKernel (1 blocks, avg 97.063 ms)
[CLVK] 0.00 ms -> clSetKernelArg (1 blocks, avg 0.002 ms)
[CLVK] 0.01 ms -> clCreateBuffer (1 blocks, avg 0.012 ms)
[CLVK] 0.01 ms -> clCreateCommandQueue (1 blocks, avg 0.009 ms)
[CLVK] 0.19 ms -> clCreateKernel (1 blocks, avg 0.187 ms)
[CLVK] 237.71 ms -> clBuildProgram (1 blocks, avg 237.713 ms)
[CLVK] 0.02 ms -> clCreateProgramWithSource (1 blocks, avg 0.016 ms)
[CLVK] 0.00 ms -> clCreateContext (1 blocks, avg 0.000 ms)
[CLVK] 0.00 ms -> clGetDeviceInfo (1 blocks, avg 0.001 ms)
[CLVK] 0.00 ms -> clGetDeviceIDs (1 blocks, avg 0.001 ms)
[CLVK] 0.00 ms -> clGetPlatformInfo (1 blocks, avg 0.002 ms)
[CLVK] 0.00 ms -> clGetPlatformIDs (1 blocks, avg 0.000 ms)
```

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


# Configuration

Many aspects of clvk's behaviour can be configured using configuration files
and/or environment variables. clvk attempts to get its configuration from the
following sources (in the order documented here). Values obtained from each
source take precedence over previously obtained values.

1. System-wide configuration in `/etc/clvk.conf`
2. Configuration file in `/usr/local/etc/clvk.conf`
3. Per-user configuration in `~/.config/clvk.conf`
4. `clvk.conf` in the current directory
5. An additional configuration file specified using the `CLVK_CONFIG_FILE`
  environment variable, if provided
6. Environment variables for individual configuration options

Configuration files use a key-value format and allow comments beginning with `#`:

```
# Here's a comment
option = value

other_option = 42
```

Options names are lowercase (e.g `myoption`) in configuration files
but uppercase and prefixed with `CLVK_` in environment variables
(e.g. `CLVK_MYOPTION`).

The list of options are available (with their documentation) in
[src/config.def](https://github.com/kpet/clvk/blob/main/src/config.def).

There are 3 kinds of options:
- `OPTION`: Standard option
- `EARLY_OPTION`: Option which are parsed before the rest to allow enabling
  logging to help debug early part of clvk.
- `PROPERTY`: Option which default value can be overriden in
  [src/device_properties.cpp](https://github.com/kpet/clvk/blob/main/src/device_properties.cpp)

Each option has 3 arguments: `OPTION(type, name, default_value)`

# Limitations

* Only one device per CL context
* No support for out-of-order queues
* No support for device partitioning
* No support for native kernels
* All the limitations implied by the use of clspv
* ... and problably others
