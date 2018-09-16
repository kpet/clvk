
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
* Let others know whether your favourite application works or not, send a PR :)

TODO move to a separate doc file and list the status for all applications that
     have been tested

# Getting dependencies

```
git submodule update --init --recursive
cd external/clspv
./utils/fetch_sources.py
cd ../..
```

# Building

```
mkdir -p build
cd build
cmake ../
make -jN
```

# Using

```
LD_LIBRARY_PATH=/path/to/build /path/to/application

# Running the included simple test
LD_LIBRARY_PATH=./build ./build/simple_test
```

# Environment variables

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

