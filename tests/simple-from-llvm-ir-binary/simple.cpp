// Copyright 2022 The clvk authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#define CL_TARGET_OPENCL_VERSION 120
#include "CL/cl.h"

#define BUFFER_SIZE 1024

#define CHECK_CL_ERRCODE(err)                                                  \
    do {                                                                       \
        if (err != CL_SUCCESS) {                                               \
            fprintf(stderr, "%s:%d error after CL call: %d\n", __FILE__,       \
                    __LINE__, err);                                            \
            return EXIT_FAILURE;                                               \
        }                                                                      \
    } while (0)

int main(int argc, char* argv[]) {
    cl_platform_id platform;
    cl_device_id device;
    cl_int err;

    // Get the first GPU device of the first platform
    err = clGetPlatformIDs(1, &platform, nullptr);
    CHECK_CL_ERRCODE(err);

    char platform_name[128];
    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name),
                            platform_name, nullptr);
    CHECK_CL_ERRCODE(err);

    printf("Platform: %s\n", platform_name);

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, nullptr);
    CHECK_CL_ERRCODE(err);

    char device_name[128];
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name),
                          device_name, nullptr);
    CHECK_CL_ERRCODE(err);

    printf("Device: %s\n", device_name);

    auto context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CHECK_CL_ERRCODE(err);

    // Load program from file
    std::ifstream ifile;
    const char* fname = "simple.bc";
    ifile.open(fname, std::ios::in | std::ios::binary);

    if (!ifile.is_open()) {
        fprintf(stderr, "Failed to open file %s\n", fname);
        return EXIT_FAILURE;
    }

    ifile.seekg(0, std::ios::end);
    uint32_t size = ifile.tellg();
    ifile.seekg(0, std::ios::beg);

    std::vector<char> code;
    code.assign(size, 0);

    ifile.read(reinterpret_cast<char*>(code.data()), size);
    if (!ifile.good()) {
        fprintf(stderr, "Failed to read LLVM IR module from %s\n", fname);
        return EXIT_FAILURE;
    }
    printf("Loaded LLVM IR data from %s, size = %u\n", fname, size);

    // Create program
    size_t program_size = size;
    const unsigned char* program_code =
        reinterpret_cast<const unsigned char*>(code.data());
    auto program = clCreateProgramWithBinary(context, 1, &device, &program_size,
                                             &program_code, nullptr, &err);
    CHECK_CL_ERRCODE(err);

    // Build program
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    CHECK_CL_ERRCODE(err);

    // Create kernel
    auto kernel = clCreateKernel(program, "test_simple", &err);
    CHECK_CL_ERRCODE(err);

    // Create command queue
    auto queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_CL_ERRCODE(err);

    // Create buffer
    auto buffer =
        clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                       BUFFER_SIZE, nullptr, &err);
    CHECK_CL_ERRCODE(err);

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
    CHECK_CL_ERRCODE(err);

    size_t gws = BUFFER_SIZE / sizeof(cl_int);
    size_t lws = 2;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &gws, &lws, 0,
                                 nullptr, nullptr);
    CHECK_CL_ERRCODE(err);

    // Complete execution
    err = clFinish(queue);
    CHECK_CL_ERRCODE(err);

    // Map the buffer
    auto ptr = clEnqueueMapBuffer(queue, buffer, CL_TRUE, CL_MAP_READ, 0,
                                  BUFFER_SIZE, 0, nullptr, nullptr, &err);
    CHECK_CL_ERRCODE(err);

    // Check the expected result
    bool success = true;
    auto buffer_data = static_cast<cl_uint*>(ptr);
    for (cl_uint i = 0; i < BUFFER_SIZE / sizeof(cl_uint); ++i) {
        if (buffer_data[i] != static_cast<cl_uint>(i)) {
            printf("Failed comparison at buffer_data[%u]: expected %u but got "
                   "%u\n",
                   i, i, buffer_data[i]);
            success = false;
        }
    }

    // Unmap the buffer
    err = clEnqueueUnmapMemObject(queue, buffer, ptr, 0, nullptr, nullptr);
    CHECK_CL_ERRCODE(err);
    err = clFinish(queue);
    CHECK_CL_ERRCODE(err);

    // Cleanup
    clReleaseMemObject(buffer);
    clReleaseCommandQueue(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);

    // Report status
    if (success) {
        printf("Buffer content verified, test passed.\n");
        return EXIT_SUCCESS;
    } else {
        printf("Test failed.\n");
        return EXIT_FAILURE;
    }
}
