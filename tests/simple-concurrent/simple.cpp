// Copyright 2018 The clvk authors.
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

#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

#define CL_TARGET_OPENCL_VERSION 120
#include "CL/cl.h"

#define CHECK_CL_ERRCODE(err)                                                  \
    do {                                                                       \
        if (err != CL_SUCCESS) {                                               \
            fprintf(stderr, "%s:%d error after CL call: %d\n", __FILE__,       \
                    __LINE__, err);                                            \
            return EXIT_FAILURE;                                               \
        }                                                                      \
    } while (0)

const char* program_source = R"(
kernel void test_simple(uint timeout)
{
   printf("Hello World! %u\n", timeout);
   while (timeout--);
}
)";

int main(int argc, char** argv) {
    cl_platform_id platform;
    cl_device_id device;
    cl_int err;

    assert(argc == 2);
    uint32_t sleep = atoi(argv[1]);

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

    // Create program
    auto program =
        clCreateProgramWithSource(context, 1, &program_source, nullptr, &err);
    CHECK_CL_ERRCODE(err);

    // Build program
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    CHECK_CL_ERRCODE(err);

    // Create kernel
    auto kernel = clCreateKernel(program, "test_simple", &err);
    CHECK_CL_ERRCODE(err);
    auto kernel2 = clCreateKernel(program, "test_simple", &err);
    CHECK_CL_ERRCODE(err);

    // Create command queue
    auto queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_CL_ERRCODE(err);

    // Set kernel arguments
    cl_uint timeout = 10000;
    err = clSetKernelArg(kernel, 0, sizeof(cl_uint), &timeout);
    CHECK_CL_ERRCODE(err);

    size_t gws = 1;
    size_t lws = 1;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &gws, &lws, 0,
                                 nullptr, nullptr);
    CHECK_CL_ERRCODE(err);

    timeout = 0;
    err = clSetKernelArg(kernel2, 0, sizeof(cl_uint), &timeout);
    CHECK_CL_ERRCODE(err);

    err = clFlush(queue);
    CHECK_CL_ERRCODE(err);

    usleep(sleep);

    err = clEnqueueNDRangeKernel(queue, kernel2, 1, nullptr, &gws, &lws, 0,
                                 nullptr, nullptr);
    CHECK_CL_ERRCODE(err);

    // Complete execution
    err = clFinish(queue);
    CHECK_CL_ERRCODE(err);

    // Cleanup
    clReleaseCommandQueue(queue);
    clReleaseKernel(kernel);
    clReleaseKernel(kernel2);
    clReleaseProgram(program);
    clReleaseContext(context);

    return 0;
}
