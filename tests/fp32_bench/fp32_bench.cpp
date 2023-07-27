// Copyright 2023 The clvk authors.
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

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#define CL_TARGET_OPENCL_VERSION 120
#include "CL/cl.h"

#define BUFFER_SIZE 1024 * sizeof(cl_uint)

#define CHECK_CL_ERRCODE(err)                                                  \
    do {                                                                       \
        if (err != CL_SUCCESS) {                                               \
            fprintf(stderr, "%s:%d error after CL call: %d\n", __FILE__,       \
                    __LINE__, err);                                            \
            return EXIT_FAILURE;                                               \
        }                                                                      \
    } while (0)

const char* program_source = R"(
kernel void kernel_float(global float* data) {
	float x = (float)get_global_id(0);
	float y = (float)get_local_id(0);
	for(uint i=0u; i<1000000u; i++) {
		x = fma(y, x, y);
		y = fma(x, y, x);
	}
	data[get_global_id(0)] = y;
}

)";

int main(int argc, char** argv) {

    // Check if the user provided any command line arguments.
    if (argc != 2) {
        std::cerr << "please specify number of iterations as such: " << argv[0]
                  << " <number of iterations> ." << std::endl;
        return 1;
    }

    int num_iterations = atoi(argv[1]);
    cl_device_id device;
    cl_int err;
    cl_uint num_platforms;
    cl_platform_id platforms[10];
    char device_name[128];
    err = clGetPlatformIDs(10, platforms, &num_platforms);
    CHECK_CL_ERRCODE(err);

    // Check if any GPUs were found
    bool found = false;
    if (num_platforms > 0) {
        for (cl_uint i = 0; i < num_platforms; i++) {
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device,
                                 nullptr);
            CHECK_CL_ERRCODE(err);
            if (device) {
                found = true;
                err =
                    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name),
                                    device_name, nullptr);
                CHECK_CL_ERRCODE(err);
                printf("Device name: %s\n", device_name);
                break;
            }
        }
        if (!found) {
            printf("No GPU found! \n");
            return -1;
        }
    } else {
        printf("No devices found! \n");
        return -1;
    }

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
    auto kernel = clCreateKernel(program, "kernel_float", &err);
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

    // Define work group and global size
    size_t work_group_size;
    size_t global_work_size = BUFFER_SIZE / sizeof(cl_uint);
    err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                                   sizeof(work_group_size), &work_group_size,
                                   nullptr);
    CHECK_CL_ERRCODE(err);
    printf("global work size = %ld, local work size = %ld \n", global_work_size,
           work_group_size);
    // Start time for benchmarking
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    start = std::chrono::high_resolution_clock::now();

    // Execute the kernel
    for (int exec = 0; exec < num_iterations; exec++) {
        err =
            clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size,
                                   &work_group_size, 0, nullptr, nullptr);
        CHECK_CL_ERRCODE(err);

        // Complete execution
        err = clFinish(queue);
        CHECK_CL_ERRCODE(err);
    }
    end = std::chrono::high_resolution_clock::now();
    CHECK_CL_ERRCODE(err);

    err = clFinish(queue);
    CHECK_CL_ERRCODE(err);

    // Cleanup
    clReleaseMemObject(buffer);
    clReleaseCommandQueue(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);

    // Report time.
    std::chrono::duration<double> kernel_time = end - start;
    printf("Kernel execution time: %f seconds for %d iterations \n ", kernel_time.count(), num_iterations);

    return 0;
}
