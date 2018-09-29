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

#include <cstdio>
#include <string>

#define CL_TARGET_OPENCL_VERSION 120
#include "CL/cl.h"

#define IMAGE_HEIGHT 128
#define IMAGE_WIDTH 128

#define CHECK_CL_ERRCODE(err) do { \
    if (err != CL_SUCCESS) {       \
        fprintf(stderr, "%s:%d error after CL call: %d\n", __FILE__, __LINE__, err); \
        return EXIT_FAILURE; \
    } \
    } while (0)

const char *program_source = R"(
kernel void write(image2d_t write_only img)
{
    int2 coord = {get_global_id(0), get_global_id(1)};
    float4 color = {1, 2, 3, 4};
    write_imagef(img, coord, color);
}
kernel void copy(image2d_t read_only img, sampler_t sampler, global float4 *buffer)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int2 coord = {x, y};
    float4 color = read_imagef(img, sampler, coord);
    int row_pitch = 128;
    buffer[(y * row_pitch) + x] = color;
}
)";

int main(int argc, char* argv[])
{
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

    // Create program
    auto program = clCreateProgramWithSource(context, 1, &program_source,
                                             nullptr, &err);
    CHECK_CL_ERRCODE(err);

    // Build program
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        CHECK_CL_ERRCODE(err);
        std::string build_log;
        build_log.reserve(log_size);
        auto data_ptr = const_cast<char*>(build_log.c_str());
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, data_ptr, nullptr);
        CHECK_CL_ERRCODE(err);

        printf("Build log:\n%s\n", build_log.c_str());
    }

    // Create kernels
    auto kernel_write = clCreateKernel(program, "write", &err);
    CHECK_CL_ERRCODE(err);
    auto kernel_copy = clCreateKernel(program, "copy", &err);
    CHECK_CL_ERRCODE(err);

    // Create command queue
    auto queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_CL_ERRCODE(err);

    // Create the image
    cl_image_format format = {CL_RGBA, CL_FLOAT};
    cl_image_desc desc = {
        CL_MEM_OBJECT_IMAGE2D, // image_type
        IMAGE_WIDTH, // image_width
        IMAGE_HEIGHT, // image_height
        1,   // image_depth
        1,   // image_array_size
        0,   // image_row_pitch
        0,   // image_slice_pitch
        0,   // num_mip_levels
        0,   // num_samples
        nullptr, // buffer
    };
    auto image = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, nullptr, &err);
    CHECK_CL_ERRCODE(err);

    // Create the sampler
    auto sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP, CL_FILTER_NEAREST, &err);
    CHECK_CL_ERRCODE(err);

    // Create buffer
    auto buffer_size = IMAGE_HEIGHT * IMAGE_WIDTH * 4 * sizeof(float);
    auto buffer = clCreateBuffer(context,
                                 CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                 buffer_size, nullptr, &err);
    CHECK_CL_ERRCODE(err);

    // Set kernel arguments
    err = clSetKernelArg(kernel_write, 0, sizeof(cl_mem), &image);
    CHECK_CL_ERRCODE(err);
    err = clSetKernelArg(kernel_copy, 0, sizeof(cl_mem), &image);
    CHECK_CL_ERRCODE(err);
    err = clSetKernelArg(kernel_copy, 1, sizeof(cl_sampler), &sampler);
    CHECK_CL_ERRCODE(err);
    err = clSetKernelArg(kernel_copy, 2, sizeof(cl_mem), &buffer);
    CHECK_CL_ERRCODE(err);

    // Enqueue kernels
    size_t gws[3] = {IMAGE_WIDTH, IMAGE_HEIGHT, 0};
    size_t lws[3] = {1, 1, 1};

    err = clEnqueueNDRangeKernel(queue, kernel_write, 2, nullptr, gws, lws,
                                 0, nullptr, nullptr);
    CHECK_CL_ERRCODE(err);
    err = clEnqueueNDRangeKernel(queue, kernel_copy, 2, nullptr, gws, lws,
                                 0, nullptr, nullptr);
    CHECK_CL_ERRCODE(err);

    // Complete execution
    err = clFinish(queue);
    CHECK_CL_ERRCODE(err);


    // Map the buffer
    auto ptr = clEnqueueMapBuffer(queue, buffer, CL_TRUE, CL_MAP_READ, 0,
                                  buffer_size, 0, nullptr, nullptr, &err);
    CHECK_CL_ERRCODE(err);

    // Check the expected result
    auto buffer_data = static_cast<cl_float4*>(ptr);
    bool success = true;
    for (cl_uint i = 0; i < IMAGE_HEIGHT * IMAGE_WIDTH; ++i) {
        auto val = buffer_data[i];
        if ((val.x != 1.0f) || (val.y != 2.0f) || (val.z != 3.0f) || (val.w != 4.0f)) {
            printf("Failed comparison at buffer_data[%d]: "
                   "expected {1.0,2.0,3.0,4.0} but got {%f,%f,%f,%f}\n",
                   i, val.x, val.y, val.z, val.w);
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
    clReleaseSampler(sampler);
    clReleaseMemObject(image);
    clReleaseCommandQueue(queue);
    clReleaseKernel(kernel_copy);
    clReleaseKernel(kernel_write);
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

