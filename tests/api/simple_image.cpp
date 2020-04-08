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

#include "testcl.hpp"

static const size_t IMAGE_HEIGHT = 128;
static const size_t IMAGE_WIDTH = 128;

static const char* program_source = R"(
kernel void write(image2d_t write_only img)
{
    int2 coord = {(int)get_global_id(0), (int)get_global_id(1)};
    float4 color = {1, 2, 3, 4};
    write_imagef(img, coord, color);
}
kernel void copy(image2d_t read_only img, sampler_t sampler, global float4 *buffer, int row_pitch)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    int2 coord = {x, y};
    float4 color = read_imagef(img, sampler, coord);
    buffer[(y * row_pitch) + x] = color;
}
)";

TEST_F(WithCommandQueue, DISABLED_TALVOS_SWIFTSHADER(SimpleImage)) {
    // Create and build program
    auto program = CreateAndBuildProgram(program_source);

    // Create kernels
    auto kernel_write = CreateKernel(program, "write");
    auto kernel_copy = CreateKernel(program, "copy");

    // Create the image
    cl_image_format format = {CL_RGBA, CL_FLOAT};
    cl_image_desc desc = {
        CL_MEM_OBJECT_IMAGE2D, // image_type
        IMAGE_WIDTH,           // image_width
        IMAGE_HEIGHT,          // image_height
        1,                     // image_depth
        1,                     // image_array_size
        0,                     // image_row_pitch
        0,                     // image_slice_pitch
        0,                     // num_mip_levels
        0,                     // num_samples
        nullptr,               // buffer
    };
    auto image = CreateImage(CL_MEM_READ_WRITE, &format, &desc, nullptr);

    // Create the sampler
    auto sampler = CreateSampler(CL_FALSE, CL_ADDRESS_CLAMP, CL_FILTER_NEAREST);

    // Create buffer
    auto buffer_size = IMAGE_HEIGHT * IMAGE_WIDTH * sizeof(cl_float4);
    auto buffer = CreateBuffer(CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                               buffer_size, nullptr);

    // Dispatch kernels
    size_t gws[3] = {IMAGE_WIDTH, IMAGE_HEIGHT, 0};
    size_t lws[3] = {1, 1, 1};

    SetKernelArg(kernel_write, 0, image);
    SetKernelArg(kernel_copy, 0, image);
    SetKernelArg(kernel_copy, 1, sampler);
    SetKernelArg(kernel_copy, 2, buffer);
    cl_int buffer_row_pitch = IMAGE_WIDTH;
    SetKernelArg(kernel_copy, 3, &buffer_row_pitch);

    EnqueueNDRangeKernel(kernel_write, 2, nullptr, gws, lws);
    EnqueueNDRangeKernel(kernel_copy, 2, nullptr, gws, lws);

    // Complete execution
    Finish();

    // Map the buffer
    auto data = EnqueueMapBuffer<cl_float4>(buffer, CL_TRUE, CL_MAP_READ, 0,
                                            buffer_size);

    // Check the expected result
    bool success = true;
    for (cl_uint i = 0; i < IMAGE_HEIGHT * IMAGE_WIDTH; ++i) {
        auto val = data[i];
        if ((val.x != 1.0f) || (val.y != 2.0f) || (val.z != 3.0f) ||
            (val.w != 4.0f)) {
            printf("Failed comparison at data[%d]: "
                   "expected {1.0,2.0,3.0,4.0} but got {%f,%f,%f,%f}\n",
                   i, val.x, val.y, val.z, val.w);
            success = false;
        }
    }
    EXPECT_TRUE(success);

    // Unmap the buffer
    EnqueueUnmapMemObject(buffer, data);
    Finish();
}
