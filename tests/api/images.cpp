// Copyright 2019 The clvk authors.
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

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#include "testcl.hpp"

TEST_F(WithContext, CreateImageLegacy) {
    cl_image_format format = {CL_RGBA, CL_UNORM_INT8};

    size_t width = 97, height = 135, depth = 7;
    size_t row_pitch = 128, slice_pitch = row_pitch * height;

    cl_int err;

    // Create 2D image
    auto im2d = clCreateImage2D(m_context, CL_MEM_READ_WRITE, &format, width,
                                height, row_pitch, nullptr, &err);
    ASSERT_CL_SUCCESS(err);

    size_t qval;
    cl_image_format qfmt;

    // Query its properties
    GetImageInfo(im2d, CL_IMAGE_WIDTH, sizeof(qval), &qval, nullptr);
    EXPECT_EQ(qval, width);
    GetImageInfo(im2d, CL_IMAGE_HEIGHT, sizeof(qval), &qval, nullptr);
    EXPECT_EQ(qval, height);
    GetImageInfo(im2d, CL_IMAGE_ROW_PITCH, sizeof(qval), &qval, nullptr);
    EXPECT_EQ(qval, row_pitch);
    GetImageInfo(im2d, CL_IMAGE_FORMAT, sizeof(qfmt), &qfmt, nullptr);
    EXPECT_EQ(qfmt.image_channel_order, format.image_channel_order);
    EXPECT_EQ(qfmt.image_channel_data_type, format.image_channel_data_type);

    cl_mem_object_type qtype;
    GetMemObjectInfo(im2d, CL_MEM_TYPE, sizeof(qtype), &qtype, nullptr);
    EXPECT_EQ(qtype, CL_MEM_OBJECT_IMAGE2D);

    // Do the same for 3D images
    auto im3d =
        clCreateImage3D(m_context, CL_MEM_READ_WRITE, &format, width, height,
                        depth, row_pitch, slice_pitch, nullptr, &err);
    ASSERT_CL_SUCCESS(err);
    GetImageInfo(im3d, CL_IMAGE_WIDTH, sizeof(qval), &qval, nullptr);
    EXPECT_EQ(qval, width);
    GetImageInfo(im3d, CL_IMAGE_HEIGHT, sizeof(qval), &qval, nullptr);
    EXPECT_EQ(qval, height);
    GetImageInfo(im3d, CL_IMAGE_DEPTH, sizeof(qval), &qval, nullptr);
    EXPECT_EQ(qval, depth);
    GetImageInfo(im3d, CL_IMAGE_ROW_PITCH, sizeof(qval), &qval, nullptr);
    EXPECT_EQ(qval, row_pitch);
    GetImageInfo(im3d, CL_IMAGE_SLICE_PITCH, sizeof(qval), &qval, nullptr);
    EXPECT_EQ(qval, slice_pitch);
    GetImageInfo(im3d, CL_IMAGE_FORMAT, sizeof(qfmt), &qfmt, nullptr);
    EXPECT_EQ(qfmt.image_channel_order, format.image_channel_order);
    EXPECT_EQ(qfmt.image_channel_data_type, format.image_channel_data_type);

    GetMemObjectInfo(im3d, CL_MEM_TYPE, sizeof(qtype), &qtype, nullptr);
    EXPECT_EQ(qtype, CL_MEM_OBJECT_IMAGE3D);

    clReleaseMemObject(im2d);
    clReleaseMemObject(im3d);
}

TEST_F(WithCommandQueue, DISABLED_TALVOS(BasicImageMapUnmap)) {
    const size_t IMAGE_WIDTH = 97;
    const size_t IMAGE_HEIGHT = 13;

    // Create an image
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
    auto image = CreateImage(CL_MEM_READ_WRITE, &format, &desc);

    // Map it
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {IMAGE_WIDTH, IMAGE_HEIGHT, 1};
    size_t row_pitch;
    auto data = EnqueueMapImage<cl_float4>(image, CL_BLOCKING,
                                           CL_MAP_WRITE_INVALIDATE_REGION,
                                           origin, region, &row_pitch, nullptr);
    size_t row_pitch_pixels = row_pitch / sizeof(cl_float4);
    // Fill with pattern
    for (auto row = 0u; row < IMAGE_HEIGHT; row++) {
        for (auto pix = 0u; pix < IMAGE_WIDTH; pix++) {
            data[row * row_pitch_pixels + pix] = {1.0f, 2.0f, 3.0f, 4.0f};
        }
    }

    // Unmap
    EnqueueUnmapMemObject(image, data);
    Finish();

    // Create a sampler
    auto sampler = CreateSampler(CL_FALSE, CL_ADDRESS_CLAMP, CL_FILTER_NEAREST);

    // Create a buffer
    auto buffer_size = IMAGE_HEIGHT * IMAGE_WIDTH * sizeof(cl_float4);
    auto buffer = CreateBuffer(CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                               buffer_size, nullptr);

    // Enqueue kernel to copy to a buffer
    static const char* source = R"(
    kernel void copy(image2d_t read_only img, sampler_t sampler, uint row_pitch, global float4 *buffer)
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    int2 coord = {(int)x, (int)y};
    float4 color = read_imagef(img, sampler, coord);
    buffer[(y * row_pitch) + x] = color;
}
)";

    auto kernel = CreateKernel(source, "copy");
    SetKernelArg(kernel, 0, image);
    SetKernelArg(kernel, 1, sampler);
    cl_uint row_pitch_pixels_buffer = IMAGE_WIDTH;
    SetKernelArg(kernel, 2, sizeof(cl_uint), &row_pitch_pixels_buffer);
    SetKernelArg(kernel, 3, buffer);

    size_t gws[3] = {IMAGE_WIDTH, IMAGE_HEIGHT, 0};
    size_t lws[3] = {1, 1, 1};

    EnqueueNDRangeKernel(kernel, 2, nullptr, gws, lws);

    // Map the buffer
    auto bdata = EnqueueMapBuffer<cl_float4>(buffer, CL_TRUE, CL_MAP_READ, 0,
                                             buffer_size);

    // Verify the content
    bool success = true;
    for (cl_uint i = 0; i < IMAGE_HEIGHT * IMAGE_WIDTH; ++i) {
        auto val = bdata[i];
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
    EnqueueUnmapMemObject(buffer, bdata);
    Finish();
}

TEST_F(WithCommandQueue, ImageReadMappingCantChangeImage) {
    // Create image
    const size_t IMAGE_WIDTH = 97;
    const size_t IMAGE_HEIGHT = 13;

    cl_image_format format = {CL_R, CL_UNSIGNED_INT8};
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
    auto image = CreateImage(CL_MEM_READ_WRITE, &format, &desc);

    // Init with pattern
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {IMAGE_WIDTH, IMAGE_HEIGHT, 1};
    const cl_uchar fill_value = 0xAB;
    size_t row_pitch;
    auto data = EnqueueMapImage<cl_uchar>(image, CL_BLOCKING,
                                          CL_MAP_WRITE_INVALIDATE_REGION,
                                          origin, region, &row_pitch, nullptr);
    size_t row_pitch_pixels = row_pitch / sizeof(int8_t);

    for (auto row = 0u; row < IMAGE_HEIGHT; row++) {
        for (auto pix = 0u; pix < IMAGE_WIDTH; pix++) {
            data[row * row_pitch_pixels + pix] = fill_value;
        }
    }

    EnqueueUnmapMemObject(image, data);
    Finish();

    // Map for reading a region
    data = EnqueueMapImage<cl_uchar>(image, CL_BLOCKING, CL_MAP_READ, origin,
                                     region, &row_pitch, nullptr);
    // Change pattern
    for (auto row = 0u; row < IMAGE_HEIGHT; row++) {
        for (auto pix = 0u; pix < IMAGE_WIDTH; pix++) {
            data[row * row_pitch_pixels + pix] = fill_value + pix;
        }
    }

    // Unmap
    EnqueueUnmapMemObject(image, data);
    Finish();

    // Map again
    data = EnqueueMapImage<cl_uchar>(image, CL_BLOCKING, CL_MAP_READ, origin,
                                     region, &row_pitch, nullptr);

    // Check the new pattern isn't in the image
    bool success = true;
    for (cl_uint i = 0; i < IMAGE_HEIGHT * IMAGE_WIDTH; ++i) {
        auto val = data[i];
        if (val != fill_value) {
            success = false;
        }
    }
    EXPECT_TRUE(success);

    EnqueueUnmapMemObject(image, data);
    Finish();
}

TEST_F(WithCommandQueue,
       DISABLED_TALVOS(ImageWriteInvalidateMappingDoesntCopyImageContent)) {
    // Create an image
    const size_t IMAGE_WIDTH = 97;
    const size_t IMAGE_HEIGHT = 13;

    cl_image_format format = {CL_R, CL_UNSIGNED_INT8};
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
    auto image = CreateImage(CL_MEM_READ_WRITE, &format, &desc);

    // Init content
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {IMAGE_WIDTH, IMAGE_HEIGHT, 1};
    const cl_uchar fill_value = 0x42;
    size_t row_pitch;
    auto data = EnqueueMapImage<cl_uchar>(image, CL_BLOCKING,
                                          CL_MAP_WRITE_INVALIDATE_REGION,
                                          origin, region, &row_pitch, nullptr);
    size_t row_pitch_pixels = row_pitch / sizeof(int8_t);

    for (auto row = 0u; row < IMAGE_HEIGHT; row++) {
        for (auto pix = 0u; pix < IMAGE_WIDTH; pix++) {
            data[row * row_pitch_pixels + pix] = fill_value;
        }
    }

    EnqueueUnmapMemObject(image, data);
    Finish();

    // Create a buffer of the size of the image to make it less likely that the
    // mapping buffer of the next image map command gets the same region of
    // memory occupied by the mapping buffer of the previous image map.
    auto buffer =
        CreateBuffer(CL_MEM_READ_WRITE, IMAGE_HEIGHT * IMAGE_WIDTH, nullptr);

    // Map with CL_MAP_WRITE_INVALIDATE_REGION
    data = EnqueueMapImage<cl_uchar>(image, CL_BLOCKING,
                                     CL_MAP_WRITE_INVALIDATE_REGION, origin,
                                     region, &row_pitch, nullptr);

    // Check the pattern isn't visible
    bool success = false;
    for (cl_uint i = 0; i < IMAGE_HEIGHT * IMAGE_WIDTH; ++i) {
        auto val = data[i];
        if (val != fill_value) {
            success = true;
        }
    }
    EXPECT_TRUE(success);

    EnqueueUnmapMemObject(image, data);
    Finish();
}
