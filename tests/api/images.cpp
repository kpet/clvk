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

TEST_F(WithContext, Issue303LuminanceImageFormats) {
    auto format = cl_image_format{CL_LUMINANCE, CL_FLOAT};
    uint32_t W = 55, H = 43;
    auto image_desc = cl_image_desc{
        CL_MEM_OBJECT_IMAGE2D, (size_t)W, (size_t)H, 1, 1, 0, 0, 0, 0, NULL};
    auto image = CreateImage(CL_MEM_READ_WRITE, &format, &image_desc, NULL);
}

TEST_F(WithCommandQueue, BasicImageMapUnmap) {
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

TEST_F(WithCommandQueue, ImageWriteInvalidateMappingDoesntCopyImageContent) {
    // Create an image
    const size_t IMAGE_WIDTH = 97;
    const size_t IMAGE_HEIGHT = 13;

    const cl_uchar fill_value = 0x42;
    cl_uchar host_data[IMAGE_WIDTH * IMAGE_HEIGHT];
    for (auto row = 0u; row < IMAGE_HEIGHT; row++) {
        for (auto pix = 0u; pix < IMAGE_WIDTH; pix++) {
            host_data[row * IMAGE_HEIGHT + pix] = fill_value;
        }
    }

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
    auto image = CreateImage(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, &format,
                             &desc, host_data);

    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {IMAGE_WIDTH, IMAGE_HEIGHT, 1};
    cl_uchar* data;
    size_t row_pitch;

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
            break;
        }
    }
    EXPECT_TRUE(success);

    EnqueueUnmapMemObject(image, data);
    Finish();
}

TEST_F(WithCommandQueue, ImageWriteOffset) {
    // Create a 2D image.
    const size_t IMAGE_WIDTH = 16;
    const size_t IMAGE_HEIGHT = 16;

    // This is the full image (used for initial fill and final read back).
    size_t full_origin[3] = {0, 0, 0};
    size_t full_region[3] = {IMAGE_WIDTH, IMAGE_HEIGHT, 1};

    // We are going to write to this region.
    size_t write_origin[3] = {IMAGE_WIDTH / 4, IMAGE_HEIGHT / 4, 0};
    size_t write_region[3] = {IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2, 1};

    cl_uchar fill_value = 0xFF;
    std::vector<cl_uchar> write_data(write_region[0] * write_region[1], 0);
    std::vector<cl_uchar> read_data(IMAGE_WIDTH * IMAGE_HEIGHT, 0);

    // Fill the source data.
    for (int y = 0; y < write_region[1]; y++) {
        for (int x = 0; x < write_region[0]; x++) {
            write_data[x + y * write_region[0]] = x + y * write_region[0];
        }
    }

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
    auto image = CreateImage(CL_MEM_READ_WRITE, &format, &desc, nullptr);

    // Fill the image.
    {
        cl_uint pattern[4] = {fill_value, fill_value, fill_value, fill_value};
        EnqueueFillImage(image, &pattern, full_origin, full_region);
        Finish();
    }

    // Write data to subsection of the image.
    {
        EnqueueWriteImage(image, CL_FALSE, write_origin, write_region, 0, 0,
                          write_data.data());
        Finish();
    }

    // Read the data back from the image.
    {
        EnqueueReadImage(image, CL_FALSE, full_origin, full_region, 0, 0,
                         read_data.data());
        Finish();
    }

    // Returns true if (x, y) is inside the target write region.
    auto in_target = [&](int x, int y) {
        return x >= write_origin[0] && x < write_origin[0] + write_region[0] &&
               y >= write_origin[1] && y < write_origin[1] + write_region[1];
    };

    // Check that we got the correct values.
    bool success = true;

    // Outside the target write region should match the initial fill_value.
    for (int y = 0; y < IMAGE_HEIGHT; ++y) {
        for (int x = 0; x < IMAGE_WIDTH; ++x) {
            cl_uchar val = read_data[x + y * IMAGE_WIDTH];
            if (in_target(x, y)) {
                continue;
            }
            if (val != fill_value) {
                printf("Failed comparison at (%d,%d): expected %d but got %d\n",
                       x, y, fill_value, val);
                success = false;
            }
        }
    }

    // Inside the target write region should be the same as `write_data`.
    for (int y = 0; y < write_region[1]; ++y) {
        for (int x = 0; x < write_region[0]; ++x) {
            cl_uchar val = read_data[(x + write_origin[0]) +
                                     (y + write_origin[1]) * IMAGE_WIDTH];
            cl_uchar ref = write_data[x + y * write_region[0]];
            if (val != ref) {
                printf("Failed comparison at (%d,%d): expected %d but got "
                       "%d\n",
                       x, y, ref, val);
                success = false;
            }
        }
    }

    EXPECT_TRUE(success);
}

TEST_F(WithContext, Image1DBuffer) {
    const size_t IMAGE_WIDTH = 128;

    auto buffer_size = IMAGE_WIDTH * sizeof(cl_float4);
    auto buffer = CreateBuffer(CL_MEM_READ_WRITE, buffer_size, nullptr);

    cl_image_format format = {CL_RGBA, CL_FLOAT};
    cl_image_desc desc = {
        CL_MEM_OBJECT_IMAGE1D_BUFFER, // image_type
        IMAGE_WIDTH,                  // image_width
        1,                            // image_height
        1,                            // image_depth
        1,                            // image_array_size
        0,                            // image_row_pitch
        0,                            // image_slice_pitch
        0,                            // num_mip_levels
        0,                            // num_samples
        buffer,                       // buffer
    };

    auto image = CreateImage(CL_MEM_READ_WRITE, &format, &desc);
}

TEST_F(WithCommandQueue, ImageCopyHostPtrPadding) {
    // Create a 2D image array.
    const size_t IMAGE_WIDTH = 16;
    const size_t IMAGE_HEIGHT = 16;
    const size_t IMAGE_ARRAY_SIZE = 2;

    // Pad the host pointer.
    const size_t IMAGE_ROW_PITCH = 24;
    const size_t IMAGE_SLICE_PITCH = IMAGE_ROW_PITCH * 32;

    const cl_uchar init_value = 0xAB;
    std::vector<cl_uchar> host_data(IMAGE_SLICE_PITCH * IMAGE_ARRAY_SIZE, 0xFF);
    std::vector<cl_uchar> read_data(
        IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_ARRAY_SIZE, 0);

    // Fill the host pointer (skipping the padding bytes).
    for (int a = 0; a < IMAGE_ARRAY_SIZE; a++) {
        for (int y = 0; y < IMAGE_HEIGHT; y++) {
            for (int x = 0; x < IMAGE_WIDTH; x++) {
                host_data[x + y * IMAGE_ROW_PITCH + a * IMAGE_SLICE_PITCH] =
                    init_value;
            }
        }
    }

    cl_image_format format = {CL_R, CL_UNSIGNED_INT8};
    cl_image_desc desc = {
        CL_MEM_OBJECT_IMAGE2D_ARRAY, // image_type
        IMAGE_WIDTH,                 // image_width
        IMAGE_HEIGHT,                // image_height
        1,                           // image_depth
        IMAGE_ARRAY_SIZE,            // image_array_size
        IMAGE_ROW_PITCH,             // image_row_pitch
        IMAGE_SLICE_PITCH,           // image_slice_pitch
        0,                           // num_mip_levels
        0,                           // num_samples
        nullptr,                     // buffer
    };
    auto image = CreateImage(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, &format,
                             &desc, host_data.data());

    // Read the data back from the image.
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_ARRAY_SIZE};
    EnqueueReadImage(image, CL_FALSE, origin, region, 0, 0, read_data.data());
    Finish();

    // Check that we got the values copied from the initial host pointer.
    bool success = true;
    for (int i = 0; i < IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_ARRAY_SIZE; ++i) {
        auto val = read_data[i];
        if (val != init_value) {
            printf("Failed comparison at data[%d]: expected %d but got %d\n", i,
                   init_value, val);
            success = false;
        }
    }
    EXPECT_TRUE(success);
}

TEST_F(WithCommandQueue, ImageCopyHostPtrMultiQueue) {
    // Create image
    const size_t IMAGE_WIDTH = 16;
    const size_t IMAGE_HEIGHT = 16;

    const cl_uchar init_value = 0xAB;
    const cl_uchar write_value = 0xFF;
    std::vector<cl_uchar> host_data(IMAGE_WIDTH * IMAGE_HEIGHT, init_value);
    std::vector<cl_uchar> write_data(IMAGE_WIDTH * IMAGE_HEIGHT, write_value);
    std::vector<cl_uchar> read_data(IMAGE_WIDTH * IMAGE_HEIGHT, 0);

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
    auto image = CreateImage(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, &format,
                             &desc, host_data.data());

    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {IMAGE_WIDTH, IMAGE_HEIGHT, 1};

    // Enqueue a command that writes to the image, blocked behind a user event.
    auto user_event = CreateUserEvent();
    cl_event wait_list = user_event;
    EnqueueWriteImage(image, CL_FALSE, origin, region, 0, 0, write_data.data(),
                      1, &wait_list, nullptr);
    Flush();

    // Read from the image with a different queue.
    auto queue2 = CreateCommandQueue(device(), 0);
    cl_int err = clEnqueueReadImage(queue2, image, CL_FALSE, origin, region, 0,
                                    0, read_data.data(), 0, nullptr, nullptr);
    ASSERT_CL_SUCCESS(err);
    Finish(queue2);

    // Check that we got the values copied from the initial host pointer.
    bool success = true;
    for (cl_uint i = 0; i < IMAGE_HEIGHT * IMAGE_WIDTH; ++i) {
        auto val = read_data[i];
        if (val != init_value) {
            printf("Failed comparison at data[%d]: expected %d but got %d\n", i,
                   init_value, val);
            success = false;
        }
    }
    EXPECT_TRUE(success);

    // Unblock the first queue.
    SetUserEventStatus(user_event, CL_COMPLETE);
    Finish();
}

TEST_F(WithCommandQueue, ImageChannelGetter) {
    uint32_t num_format;
    GetSupportedImageFormats(CL_MEM_OBJECT_IMAGE2D, 0, nullptr, &num_format);
    if (num_format > 4)
        num_format = 4;
    cl_image_format formats[4];
    GetSupportedImageFormats(CL_MEM_OBJECT_IMAGE2D, num_format, formats,
                             nullptr);

    const cl_image_desc desc = {
        CL_MEM_OBJECT_IMAGE2D, 1, 1, 1, 1, 0, 0, 0, 0, nullptr};
    cl_uint dst[3];
    auto dst_buffer = CreateBuffer(CL_MEM_WRITE_ONLY, sizeof(dst), nullptr);

    const char* source = R"(
kernel void test(global uint* dst, uint magic, image2d_t read_only image, uint off)
{
    dst[0] = (uint)get_image_channel_order(image) + off;
    dst[1] = (uint)get_image_channel_data_type(image) + off;
    dst[2] = magic + off;
}
)";

    for (unsigned i = 0; i < num_format; i++) {
        const cl_channel_order order = formats[i].image_channel_order;
        const cl_channel_type data_type = formats[i].image_channel_data_type;
        const cl_image_format format = {order, data_type};
        auto image = CreateImage(CL_MEM_READ_ONLY, &format, &desc);

        auto kernel = CreateKernel(source, "test");
        SetKernelArg(kernel, 0, dst_buffer);
        cl_uint offset = 42;
        cl_uint magic = offset + 13;
        SetKernelArg(kernel, 1, sizeof(cl_uint), &magic);
        SetKernelArg(kernel, 2, image);
        SetKernelArg(kernel, 3, sizeof(cl_uint), &offset);

        size_t gws = 1;
        EnqueueNDRangeKernel(kernel, 1, nullptr, &gws, nullptr);

        EnqueueReadBuffer(dst_buffer, true, 0, sizeof(dst), dst);

        EXPECT_TRUE((dst[0] == (order + offset)) &&
                    (dst[1] == (data_type + offset)) &&
                    (dst[2] == (offset + magic)));
    }
}

TEST_F(WithCommandQueue, 1DBufferImageFromSubBuffer) {
    const size_t IMAGE_WIDTH = 128;

    auto subbuffer_size = IMAGE_WIDTH * sizeof(cl_float4);
    auto buffer_size = subbuffer_size + 2 * sizeof(cl_float4);
    auto buffer = CreateBuffer(CL_MEM_READ_WRITE, buffer_size, nullptr);
    auto subbuffer =
        CreateSubBuffer(buffer, 0, sizeof(cl_float4), subbuffer_size);

    cl_image_format format = {CL_RGBA, CL_FLOAT};
    cl_image_desc desc = {
        CL_MEM_OBJECT_IMAGE1D_BUFFER, // image_type
        IMAGE_WIDTH,                  // image_width
        1,                            // image_height
        1,                            // image_depth
        1,                            // image_array_size
        0,                            // image_row_pitch
        0,                            // image_slice_pitch
        0,                            // num_mip_levels
        0,                            // num_samples
        subbuffer,                    // buffer
    };

    auto image = CreateImage(CL_MEM_READ_WRITE, &format, &desc);

    const char* source = R"(
kernel void test(image1d_buffer_t write_only image)
{
  int gid = get_global_id(0);
  write_imagef(image, gid, (float4)(0.0, 0.0, 0.0, 0.0));
}
)";
    auto kernel = CreateKernel(source, "test");
    SetKernelArg(kernel, 0, image);
    cl_float pattern = -42.0;
    EnqueueFillBuffer(buffer, &pattern, sizeof(pattern), buffer_size);
    EnqueueNDRangeKernel(kernel, 1, nullptr, &IMAGE_WIDTH, nullptr);

    cl_float4 output[IMAGE_WIDTH + 2];
    EnqueueReadBuffer(buffer, CL_TRUE, 0, buffer_size, output);
    EXPECT_TRUE(output[0].s0 == pattern && output[0].s1 == pattern &&
                output[0].s2 == pattern && output[0].s3 == pattern &&
                output[IMAGE_WIDTH + 1].s0 == pattern &&
                output[IMAGE_WIDTH + 1].s1 == pattern &&
                output[IMAGE_WIDTH + 1].s2 == pattern &&
                output[IMAGE_WIDTH + 1].s3 == pattern);
    for (unsigned i = 0; i < IMAGE_WIDTH; i++) {
        EXPECT_TRUE(output[i + 1].s0 == 0.0 && output[i + 1].s1 == 0.0 &&
                    output[i + 1].s2 == 0.0 && output[i + 1].s3 == 0.0);
    }
}
