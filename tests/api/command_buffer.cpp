// Copyright 2026 The clvk authors.
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

#include <array>
#include <vector>

namespace {

class WithCommandBuffer : public WithCommandQueue {
protected:
    void SetUp() override {
        WithCommandQueue::SetUp();
        if (!HasExtension(CL_KHR_COMMAND_BUFFER_EXTENSION_NAME)) {
            GTEST_SKIP() << "Device does not support "
                         << CL_KHR_COMMAND_BUFFER_EXTENSION_NAME;
        }
        LoadFn(m_create_cmdbuf, "clCreateCommandBufferKHR");
        LoadFn(m_release_cmdbuf, "clReleaseCommandBufferKHR");
        LoadFn(m_finalize_cmdbuf, "clFinalizeCommandBufferKHR");
        LoadFn(m_enqueue_cmdbuf, "clEnqueueCommandBufferKHR");
        LoadFn(m_cmd_copy_buffer, "clCommandCopyBufferKHR");
        LoadFn(m_cmd_copy_buffer_rect, "clCommandCopyBufferRectKHR");
        LoadFn(m_cmd_fill_buffer, "clCommandFillBufferKHR");
        LoadFn(m_cmd_ndrange_kernel, "clCommandNDRangeKernelKHR");
        LoadFn(m_cmd_copy_image, "clCommandCopyImageKHR");
        LoadFn(m_cmd_copy_buffer_to_image, "clCommandCopyBufferToImageKHR");
        LoadFn(m_cmd_copy_image_to_buffer, "clCommandCopyImageToBufferKHR");
        LoadFn(m_cmd_fill_image, "clCommandFillImageKHR");

        cl_int err = CL_SUCCESS;
        m_cmdbuf = m_create_cmdbuf(1, &m_queue, nullptr, &err);
        ASSERT_CL_SUCCESS(err);
        ASSERT_NE(m_cmdbuf, nullptr);
    }

    void TearDown() override {
        if (m_cmdbuf != nullptr) {
            ASSERT_CL_SUCCESS(m_release_cmdbuf(m_cmdbuf));
            m_cmdbuf = nullptr;
        }
        WithCommandQueue::TearDown();
    }

    template <typename Fn> void LoadFn(Fn& fn, const char* name) {
        fn = reinterpret_cast<Fn>(
            clGetExtensionFunctionAddressForPlatform(gPlatform, name));
        ASSERT_NE(fn, nullptr) << "Failed to load " << name;
    }

    cl_command_buffer_khr m_cmdbuf = nullptr;
    clCreateCommandBufferKHR_fn m_create_cmdbuf = nullptr;
    clReleaseCommandBufferKHR_fn m_release_cmdbuf = nullptr;
    clFinalizeCommandBufferKHR_fn m_finalize_cmdbuf = nullptr;
    clEnqueueCommandBufferKHR_fn m_enqueue_cmdbuf = nullptr;
    clCommandCopyBufferKHR_fn m_cmd_copy_buffer = nullptr;
    clCommandCopyBufferRectKHR_fn m_cmd_copy_buffer_rect = nullptr;
    clCommandFillBufferKHR_fn m_cmd_fill_buffer = nullptr;
    clCommandNDRangeKernelKHR_fn m_cmd_ndrange_kernel = nullptr;
    clCommandCopyImageKHR_fn m_cmd_copy_image = nullptr;
    clCommandCopyBufferToImageKHR_fn m_cmd_copy_buffer_to_image = nullptr;
    clCommandCopyImageToBufferKHR_fn m_cmd_copy_image_to_buffer = nullptr;
    clCommandFillImageKHR_fn m_cmd_fill_image = nullptr;
};

} // namespace

TEST_F(WithCommandBuffer, FillBufferValidation) {
    auto buf = CreateBuffer(CL_MEM_READ_WRITE, 256, nullptr);
    cl_image_format fmt = {CL_RGBA, CL_UNSIGNED_INT8};
    cl_image_desc desc = {CL_MEM_OBJECT_IMAGE2D, 4, 4, 1, 1};
    auto img = CreateImage(CL_MEM_READ_WRITE, &fmt, &desc);

    std::array<char, 256> pattern{};

    // Invalid buffer / image passed as buffer
    EXPECT_EQ(m_cmd_fill_buffer(m_cmdbuf, nullptr, nullptr, nullptr,
                                pattern.data(), 4, 0, 16, 0, nullptr, nullptr,
                                nullptr),
              CL_INVALID_MEM_OBJECT);
    EXPECT_EQ(m_cmd_fill_buffer(m_cmdbuf, nullptr, nullptr, img, pattern.data(),
                                4, 0, 16, 0, nullptr, nullptr, nullptr),
              CL_INVALID_MEM_OBJECT);

    // Null pattern and invalid pattern sizes (including > 128 overflow case)
    EXPECT_EQ(m_cmd_fill_buffer(m_cmdbuf, nullptr, nullptr, buf, nullptr, 4, 0,
                                16, 0, nullptr, nullptr, nullptr),
              CL_INVALID_VALUE);
    EXPECT_EQ(m_cmd_fill_buffer(m_cmdbuf, nullptr, nullptr, buf, pattern.data(),
                                0, 0, 16, 0, nullptr, nullptr, nullptr),
              CL_INVALID_VALUE);
    EXPECT_EQ(m_cmd_fill_buffer(m_cmdbuf, nullptr, nullptr, buf, pattern.data(),
                                3, 0, 12, 0, nullptr, nullptr, nullptr),
              CL_INVALID_VALUE);
    EXPECT_EQ(m_cmd_fill_buffer(m_cmdbuf, nullptr, nullptr, buf, pattern.data(),
                                256, 0, 256, 0, nullptr, nullptr, nullptr),
              CL_INVALID_VALUE);

    // Unaligned offset / size
    EXPECT_EQ(m_cmd_fill_buffer(m_cmdbuf, nullptr, nullptr, buf, pattern.data(),
                                4, 2, 16, 0, nullptr, nullptr, nullptr),
              CL_INVALID_VALUE);
    EXPECT_EQ(m_cmd_fill_buffer(m_cmdbuf, nullptr, nullptr, buf, pattern.data(),
                                4, 0, 6, 0, nullptr, nullptr, nullptr),
              CL_INVALID_VALUE);

    // Zero size and out-of-bounds
    EXPECT_EQ(m_cmd_fill_buffer(m_cmdbuf, nullptr, nullptr, buf, pattern.data(),
                                4, 0, 0, 0, nullptr, nullptr, nullptr),
              CL_INVALID_VALUE);
    EXPECT_EQ(m_cmd_fill_buffer(m_cmdbuf, nullptr, nullptr, buf, pattern.data(),
                                4, 252, 8, 0, nullptr, nullptr, nullptr),
              CL_INVALID_VALUE);

    // Valid call
    EXPECT_CL_SUCCESS(m_cmd_fill_buffer(m_cmdbuf, nullptr, nullptr, buf,
                                        pattern.data(), 128, 0, 256, 0, nullptr,
                                        nullptr, nullptr));
}

TEST_F(WithCommandBuffer, NDRangeKernelValidation) {
    static const char* source = R"(
    kernel void test_kernel(global uint* out) {
        out[get_global_id(0)] = 1;
    }
    )";
    auto kernel = CreateKernel(source, "test_kernel");
    auto buf = CreateBuffer(CL_MEM_READ_WRITE, 16 * sizeof(cl_uint), nullptr);

    size_t gws[4] = {4, 1, 1, 1};
    size_t lws_zero[3] = {0, 1, 1};

    // Invalid kernel
    EXPECT_EQ(m_cmd_ndrange_kernel(m_cmdbuf, nullptr, nullptr, nullptr, 1,
                                   nullptr, gws, nullptr, 0, nullptr, nullptr,
                                   nullptr),
              CL_INVALID_KERNEL);

    // Invalid work_dim (0 and > 3 OOB case)
    EXPECT_EQ(m_cmd_ndrange_kernel(m_cmdbuf, nullptr, nullptr, kernel, 0,
                                   nullptr, gws, nullptr, 0, nullptr, nullptr,
                                   nullptr),
              CL_INVALID_WORK_DIMENSION);
    EXPECT_EQ(m_cmd_ndrange_kernel(m_cmdbuf, nullptr, nullptr, kernel, 4,
                                   nullptr, gws, nullptr, 0, nullptr, nullptr,
                                   nullptr),
              CL_INVALID_WORK_DIMENSION);

    // Missing kernel arguments
    EXPECT_EQ(m_cmd_ndrange_kernel(m_cmdbuf, nullptr, nullptr, kernel, 1,
                                   nullptr, gws, nullptr, 0, nullptr, nullptr,
                                   nullptr),
              CL_INVALID_KERNEL_ARGS);

    SetKernelArg(kernel, 0, buf);

    // Zero local work size
    EXPECT_EQ(m_cmd_ndrange_kernel(m_cmdbuf, nullptr, nullptr, kernel, 1,
                                   nullptr, gws, lws_zero, 0, nullptr, nullptr,
                                   nullptr),
              CL_INVALID_WORK_GROUP_SIZE);

    // Null global_work_size (OpenCL 2.1+ zero-sized NDRange) should not crash
    EXPECT_CL_SUCCESS(m_cmd_ndrange_kernel(m_cmdbuf, nullptr, nullptr, kernel,
                                           1, nullptr, nullptr, nullptr, 0,
                                           nullptr, nullptr, nullptr));
    // Valid call
    EXPECT_CL_SUCCESS(m_cmd_ndrange_kernel(m_cmdbuf, nullptr, nullptr, kernel,
                                           1, nullptr, gws, nullptr, 0, nullptr,
                                           nullptr, nullptr));
    ASSERT_CL_SUCCESS(m_finalize_cmdbuf(m_cmdbuf));
    ASSERT_CL_SUCCESS(
        m_enqueue_cmdbuf(0, nullptr, m_cmdbuf, 0, nullptr, nullptr));
    Finish();
}

TEST_F(WithCommandBuffer, CopyBufferValidation) {
    auto buf1 = CreateBuffer(CL_MEM_READ_WRITE, 64, nullptr);
    auto buf2 = CreateBuffer(CL_MEM_READ_WRITE, 64, nullptr);
    cl_image_format fmt = {CL_RGBA, CL_UNSIGNED_INT8};
    cl_image_desc desc = {CL_MEM_OBJECT_IMAGE2D, 4, 4, 1, 1};
    auto img = CreateImage(CL_MEM_READ_WRITE, &fmt, &desc);

    // Invalid buffer / image type confusion
    EXPECT_EQ(m_cmd_copy_buffer(m_cmdbuf, nullptr, nullptr, nullptr, buf2, 0, 0,
                                16, 0, nullptr, nullptr, nullptr),
              CL_INVALID_MEM_OBJECT);
    EXPECT_EQ(m_cmd_copy_buffer(m_cmdbuf, nullptr, nullptr, buf1, nullptr, 0, 0,
                                16, 0, nullptr, nullptr, nullptr),
              CL_INVALID_MEM_OBJECT);
    EXPECT_EQ(m_cmd_copy_buffer(m_cmdbuf, nullptr, nullptr, img, buf2, 0, 0, 16,
                                0, nullptr, nullptr, nullptr),
              CL_INVALID_MEM_OBJECT);
    EXPECT_EQ(m_cmd_copy_buffer(m_cmdbuf, nullptr, nullptr, buf1, img, 0, 0, 16,
                                0, nullptr, nullptr, nullptr),
              CL_INVALID_MEM_OBJECT);

    // Zero size / out of bounds
    EXPECT_EQ(m_cmd_copy_buffer(m_cmdbuf, nullptr, nullptr, buf1, buf2, 0, 0, 0,
                                0, nullptr, nullptr, nullptr),
              CL_INVALID_VALUE);
    EXPECT_EQ(m_cmd_copy_buffer(m_cmdbuf, nullptr, nullptr, buf1, buf2, 48, 0,
                                32, 0, nullptr, nullptr, nullptr),
              CL_INVALID_VALUE);
    EXPECT_EQ(m_cmd_copy_buffer(m_cmdbuf, nullptr, nullptr, buf1, buf2, 0, 48,
                                32, 0, nullptr, nullptr, nullptr),
              CL_INVALID_VALUE);

    // Overlapping copy in same buffer
    EXPECT_EQ(m_cmd_copy_buffer(m_cmdbuf, nullptr, nullptr, buf1, buf1, 0, 8,
                                16, 0, nullptr, nullptr, nullptr),
              CL_MEM_COPY_OVERLAP);

    // Adjacent non-overlapping copy in same buffer succeeds
    EXPECT_CL_SUCCESS(m_cmd_copy_buffer(m_cmdbuf, nullptr, nullptr, buf1, buf1,
                                        0, 16, 16, 0, nullptr, nullptr,
                                        nullptr));
}

TEST_F(WithCommandBuffer, CopyBufferRectValidation) {
    auto buf1 = CreateBuffer(CL_MEM_READ_WRITE, 64, nullptr);
    auto buf2 = CreateBuffer(CL_MEM_READ_WRITE, 64, nullptr);
    cl_image_format fmt = {CL_RGBA, CL_UNSIGNED_INT8};
    cl_image_desc desc = {CL_MEM_OBJECT_IMAGE2D, 4, 4, 1, 1};
    auto img = CreateImage(CL_MEM_READ_WRITE, &fmt, &desc);

    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {8, 4, 1};
    size_t zero_region[3] = {8, 0, 1};
    size_t oob_region[3] = {16, 8, 1};

    // Invalid buffer / image type confusion
    EXPECT_EQ(m_cmd_copy_buffer_rect(m_cmdbuf, nullptr, nullptr, nullptr, buf2,
                                     origin, origin, region, 0, 0, 0, 0, 0,
                                     nullptr, nullptr, nullptr),
              CL_INVALID_MEM_OBJECT);
    EXPECT_EQ(m_cmd_copy_buffer_rect(m_cmdbuf, nullptr, nullptr, buf1, img,
                                     origin, origin, region, 0, 0, 0, 0, 0,
                                     nullptr, nullptr, nullptr),
              CL_INVALID_MEM_OBJECT);

    // Null origin / region
    EXPECT_EQ(m_cmd_copy_buffer_rect(m_cmdbuf, nullptr, nullptr, buf1, buf2,
                                     nullptr, origin, region, 0, 0, 0, 0, 0,
                                     nullptr, nullptr, nullptr),
              CL_INVALID_VALUE);
    EXPECT_EQ(m_cmd_copy_buffer_rect(m_cmdbuf, nullptr, nullptr, buf1, buf2,
                                     origin, nullptr, region, 0, 0, 0, 0, 0,
                                     nullptr, nullptr, nullptr),
              CL_INVALID_VALUE);
    EXPECT_EQ(m_cmd_copy_buffer_rect(m_cmdbuf, nullptr, nullptr, buf1, buf2,
                                     origin, origin, nullptr, 0, 0, 0, 0, 0,
                                     nullptr, nullptr, nullptr),
              CL_INVALID_VALUE);

    // Zero region / out of bounds / invalid pitch
    EXPECT_EQ(m_cmd_copy_buffer_rect(m_cmdbuf, nullptr, nullptr, buf1, buf2,
                                     origin, origin, zero_region, 0, 0, 0, 0, 0,
                                     nullptr, nullptr, nullptr),
              CL_INVALID_VALUE);
    EXPECT_EQ(m_cmd_copy_buffer_rect(m_cmdbuf, nullptr, nullptr, buf1, buf2,
                                     origin, origin, oob_region, 0, 0, 0, 0, 0,
                                     nullptr, nullptr, nullptr),
              CL_INVALID_VALUE);
    EXPECT_EQ(m_cmd_copy_buffer_rect(m_cmdbuf, nullptr, nullptr, buf1, buf2,
                                     origin, origin, region, 4, 0, 0, 0, 0,
                                     nullptr, nullptr, nullptr),
              CL_INVALID_VALUE);

    EXPECT_CL_SUCCESS(m_cmd_copy_buffer_rect(
        m_cmdbuf, nullptr, nullptr, buf1, buf2, origin, origin, region, 0, 0, 0,
        0, 0, nullptr, nullptr, nullptr));
}

TEST_F(WithCommandBuffer, ImageCommandsValidation) {
    auto buf = CreateBuffer(CL_MEM_READ_WRITE, 256, nullptr);
    cl_image_format fmt_rgba = {CL_RGBA, CL_UNSIGNED_INT8};
    cl_image_format fmt_r = {CL_R, CL_UNSIGNED_INT8};
    cl_image_desc desc = {CL_MEM_OBJECT_IMAGE2D, 8, 8, 1, 1};
    auto img1 = CreateImage(CL_MEM_READ_WRITE, &fmt_rgba, &desc);
    auto img2 = CreateImage(CL_MEM_READ_WRITE, &fmt_rgba, &desc);
    auto img_r = CreateImage(CL_MEM_READ_WRITE, &fmt_r, &desc);

    size_t origin[3] = {0, 0, 0};
    size_t overlap_dst[3] = {2, 2, 0};
    size_t disjoint_dst[3] = {4, 0, 0};
    size_t region[3] = {4, 4, 1};
    cl_uint fill_color[4] = {1, 2, 3, 4};

    // clCommandCopyImageKHR
    EXPECT_EQ(m_cmd_copy_image(m_cmdbuf, nullptr, nullptr, nullptr, img2,
                               origin, origin, region, 0, nullptr, nullptr,
                               nullptr),
              CL_INVALID_MEM_OBJECT);
    EXPECT_EQ(m_cmd_copy_image(m_cmdbuf, nullptr, nullptr, buf, img2, origin,
                               origin, region, 0, nullptr, nullptr, nullptr),
              CL_INVALID_MEM_OBJECT);
    EXPECT_EQ(m_cmd_copy_image(m_cmdbuf, nullptr, nullptr, img1, buf, origin,
                               origin, region, 0, nullptr, nullptr, nullptr),
              CL_INVALID_MEM_OBJECT);
    EXPECT_EQ(m_cmd_copy_image(m_cmdbuf, nullptr, nullptr, img1, img_r, origin,
                               origin, region, 0, nullptr, nullptr, nullptr),
              CL_IMAGE_FORMAT_MISMATCH);
    EXPECT_EQ(m_cmd_copy_image(m_cmdbuf, nullptr, nullptr, img1, img2, nullptr,
                               origin, region, 0, nullptr, nullptr, nullptr),
              CL_INVALID_VALUE);
    EXPECT_EQ(m_cmd_copy_image(m_cmdbuf, nullptr, nullptr, img1, img2, origin,
                               nullptr, region, 0, nullptr, nullptr, nullptr),
              CL_INVALID_VALUE);
    EXPECT_EQ(m_cmd_copy_image(m_cmdbuf, nullptr, nullptr, img1, img2, origin,
                               origin, nullptr, 0, nullptr, nullptr, nullptr),
              CL_INVALID_VALUE);
    EXPECT_EQ(m_cmd_copy_image(m_cmdbuf, nullptr, nullptr, img1, img1, origin,
                               overlap_dst, region, 0, nullptr, nullptr,
                               nullptr),
              CL_MEM_COPY_OVERLAP);
    EXPECT_CL_SUCCESS(m_cmd_copy_image(m_cmdbuf, nullptr, nullptr, img1, img1,
                                       origin, disjoint_dst, region, 0, nullptr,
                                       nullptr, nullptr));

    // clCommandCopyBufferToImageKHR
    EXPECT_EQ(m_cmd_copy_buffer_to_image(m_cmdbuf, nullptr, nullptr, img1, img2,
                                         0, origin, region, 0, nullptr, nullptr,
                                         nullptr),
              CL_INVALID_MEM_OBJECT);
    EXPECT_EQ(m_cmd_copy_buffer_to_image(m_cmdbuf, nullptr, nullptr, buf, buf,
                                         0, origin, region, 0, nullptr, nullptr,
                                         nullptr),
              CL_INVALID_MEM_OBJECT);
    EXPECT_EQ(m_cmd_copy_buffer_to_image(m_cmdbuf, nullptr, nullptr, buf, img1,
                                         0, nullptr, region, 0, nullptr,
                                         nullptr, nullptr),
              CL_INVALID_VALUE);
    EXPECT_EQ(m_cmd_copy_buffer_to_image(m_cmdbuf, nullptr, nullptr, buf, img1,
                                         0, origin, nullptr, 0, nullptr,
                                         nullptr, nullptr),
              CL_INVALID_VALUE);
    EXPECT_CL_SUCCESS(m_cmd_copy_buffer_to_image(m_cmdbuf, nullptr, nullptr,
                                                 buf, img1, 0, origin, region,
                                                 0, nullptr, nullptr, nullptr));

    // clCommandCopyImageToBufferKHR
    EXPECT_EQ(m_cmd_copy_image_to_buffer(m_cmdbuf, nullptr, nullptr, buf, buf,
                                         origin, region, 0, 0, nullptr, nullptr,
                                         nullptr),
              CL_INVALID_MEM_OBJECT);
    EXPECT_EQ(m_cmd_copy_image_to_buffer(m_cmdbuf, nullptr, nullptr, img1, img2,
                                         origin, region, 0, 0, nullptr, nullptr,
                                         nullptr),
              CL_INVALID_MEM_OBJECT);
    EXPECT_EQ(m_cmd_copy_image_to_buffer(m_cmdbuf, nullptr, nullptr, img1, buf,
                                         nullptr, region, 0, 0, nullptr,
                                         nullptr, nullptr),
              CL_INVALID_VALUE);
    EXPECT_EQ(m_cmd_copy_image_to_buffer(m_cmdbuf, nullptr, nullptr, img1, buf,
                                         origin, nullptr, 0, 0, nullptr,
                                         nullptr, nullptr),
              CL_INVALID_VALUE);
    EXPECT_CL_SUCCESS(m_cmd_copy_image_to_buffer(m_cmdbuf, nullptr, nullptr,
                                                 img1, buf, origin, region, 0,
                                                 0, nullptr, nullptr, nullptr));

    // clCommandFillImageKHR
    EXPECT_EQ(m_cmd_fill_image(m_cmdbuf, nullptr, nullptr, nullptr, fill_color,
                               origin, region, 0, nullptr, nullptr, nullptr),
              CL_INVALID_MEM_OBJECT);
    EXPECT_EQ(m_cmd_fill_image(m_cmdbuf, nullptr, nullptr, buf, fill_color,
                               origin, region, 0, nullptr, nullptr, nullptr),
              CL_INVALID_MEM_OBJECT);
    EXPECT_EQ(m_cmd_fill_image(m_cmdbuf, nullptr, nullptr, img1, nullptr,
                               origin, region, 0, nullptr, nullptr, nullptr),
              CL_INVALID_VALUE);
    EXPECT_EQ(m_cmd_fill_image(m_cmdbuf, nullptr, nullptr, img1, fill_color,
                               nullptr, region, 0, nullptr, nullptr, nullptr),
              CL_INVALID_VALUE);
    EXPECT_EQ(m_cmd_fill_image(m_cmdbuf, nullptr, nullptr, img1, fill_color,
                               origin, nullptr, 0, nullptr, nullptr, nullptr),
              CL_INVALID_VALUE);
    EXPECT_CL_SUCCESS(m_cmd_fill_image(m_cmdbuf, nullptr, nullptr, img1,
                                       fill_color, origin, region, 0, nullptr,
                                       nullptr, nullptr));
}
