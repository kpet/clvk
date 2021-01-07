// Copyright 2020 The clvk authors.
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

TEST_F(WithContext, BuildLog) {
    static const char* source_warning =
        "#warning THIS IS A WARNING\nvoid kernel test(){}\n";
    static const char* source_error = "#error THIS IS AN ERROR";
    std::string build_log;

    // Test log for successful compilation
    auto program_warning = CreateAndBuildProgram(source_warning);

    build_log = GetProgramBuildLog(program_warning);
    ASSERT_TRUE(build_log.find("THIS IS A WARNING") != std::string::npos);

    // Test log for failed compilation
    auto program_error = CreateProgram(source_error);
    cl_int err =
        clBuildProgram(program_error, 1, &gDevice, nullptr, nullptr, nullptr);
    ASSERT_EQ(err, CL_BUILD_PROGRAM_FAILURE);

    build_log = GetProgramBuildLog(program_error);
    ASSERT_TRUE(build_log.find("THIS IS AN ERROR") != std::string::npos);
}

// Test that push constant information is propagated correctly when linking.
TEST_F(WithCommandQueue, CompileAndLinkWithPushConstants) {
    static const char* source = R"(
        kernel void test(global uint *gws_output,
                         global uint *lws_output,
                         global uint *dim_output) {
          *gws_output = get_global_size(0);
          *lws_output = get_local_size(0);
          *dim_output = get_work_dim();
        }
    )";

    auto program = CreateProgram(source);

    // Enable 2.0 to generate push constants for the global size.
    CompileProgram(program, "-cl-std=CL2.0");

    cl_program program_list = program;
    holder<cl_program> linked_program = LinkProgram(1, &program_list);

    auto gws_output = CreateBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint));
    auto lws_output = CreateBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint));
    auto dim_output = CreateBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint));

    auto kernel = CreateKernel(linked_program, "test");
    SetKernelArg(kernel, 0, gws_output);
    SetKernelArg(kernel, 1, lws_output);
    SetKernelArg(kernel, 2, dim_output);

    size_t gws[3] = {32, 1, 1};
    size_t lws[3] = {8, 1, 1};
    EnqueueNDRangeKernel(kernel, 1, nullptr, gws, lws);
    Finish();

    // Check results.
    cl_uint result = -1;
    EnqueueReadBuffer(gws_output, CL_TRUE, 0, sizeof(cl_uint), &result);
    EXPECT_EQ(result, gws[0]);
    EnqueueReadBuffer(lws_output, CL_TRUE, 0, sizeof(cl_uint), &result);
    EXPECT_EQ(result, lws[0]);
    EnqueueReadBuffer(dim_output, CL_TRUE, 0, sizeof(cl_uint), &result);
    EXPECT_EQ(result, 1);
}

// Test that literal sampler information is propagated correctly when linking.
TEST_F(WithCommandQueue, CompileAndLinkWithLiteralSamplers) {
    // Read just past the end of a 1D image width two different samplers.
    static const char* source = R"(
        static constant sampler_t sampler_clamp = CLK_ADDRESS_CLAMP_TO_EDGE |
                                                  CLK_NORMALIZED_COORDS_TRUE |
                                                  CLK_FILTER_NEAREST;
        static constant sampler_t sampler_repeat = CLK_ADDRESS_REPEAT |
                                                   CLK_NORMALIZED_COORDS_TRUE |
                                                   CLK_FILTER_NEAREST;
        kernel void test(read_only image1d_t input,
                         global uint *output_clamp,
                         global uint *output_repeat) {
          *output_clamp = read_imageui(input, sampler_clamp, 1.f).x;
          *output_repeat = read_imageui(input, sampler_repeat, 1.f).x;
        }
    )";

    auto program = CreateProgram(source);

    CompileProgram(program);

    cl_program program_list = program;
    holder<cl_program> linked_program = LinkProgram(1, &program_list);

    // Create a 1D input image.
    size_t IMAGE_WIDTH = 4;
    cl_image_format format = {CL_R, CL_UNSIGNED_INT8};
    cl_image_desc desc = {
        CL_MEM_OBJECT_IMAGE1D, // image_type
        IMAGE_WIDTH,           // image_width
        1,                     // image_height
        1,                     // image_depth
        1,                     // image_array_size
        0,                     // image_row_pitch
        0,                     // image_slice_pitch
        0,                     // num_mip_levels
        0,                     // num_samples
        nullptr,               // buffer
    };
    auto input = CreateImage(CL_MEM_READ_ONLY, &format, &desc);

    // Set the input image values.
    cl_uchar value_start = 7;
    cl_uchar value_end = 42;
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {IMAGE_WIDTH, 1, 1};
    size_t row_pitch = IMAGE_WIDTH;
    auto data = EnqueueMapImage<cl_uchar>(input, CL_BLOCKING,
                                          CL_MAP_WRITE_INVALIDATE_REGION,
                                          origin, region, &row_pitch, nullptr);
    memset(data, 0xFF, IMAGE_WIDTH);
    data[0] = value_start;
    data[IMAGE_WIDTH - 1] = value_end;
    EnqueueUnmapMemObject(input, data);

    auto clamp_output = CreateBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint));
    auto repeat_output = CreateBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint));

    auto kernel = CreateKernel(linked_program, "test");
    SetKernelArg(kernel, 0, input);
    SetKernelArg(kernel, 1, clamp_output);
    SetKernelArg(kernel, 2, repeat_output);
    size_t gws[3] = {32, 1, 1};
    size_t lws[3] = {8, 1, 1};
    EnqueueNDRangeKernel(kernel, 1, nullptr, gws, lws);
    Finish();

    // Check results.
    cl_uint result = -1;
    EnqueueReadBuffer(clamp_output, CL_TRUE, 0, sizeof(cl_uint), &result);
    EXPECT_EQ(result, value_end);
    EnqueueReadBuffer(repeat_output, CL_TRUE, 0, sizeof(cl_uint), &result);
    EXPECT_EQ(result, value_start);
}
