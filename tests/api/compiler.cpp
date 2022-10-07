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

TEST_F(WithContext, DISABLED_NOCOMPILER(BuildLog)) {
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
    auto linked_program = LinkProgram(1, &program_list);

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
    EnqueueReadBuffer(gws_output, CL_BLOCKING, 0, sizeof(cl_uint), &result);
    EXPECT_EQ(result, gws[0]);
    EnqueueReadBuffer(lws_output, CL_BLOCKING, 0, sizeof(cl_uint), &result);
    EXPECT_EQ(result, lws[0]);
    EnqueueReadBuffer(dim_output, CL_BLOCKING, 0, sizeof(cl_uint), &result);
    EXPECT_EQ(result, 1);
}

// Test that literal sampler information is propagated correctly when linking.
TEST_F(WithCommandQueue,
       DISABLED_SWIFTSHADER(CompileAndLinkWithLiteralSamplers)) {
    // Read just past the end of a 1D image with two different samplers.
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
    auto linked_program = LinkProgram(1, &program_list);

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
    EnqueueReadBuffer(clamp_output, CL_BLOCKING, 0, sizeof(cl_uint), &result);
    EXPECT_EQ(result, value_end);
    EnqueueReadBuffer(repeat_output, CL_BLOCKING, 0, sizeof(cl_uint), &result);
    EXPECT_EQ(result, value_start);
}

TEST_F(WithContext, CompileWithHeaderInDirectory) {
    static const char* header = R"(
      int foo() { return 0;}
    )";
    static const char* source = R"(
      #include "dir/header.h"
      kernel void test(global uint *output) {
        output[0] = foo();
      }
    )";

    auto program = CreateProgram(source);
    auto program_header = CreateProgram(header);

    const char* header_name = "dir/header.h";
    cl_program program_list[1] = {program_header};
    CompileProgram(program, nullptr, 1, program_list, &header_name);
}

// Test that module scope constant setup is working
TEST_F(WithCommandQueue, ModuleScopeConstantData) {
    static const char* source = R"(
        __constant uint ppp[2][3] = {{1,2,3}, {5}};
        kernel void test(global uint *output, uint off) {
          output[0] = ppp[0+off][0];
          output[1] = ppp[0+off][1];
          output[2] = ppp[0+off][2];
          output[3] = ppp[1+off][0];
          output[4] = ppp[1+off][1];
          output[5] = ppp[1+off][2];
        }
    )";

    auto kernel = CreateKernel(source, "test");

    auto output = CreateBuffer(CL_MEM_READ_WRITE, 24);

    cl_uint off = 0;
    SetKernelArg(kernel, 0, output);
    SetKernelArg(kernel, 1, &off);

    size_t gws = 1;
    EnqueueNDRangeKernel(kernel, 1, nullptr, &gws, nullptr);

    // Check results.
    cl_int result[6] = {-1};
    EnqueueReadBuffer(output, CL_BLOCKING, 0, 24, &result);
    EXPECT_EQ(result[0], 1);
    EXPECT_EQ(result[1], 2);
    EXPECT_EQ(result[2], 3);
    EXPECT_EQ(result[3], 5);
    EXPECT_EQ(result[4], 0);
    EXPECT_EQ(result[5], 0);
}

TEST_F(WithCommandQueue, ProgramBinaryCompile) {
    static const char* source = R"(
      kernel void test(global uint *output) {
        uint gid = get_global_id(0);
        output[gid] = gid;
      }
    )";
    auto program = CreateProgram(source);
    const size_t gws = 4;
    const size_t buffer_size = gws * sizeof(cl_uint);
    auto buffer = CreateBuffer(CL_MEM_WRITE_ONLY, buffer_size);
    cl_uint result[gws] = {0};
    ASSERT_EQ(buffer_size, sizeof(result));

    CompileProgram(program);
    ASSERT_EQ(GetProgramBinaryType(program),
              CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT);
    auto compiled_binary = GetProgramBinary(program);
    auto binary_program = CreateProgramWithBinary(compiled_binary);
    ASSERT_EQ(GetProgramBinaryType(binary_program),
              CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT);

    BuildProgram(binary_program);
    auto kernel = CreateKernel(binary_program, "test");
    SetKernelArg(kernel, 0, buffer);

    EnqueueNDRangeKernel(kernel, 1, nullptr, &gws, nullptr);
    EnqueueReadBuffer(buffer, CL_BLOCKING, 0, buffer_size, result);

    ASSERT_EQ(result[0], 0);
    ASSERT_EQ(result[1], 1);
    ASSERT_EQ(result[2], 2);
    ASSERT_EQ(result[3], 3);
}

TEST_F(WithCommandQueue, ProgramBinaryLinkLibrary) {
    static const char* source = R"(
      kernel void test(global uint *output) {
        uint gid = get_global_id(0);
        output[gid] = gid;
      }
    )";
    auto program = CreateProgram(source);
    CompileProgram(program);
    const size_t gws = 4;
    const size_t buffer_size = gws * sizeof(cl_uint);
    auto buffer = CreateBuffer(CL_MEM_WRITE_ONLY, buffer_size);
    cl_uint result[gws] = {0};
    ASSERT_EQ(buffer_size, sizeof(result));

    cl_program program_list = program;
    auto linked_program = LinkProgram(1, &program_list, "-create-library");
    ASSERT_EQ(GetProgramBinaryType(linked_program),
              CL_PROGRAM_BINARY_TYPE_LIBRARY);
    auto linked_binary = GetProgramBinary(linked_program);
    auto binary_program = CreateProgramWithBinary(linked_binary);
    ASSERT_EQ(GetProgramBinaryType(binary_program),
              CL_PROGRAM_BINARY_TYPE_LIBRARY);

    BuildProgram(binary_program);
    auto kernel = CreateKernel(binary_program, "test");
    SetKernelArg(kernel, 0, buffer);

    EnqueueNDRangeKernel(kernel, 1, nullptr, &gws, nullptr);
    EnqueueReadBuffer(buffer, CL_BLOCKING, 0, buffer_size, result);

    ASSERT_EQ(result[0], 0);
    ASSERT_EQ(result[1], 1);
    ASSERT_EQ(result[2], 2);
    ASSERT_EQ(result[3], 3);
}

TEST_F(WithCommandQueue, ProgramBinaryLink) {
    static const char* source = R"(
      kernel void test(global uint *output) {
        uint gid = get_global_id(0);
        output[gid] = gid;
      }
    )";
    auto program = CreateProgram(source);
    CompileProgram(program);
    const size_t gws = 4;
    const size_t buffer_size = gws * sizeof(cl_uint);
    auto buffer = CreateBuffer(CL_MEM_WRITE_ONLY, buffer_size);
    cl_uint result[gws] = {0};
    ASSERT_EQ(buffer_size, sizeof(result));

    cl_program program_list = program;
    auto linked_program = LinkProgram(1, &program_list);
    ASSERT_EQ(GetProgramBinaryType(linked_program),
              CL_PROGRAM_BINARY_TYPE_EXECUTABLE);
    auto linked_binary = GetProgramBinary(linked_program);
    auto binary_program = CreateProgramWithBinary(linked_binary);
    ASSERT_EQ(GetProgramBinaryType(binary_program),
              CL_PROGRAM_BINARY_TYPE_EXECUTABLE);

    BuildProgram(binary_program);
    auto kernel = CreateKernel(binary_program, "test");
    SetKernelArg(kernel, 0, buffer);

    EnqueueNDRangeKernel(kernel, 1, nullptr, &gws, nullptr);
    EnqueueReadBuffer(buffer, CL_BLOCKING, 0, buffer_size, result);

    ASSERT_EQ(result[0], 0);
    ASSERT_EQ(result[1], 1);
    ASSERT_EQ(result[2], 2);
    ASSERT_EQ(result[3], 3);
}

TEST_F(WithCommandQueue, ProgramBinaryExecutable) {
    static const char* source = R"(
      kernel void test(global uint *output) {
        uint gid = get_global_id(0);
        output[gid] = gid;
      }
    )";
    auto program = CreateProgram(source);
    const size_t gws = 4;
    const size_t buffer_size = gws * sizeof(cl_uint);
    auto buffer = CreateBuffer(CL_MEM_WRITE_ONLY, buffer_size);
    cl_uint result[gws] = {0};
    ASSERT_EQ(buffer_size, sizeof(result));

    BuildProgram(program);
    ASSERT_EQ(GetProgramBinaryType(program), CL_PROGRAM_BINARY_TYPE_EXECUTABLE);
    auto built_binary = GetProgramBinary(program);
    auto binary_program = CreateProgramWithBinary(built_binary);
    ASSERT_EQ(GetProgramBinaryType(binary_program),
              CL_PROGRAM_BINARY_TYPE_EXECUTABLE);

    BuildProgram(binary_program);
    auto kernel = CreateKernel(binary_program, "test");
    SetKernelArg(kernel, 0, buffer);

    EnqueueNDRangeKernel(kernel, 1, nullptr, &gws, nullptr);
    EnqueueReadBuffer(buffer, CL_BLOCKING, 0, buffer_size, result);

    ASSERT_EQ(result[0], 0);
    ASSERT_EQ(result[1], 1);
    ASSERT_EQ(result[2], 2);
    ASSERT_EQ(result[3], 3);
}

TEST_F(WithCommandQueue, LinkPrograms) {
    static const char* sourceA = R"(
      extern void bar(global uint *dst, global uint *src);

      kernel void foo(global uint *dst, global uint *src) {
        bar(dst, src);
      }
    )";
    static const char* sourceB = R"(
      void bar(global uint *dst, global uint *src);

      void bar(global uint *dst, global uint *src) {
        int gid = get_global_id(0);
        dst[gid] = src[gid];
      }
    )";

    auto programA = CreateProgram(sourceA);
    CompileProgram(programA);

    auto programB = CreateProgram(sourceB);
    CompileProgram(programB);

    cl_program program_list[2] = {programA, programB};
    auto program = LinkProgram(2, program_list);

    const size_t gws = 4;
    const size_t buffer_size = gws * sizeof(cl_uint);
    cl_uint source[gws] = {1, 2, 3, 4};
    cl_uint result[gws] = {0};
    auto buffer_src = CreateBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   buffer_size, source);
    auto buffer_dst = CreateBuffer(CL_MEM_WRITE_ONLY, buffer_size);
    ASSERT_EQ(buffer_size, sizeof(result));
    auto kernel = CreateKernel(program, "foo");
    SetKernelArg(kernel, 0, buffer_dst);
    SetKernelArg(kernel, 1, buffer_src);

    EnqueueNDRangeKernel(kernel, 1, nullptr, &gws, nullptr);
    EnqueueReadBuffer(buffer_dst, CL_BLOCKING, 0, buffer_size, result);

    ASSERT_EQ(result[0], source[0]);
    ASSERT_EQ(result[1], source[1]);
    ASSERT_EQ(result[2], source[2]);
    ASSERT_EQ(result[3], source[3]);
}
