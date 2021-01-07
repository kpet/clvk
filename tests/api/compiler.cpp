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
