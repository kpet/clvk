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

TEST_F(WithCommandQueue, SimpleLink) {
    static const char* source = R"(
        kernel void test(global uint *gws_output,
                         global uint *lws_output,
                         global uint *dim_output) {
          *gws_output = get_global_size(0);
          *lws_output = get_local_size(0);
          *dim_output = get_work_dim();
        }
    )";

    cl_int err;
    auto program = CreateProgram(source);

    // Enable 2.0 to generate push constants for the global size.
    err = clCompileProgram(program, 1, &gDevice, "-cl-std=CL2.0", 0, nullptr,
                           nullptr, nullptr, nullptr);
    ASSERT_EQ(err, CL_SUCCESS);

    cl_program program_list = program;
    holder<cl_program> linked_program =
        clLinkProgram(m_context, 1, &gDevice, nullptr, 1, &program_list,
                      nullptr, nullptr, &err);
    ASSERT_EQ(err, CL_SUCCESS);

    auto gws_output = CreateBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr);
    auto lws_output = CreateBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr);
    auto dim_output = CreateBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr);
    cl_uint value = 42;

    auto kernel = CreateKernel(linked_program, "test");
    SetKernelArg(kernel, 0, gws_output);
    SetKernelArg(kernel, 1, lws_output);
    SetKernelArg(kernel, 2, dim_output);

    size_t gws[3] = {32, 1, 1};
    size_t lws[3] = {8, 1, 1};
    EnqueueNDRangeKernel(kernel, 1, nullptr, gws, lws);
    Finish();

    cl_uint result = -1;

    err = clEnqueueReadBuffer(m_queue, gws_output, CL_TRUE, 0, sizeof(cl_uint),
                              &result, 0, nullptr, nullptr);
    ASSERT_EQ(err, CL_SUCCESS);
    EXPECT_EQ(result, gws[0]);

    err = clEnqueueReadBuffer(m_queue, lws_output, CL_TRUE, 0, sizeof(cl_uint),
                              &result, 0, nullptr, nullptr);
    ASSERT_EQ(err, CL_SUCCESS);
    EXPECT_EQ(result, lws[0]);

    err = clEnqueueReadBuffer(m_queue, dim_output, CL_TRUE, 0, sizeof(cl_uint),
                              &result, 0, nullptr, nullptr);
    ASSERT_EQ(err, CL_SUCCESS);
    EXPECT_EQ(result, 1);
}
