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

static const size_t BUFFER_SIZE = 1024;

static const char* program_source = R"(
kernel void test_simple(global uint4* out, constant uint4* c_data)
{
    size_t gid = get_global_id(0);
    out[gid] = (uint4)(gid,gid,gid,gid) + c_data[gid];
}
)";

TEST_F(WithCommandQueue, SimpleUBO) {
    // Create kernel
    auto kernel = CreateKernel(program_source,
                               " -constant-args-ubo -inline-entry-points ",
                               "test_simple");

    auto num_items = BUFFER_SIZE / sizeof(cl_uint);
    cl_int c_data[num_items];
    for (auto i = 0; i != num_items; ++i) {
        c_data[i] = 1;
    }
    auto c_buffer = CreateBuffer(CL_MEM_READ_ONLY, BUFFER_SIZE, nullptr);

    EnqueueWriteBuffer(c_buffer, CL_TRUE, 0, BUFFER_SIZE, c_data);

    // Complete execution
    Finish();

    // Create buffer
    auto buffer = CreateBuffer(CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                               BUFFER_SIZE, nullptr);

    // Dispatch kernel
    size_t gws = num_items / 4;
    size_t lws = 2;

    SetKernelArg(kernel, 0, buffer);
    SetKernelArg(kernel, 1, c_buffer);
    EnqueueNDRangeKernel(kernel, 1, nullptr, &gws, &lws);

    // Complete execution
    Finish();

    // Map the buffer
    auto data =
        EnqueueMapBuffer<cl_int>(buffer, CL_TRUE, CL_MAP_READ, 0, BUFFER_SIZE);

    // Check the expected result
    bool success = true;
    for (cl_uint i = 0; i < BUFFER_SIZE / sizeof(cl_uint4); ++i) {
        auto expected = i / 4 + 1;
        if (data[i] != static_cast<cl_uint>(expected)) {
            printf("Failed comparison at data[%u]: expected %u but got %u\n", i,
                   expected, data[i]);
            success = false;
        }
    }

    EXPECT_TRUE(success);

    // Unmap the buffer
    EnqueueUnmapMemObject(buffer, data);
    Finish();
}
