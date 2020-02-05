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
kernel void test_simple(global uint* out)
{
    size_t gid = get_global_id(0);
    out[gid] = gid;
}
)";

TEST_F(WithCommandQueue, Simple) {
    // Create kernel
    auto kernel = CreateKernel(program_source, "test_simple");

    // Create buffer
    auto buffer = CreateBuffer(CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                               BUFFER_SIZE, nullptr);

    // Dispatch kernel
    size_t gws = BUFFER_SIZE / sizeof(cl_int);
    size_t lws = 1;

    SetKernelArg(kernel, 0, buffer);
    EnqueueNDRangeKernel(kernel, 1, nullptr, &gws, &lws);

    // Complete execution
    Finish();

    // Map the buffer
    auto data = EnqueueMapBuffer<cl_uint>(buffer, CL_TRUE, CL_MAP_READ, 0,
                                          BUFFER_SIZE);

    // Check the expected result
    for (cl_uint i = 0; i < BUFFER_SIZE / sizeof(cl_uint); ++i) {
        EXPECT_EQ(data[i], static_cast<cl_uint>(i));
        if (data[i] != static_cast<cl_uint>(i)) {
            printf("Failed comparison at data[%u]: expected %u != got %u\n", i,
                   i, data[i]);
        }
    }

    // Unmap the buffer
    EnqueueUnmapMemObject(buffer, data);
    Finish();
}
