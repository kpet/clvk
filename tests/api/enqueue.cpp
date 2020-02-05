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

static const unsigned NUM_INSTANCES = 10000;

static const char* program_source = R"(
kernel void test_simple(global uint* out, uint id)
{
    out[id] = id;
}
)";

TEST_F(WithCommandQueue, ManyInstancesInFlight) {
    // Create kernel
    auto kernel = CreateKernel(program_source, "test_simple");

    // Create buffer
    size_t buffer_size = NUM_INSTANCES * sizeof(cl_uint);
    auto buffer = CreateBuffer(CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                               buffer_size, nullptr);

    // Dispatch kernel
    size_t gws = 1;
    size_t lws = 1;

    auto ts_start = sampleTime();
    SetKernelArg(kernel, 0, buffer);
    for (cl_uint i = 0; i < NUM_INSTANCES; i++) {
        SetKernelArg(kernel, 1, &i);
        EnqueueNDRangeKernel(kernel, 1, nullptr, &gws, &lws);
    }
    auto ts_end = sampleTime();
    RecordProperty("enqueue-time", ts_end - ts_start);

    // Complete execution
    Finish();

    // Map the buffer
    auto data =
        EnqueueMapBuffer<cl_uint>(buffer, CL_TRUE, CL_MAP_READ, 0, buffer_size);

    // Check the expected result
    for (cl_uint i = 0; i < NUM_INSTANCES; ++i) {
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
