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

TEST_F(WithCommandQueue, ManyInstancesInFlight) {

    static const unsigned NUM_INSTANCES = 10000;

    static const char* program_source = R"(
    kernel void test_simple(global uint* out, uint id)
    {
        out[id] = id;
    }
    )";

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

TEST_F(WithCommandQueue, KernelNoArguments) {
    static const char* program_source = "kernel void test_noargs(){}";

    // Create kernel
    auto kernel = CreateKernel(program_source, "test_noargs");

    // Dispatch kernel
    size_t gws = 1;
    size_t lws = 1;
    cl_event event;
    EnqueueNDRangeKernel(kernel, 1, nullptr, &gws, &lws, 0, nullptr, &event);

    // Complete execution
    Finish();

    // Check the kernel ran successfully
    cl_int status;
    GetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, &status);
    clReleaseEvent(event);
    ASSERT_EQ(status, CL_COMPLETE);
}

TEST_F(WithCommandQueue, WorkDim) {

    static const char* program_source = R"(
    kernel void test_work_dim(global uint* out, uint id)
    {
        out[id] = get_work_dim();
    }
    )";

    // Create kernel
    auto kernel = CreateKernel(program_source, "test_work_dim");

    // Create buffer
    size_t buffer_size = sizeof(cl_uint) * 3;
    auto buffer = CreateBuffer(CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                               buffer_size, nullptr);

    // Dispatch kernel
    auto ts_start = sampleTime();
    SetKernelArg(kernel, 0, buffer);
    const size_t gws[3] = {1, 1, 1};
    const size_t lws[3] = {1, 1, 1};
    for (cl_uint i = 0; i < 3; ++i) {
        cl_uint work_dim = i + 1;
        SetKernelArg(kernel, 1, &i);
        EnqueueNDRangeKernel(kernel, work_dim, nullptr, gws, lws);
    }
    auto ts_end = sampleTime();
    RecordProperty("enqueue-time", ts_end - ts_start);

    // Complete execution
    Finish();

    // Map the buffer
    auto data =
        EnqueueMapBuffer<cl_uint>(buffer, CL_TRUE, CL_MAP_READ, 0, buffer_size);

    // Check the expected result
    for (cl_uint i = 0; i < 3; ++i) {
        EXPECT_EQ(data[i], static_cast<cl_uint>(i + 1))
            << "Failure comparison at data[" << i << "]:\n - expected " << i + 1
            << "\n - got: " << data[i];
    }
}

TEST_F(WithCommandQueue, PodUBO) {
    static const char* program_source =
        "kernel void test(global int* out, int a, int4 b, int c) { *out = a + "
        "b.x + c; }";

    auto kernel = CreateKernel(program_source, " -pod-ubo ", "test");

    // Output buffer
    size_t buffer_size = sizeof(cl_uint);
    auto buffer = CreateBuffer(CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                               buffer_size, nullptr);
    SetKernelArg(kernel, 0, buffer);

    // Pod args
    cl_int a = 42;
    cl_int b[4] = {-1, -2, -3, -4};
    cl_int c = -3;
    SetKernelArg(kernel, 1, &a);
    SetKernelArg(kernel, 2, 4 * sizeof(cl_int), b);
    SetKernelArg(kernel, 3, &c);

    size_t gws = 1;
    size_t lws = 1;
    EnqueueNDRangeKernel(kernel, 1, nullptr, &gws, &lws);

    // Map the buffer
    auto data =
        EnqueueMapBuffer<cl_uint>(buffer, CL_TRUE, CL_MAP_READ, 0, buffer_size);

    EXPECT_EQ(data[0], static_cast<cl_int>(38));
}

TEST_F(WithCommandQueue, PodPushConstant) {
    static const char* program_source =
        "kernel void test(global int* out, int a, int4 b, int c) { *out = a + "
        "b.x + c; }";

    auto kernel = CreateKernel(program_source, " -pod-pushconstant ", "test");

    // Output buffer
    size_t buffer_size = sizeof(cl_uint);
    auto buffer = CreateBuffer(CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                               buffer_size, nullptr);
    SetKernelArg(kernel, 0, buffer);

    // Pod args
    cl_int a = 42;
    cl_int b[4] = {1, 2, 3, 4};
    cl_int c = -3;
    SetKernelArg(kernel, 1, &a);
    SetKernelArg(kernel, 2, 4 * sizeof(cl_int), b);
    SetKernelArg(kernel, 3, &c);

    size_t gws = 1;
    size_t lws = 1;
    EnqueueNDRangeKernel(kernel, 1, nullptr, &gws, &lws);

    // Map the buffer
    auto data =
        EnqueueMapBuffer<cl_uint>(buffer, CL_TRUE, CL_MAP_READ, 0, buffer_size);

    EXPECT_EQ(data[0], static_cast<cl_int>(40));
}
