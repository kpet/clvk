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

TEST_F(WithCommandQueue, KernelInvalidArguments) {
    static const char* program_source = R"(
    kernel void test_invalid_args(global uint* out, uint id)
    {
        out[id] = id;
    }
    )";

    // Create kernel
    auto kernel = CreateKernel(program_source, "test_invalid_args");

    // Create buffer
    auto buffer = CreateBuffer(CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                               sizeof(cl_uint), nullptr);

    size_t gws = 1;
    size_t lws = 1;
    cl_int err;

    // Try to enqueue the kernel, expecting it to fail
    err = clEnqueueNDRangeKernel(m_queue, kernel, 1, nullptr, &gws, &lws, 0,
                                 nullptr, nullptr);
    EXPECT_EQ(err, CL_INVALID_KERNEL_ARGS);

    // Set the first argument, but not the second
    SetKernelArg(kernel, 0, buffer);

    // Try to enqueue the kernel, still expecting it to fail
    err = clEnqueueNDRangeKernel(m_queue, kernel, 1, nullptr, &gws, &lws, 0,
                                 nullptr, nullptr);
    EXPECT_EQ(err, CL_INVALID_KERNEL_ARGS);

    // Set the second argument
    cl_uint arg_value = 0;
    SetKernelArg(kernel, 1, &arg_value);

    // Try to enqueue the kernel, should now succeed
    err = clEnqueueNDRangeKernel(m_queue, kernel, 1, nullptr, &gws, &lws, 0,
                                 nullptr, nullptr);
    EXPECT_EQ(err, CL_SUCCESS);

    Finish();
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

    // Unmap the buffer
    EnqueueUnmapMemObject(buffer, data);
    Finish();
}

TEST_F(WithCommandQueue, PodUBO) {
    static const char* program_source =
        "kernel void test(global int* out, int a, int4 b, int c) { *out = a + "
        "b.y + c; }";

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

    // Complete execution
    Finish();

    // Map the buffer
    auto data =
        EnqueueMapBuffer<cl_uint>(buffer, CL_TRUE, CL_MAP_READ, 0, buffer_size);

    EXPECT_EQ(data[0], static_cast<cl_int>(37));

    // Unmap the buffer
    EnqueueUnmapMemObject(buffer, data);
    Finish();
}

TEST_F(WithCommandQueue, PodPushConstant) {
    static const char* program_source =
        "kernel void test(global int* out, int a, int4 b, int c) { *out = a + "
        "b.z + c; }";

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

    // Complete execution
    Finish();

    // Map the buffer
    auto data =
        EnqueueMapBuffer<cl_uint>(buffer, CL_TRUE, CL_MAP_READ, 0, buffer_size);

    EXPECT_EQ(data[0], static_cast<cl_int>(42));

    // Unmap the buffer
    EnqueueUnmapMemObject(buffer, data);
    Finish();
}

TEST_F(WithCommandQueue, OffsetSpecConstant) {
    static const std::string program_source = R"(
kernel void test(global int* out) {
  out[0] = get_global_offset(0);
  out[1] = get_global_offset(1);
  out[2] = get_global_offset(2);
}
)";

    auto kernel = CreateKernel(program_source.c_str(), "test");

    // Output buffer
    size_t buffer_size = 3 * sizeof(cl_uint);
    auto buffer = CreateBuffer(CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                               buffer_size, nullptr);
    SetKernelArg(kernel, 0, buffer);

    size_t gws[3] = {1, 1, 1};
    size_t lws[3] = {1, 1, 1};
    size_t offset[3] = {123, 234, 345};
    EnqueueNDRangeKernel(kernel, 3, offset, gws, lws);

    // Complete execution
    Finish();

    // Map the buffer
    auto data =
        EnqueueMapBuffer<cl_uint>(buffer, CL_TRUE, CL_MAP_READ, 0, buffer_size);

    EXPECT_EQ(data[0], static_cast<cl_int>(123));
    EXPECT_EQ(data[1], static_cast<cl_int>(234));
    EXPECT_EQ(data[2], static_cast<cl_int>(345));

    // Unmap the buffer
    EnqueueUnmapMemObject(buffer, data);
    Finish();
}

TEST_F(WithCommandQueue, OffsetPushConstant) {
    static const std::string program_source = R"(
kernel void test(global int* out) {
  out[0] = get_global_offset(0);
  out[1] = get_global_offset(1);
  out[2] = get_global_offset(2);
}
)";

    auto kernel = CreateKernel(program_source.c_str(),
                               " -global-offset-push-constant ", "test");

    // Output buffer
    size_t buffer_size = 3 * sizeof(cl_uint);
    auto buffer = CreateBuffer(CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                               buffer_size, nullptr);
    SetKernelArg(kernel, 0, buffer);

    size_t gws[3] = {1, 1, 1};
    size_t lws[3] = {1, 1, 1};
    size_t offset[3] = {101, 202, 303};
    EnqueueNDRangeKernel(kernel, 3, offset, gws, lws);

    // Complete execution
    Finish();

    // Map the buffer
    auto data =
        EnqueueMapBuffer<cl_uint>(buffer, CL_TRUE, CL_MAP_READ, 0, buffer_size);

    EXPECT_EQ(data[0], static_cast<cl_int>(101));
    EXPECT_EQ(data[1], static_cast<cl_int>(202));
    EXPECT_EQ(data[2], static_cast<cl_int>(303));

    // Unmap the buffer
    EnqueueUnmapMemObject(buffer, data);
    Finish();
}

TEST_F(WithCommandQueue, BindingGreaterThanNumberOfResources) {
    static const std::string program_source = R"(
kernel void k0(int v, local int *, global int* b){}
kernel void k1(global int* b){ *b = 77; }
)";
    // XXX this test assumes that k1's argument b gets assigned binding 2
    auto kernel = CreateKernel(program_source.c_str(), "k1");

    // Output buffer
    size_t buffer_size = sizeof(cl_uint);
    auto buffer = CreateBuffer(CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                               buffer_size, nullptr);
    SetKernelArg(kernel, 0, buffer);

    size_t gws = 1;
    EnqueueNDRangeKernel(kernel, 1, nullptr, &gws, nullptr);

    // Complete execution
    Finish();

    // Map the buffer
    auto data =
        EnqueueMapBuffer<cl_uint>(buffer, CL_TRUE, CL_MAP_READ, 0, buffer_size);

    EXPECT_EQ(data[0], static_cast<cl_int>(77));

    // Unmap the buffer
    EnqueueUnmapMemObject(buffer, data);
    Finish();
}
