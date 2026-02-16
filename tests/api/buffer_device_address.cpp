// Copyright 2024 The clvk authors.
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
#include "cl_headers.hpp"
#include <cmath>

TEST_F(WithCommandQueue, BufferDeviceAddress) {
    REQUIRE_EXTENSION("cl_ext_buffer_device_address");

    // Get the extension function pointer
    auto clSetKernelArgDevicePointerEXT = 
        (clSetKernelArgDevicePointerEXT_fn)
        clGetExtensionFunctionAddressForPlatform(platform(), "clSetKernelArgDevicePointerEXT");
    
    ASSERT_NE(clSetKernelArgDevicePointerEXT, nullptr) 
        << "Failed to get clSetKernelArgDevicePointerEXT function pointer";

    // Kernel that increments each element
    static const char* program_source = R"(
    kernel void increment(global int* input) {
        int gid = get_global_id(0);
        input[gid] += 1;
    }
    )";

    const size_t BUFFER_SIZE = 1024 * sizeof(cl_int);
    const size_t NUM_ELEMENTS = BUFFER_SIZE / sizeof(cl_int);

    // Create buffer with device address extension flag
    cl_mem_properties props[] = {
        CL_MEM_DEVICE_PRIVATE_ADDRESS_EXT, CL_TRUE,
        0
    };
    cl_int err;
    auto buffer = clCreateBufferWithProperties(m_context, props, CL_MEM_READ_WRITE,
                                              BUFFER_SIZE, nullptr, &err);
    ASSERT_CL_SUCCESS(err);

    // Get device pointer
    cl_mem_device_address_ext device_ptr;
    err = clGetMemObjectInfo(buffer, CL_MEM_DEVICE_ADDRESS_EXT, 
                            sizeof(device_ptr), &device_ptr, nullptr);
    ASSERT_EQ(err, CL_SUCCESS) << "Failed to get device pointer";

    // Initialize buffer with test data
    std::vector<cl_int> host_data(NUM_ELEMENTS, 1);  // Initialize with 1s
    EnqueueWriteBuffer(buffer, CL_TRUE, 0, BUFFER_SIZE, host_data.data());

    // Create and build program
    auto kernel = CreateKernel(program_source, "increment");

    // Set kernel argument using the device pointer
    err = clSetKernelArgDevicePointerEXT(kernel, 0, device_ptr);
    ASSERT_EQ(err, CL_SUCCESS) << "Failed to set kernel argument with device pointer";

    // Execute kernel
    size_t gws = NUM_ELEMENTS;
    EnqueueNDRangeKernel(kernel, 1, nullptr, &gws, nullptr);

    // Complete execution
    Finish();

    // Read back and verify results
    std::vector<cl_int> result(NUM_ELEMENTS);
    EnqueueReadBuffer(buffer, CL_TRUE, 0, BUFFER_SIZE, result.data());

    // Verify each element was incremented by 1
    bool success = true;
    for (size_t i = 0; i < NUM_ELEMENTS; i++) {
        if (result[i] != 2) {  // Should be 1 + 1
            printf("Failed comparison at data[%zu]: expected 2 but got %d\n", 
                   i, result[i]);
            success = false;
        }
    }

    EXPECT_TRUE(success);
}

TEST_F(WithCommandQueue, BufferDeviceAddressMatrixMultiply) {
    REQUIRE_EXTENSION("cl_ext_buffer_device_address");

    // Get the extension function pointer
    auto clSetKernelArgDevicePointerEXT = 
        (clSetKernelArgDevicePointerEXT_fn)
        clGetExtensionFunctionAddressForPlatform(platform(), "clSetKernelArgDevicePointerEXT");
    
    ASSERT_NE(clSetKernelArgDevicePointerEXT, nullptr);

    // Matrix multiply kernel
    static const char* program_source = R"(
    kernel void matmul(
        global const float* A,
        global const float* B,
        global float* C,
        int M, int N, int K)
    {
        int row = get_global_id(0);
        int col = get_global_id(1);
        
        if (row >= M || col >= N) return;
        
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
    )";

    // Small matrices for testing: 64x64
    const int M = 64, N = 64, K = 64;
    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);

    // Create buffers with device address
    cl_mem_properties props[] = {
        CL_MEM_DEVICE_PRIVATE_ADDRESS_EXT, CL_TRUE,
        0
    };
    cl_int err;

    auto bufA = clCreateBufferWithProperties(m_context, props, CL_MEM_READ_ONLY, size_A, nullptr, &err);
    ASSERT_CL_SUCCESS(err);
    
    auto bufB = clCreateBufferWithProperties(m_context, props, CL_MEM_READ_ONLY, size_B, nullptr, &err);
    ASSERT_CL_SUCCESS(err);
    
    auto bufC = clCreateBufferWithProperties(m_context, props, CL_MEM_WRITE_ONLY, size_C, nullptr, &err);
    ASSERT_CL_SUCCESS(err);

    // Get device addresses
    cl_mem_device_address_ext addrA, addrB, addrC;
    err = clGetMemObjectInfo(bufA, CL_MEM_DEVICE_ADDRESS_EXT, sizeof(addrA), &addrA, nullptr);
    ASSERT_CL_SUCCESS(err);
    err = clGetMemObjectInfo(bufB, CL_MEM_DEVICE_ADDRESS_EXT, sizeof(addrB), &addrB, nullptr);
    ASSERT_CL_SUCCESS(err);
    err = clGetMemObjectInfo(bufC, CL_MEM_DEVICE_ADDRESS_EXT, sizeof(addrC), &addrC, nullptr);
    ASSERT_CL_SUCCESS(err);

    // Initialize matrices
    std::vector<float> hostA(M * K), hostB(K * N), hostC(M * N, 0.0f);
    for (int i = 0; i < M * K; i++) hostA[i] = static_cast<float>(i % 10) * 0.1f;
    for (int i = 0; i < K * N; i++) hostB[i] = static_cast<float>(i % 10) * 0.1f;

    EnqueueWriteBuffer(bufA, CL_TRUE, 0, size_A, hostA.data());
    EnqueueWriteBuffer(bufB, CL_TRUE, 0, size_B, hostB.data());

    // Build and run kernel
    auto kernel = CreateKernel(program_source, "matmul");
    
    err = clSetKernelArgDevicePointerEXT(kernel, 0, addrA);
    ASSERT_CL_SUCCESS(err);
    err = clSetKernelArgDevicePointerEXT(kernel, 1, addrB);
    ASSERT_CL_SUCCESS(err);
    err = clSetKernelArgDevicePointerEXT(kernel, 2, addrC);
    ASSERT_CL_SUCCESS(err);
    
    int m = M, n = N, k = K;
    clSetKernelArg(kernel, 3, sizeof(int), &m);
    clSetKernelArg(kernel, 4, sizeof(int), &n);
    clSetKernelArg(kernel, 5, sizeof(int), &k);

    size_t gws[2] = {static_cast<size_t>(M), static_cast<size_t>(N)};
    EnqueueNDRangeKernel(kernel, 2, nullptr, gws, nullptr);
    Finish();

    // Read results
    EnqueueReadBuffer(bufC, CL_TRUE, 0, size_C, hostC.data());

    // CPU reference computation
    std::vector<float> cpuC(M * N, 0.0f);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int kk = 0; kk < K; kk++) {
                sum += hostA[i * K + kk] * hostB[kk * N + j];
            }
            cpuC[i * N + j] = sum;
        }
    }

    // Verify results
    bool success = true;
    const float epsilon = 1e-3f;
    for (int i = 0; i < M * N; i++) {
        if (std::fabs(hostC[i] - cpuC[i]) > epsilon) {
            printf("Mismatch at %d: GPU=%f CPU=%f\n", i, hostC[i], cpuC[i]);
            success = false;
            if (i >= 10) break;
        }
    }

    EXPECT_TRUE(success) << "Matrix multiply results don't match CPU reference";

    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
}
