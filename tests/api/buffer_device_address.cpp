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

// Extension definitions
#ifndef cl_ext_buffer_device_address
#define cl_ext_buffer_device_address 1
#define CL_DEVICE_PTR_EXT 0xff01
#define CL_MEM_DEVICE_ADDRESS_EXT (1ul << 31)
#define CL_MEM_DEVICE_PTR_EXT 0xff01
typedef cl_ulong cl_mem_device_address_EXT;
typedef cl_int(CL_API_CALL *clSetKernelArgDevicePointerEXT_fn)(
    cl_kernel kernel, cl_uint arg_index, cl_mem_device_address_EXT dev_addr);
#endif

TEST_F(WithCommandQueue, BufferDeviceAddress) {
    // First check if device supports the extension
    size_t ext_size;
    GetDeviceInfo(CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_size);
    
    std::vector<char> extensions(ext_size);
    GetDeviceInfo(CL_DEVICE_EXTENSIONS, ext_size, extensions.data(), nullptr);

    bool hasBufferDeviceAddress = 
        std::string(extensions.data()).find("cl_ext_buffer_device_address") != std::string::npos;

    if (!hasBufferDeviceAddress) {
        GTEST_SKIP() << "Device does not support cl_ext_buffer_device_address extension";
    }

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
    auto buffer = CreateBuffer(CL_MEM_READ_WRITE | CL_MEM_DEVICE_ADDRESS_EXT,
                             BUFFER_SIZE, nullptr);

    // Get device pointer
    cl_mem_device_address_EXT device_ptr;
    cl_int err = clGetMemObjectInfo(buffer, CL_MEM_DEVICE_PTR_EXT, 
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
