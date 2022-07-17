// Copyright 2022 The clvk authors.
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

#include <unordered_set>

TEST(Platform, HasOneDefaultDevice) {
    cl_int err;
    cl_uint num_devices;
    err =
        clGetDeviceIDs(gPlatform, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
    ASSERT_EQ(err, CL_SUCCESS);

    std::vector<cl_device_id> devices(num_devices);

    err = clGetDeviceIDs(gPlatform, CL_DEVICE_TYPE_ALL, num_devices,
                         devices.data(), nullptr);
    ASSERT_EQ(err, CL_SUCCESS);

    unsigned num_default = 0;

    for (auto dev : devices) {
        cl_device_type dtype;
        err = clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof(dtype), &dtype,
                              nullptr);
        ASSERT_EQ(err, CL_SUCCESS);

        if (dtype & CL_DEVICE_TYPE_DEFAULT) {
            num_default++;
        }
    }

    ASSERT_EQ(num_default, 1);
}

TEST(Platform, DeviceQueryWithMultipleTypes) {
    cl_int err;

    // Get the type of the already selected device
    cl_device_type dtype;
    err = clGetDeviceInfo(gDevice, CL_DEVICE_TYPE, sizeof(dtype), &dtype,
                          nullptr);
    ASSERT_EQ(err, CL_SUCCESS);

    // Check its type is one of the expected values
    dtype &= ~CL_DEVICE_TYPE_DEFAULT;
    ASSERT_TRUE(dtype == CL_DEVICE_TYPE_GPU || dtype == CL_DEVICE_TYPE_CPU ||
                dtype == CL_DEVICE_TYPE_ACCELERATOR);

    std::unordered_set<cl_device_type> other_types = {
        CL_DEVICE_TYPE_DEFAULT, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU,
        CL_DEVICE_TYPE_ACCELERATOR, CL_DEVICE_TYPE_CUSTOM};

    other_types.erase(dtype);

    // Check that the device can still be enumerated when the requested device
    // type specifies other device types as well
    for (auto otype : other_types) {
        cl_device_id device;
        err = clGetDeviceIDs(gPlatform, dtype | otype, 1, &device, nullptr);
        ASSERT_EQ(err, CL_SUCCESS);
    }
}
