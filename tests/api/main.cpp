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

#include <cstdlib>
#include <iostream>

#include "testcl.hpp"

#define RETURN_ON_CL_FAILURE(err, ret) do {                               \
    if (err != CL_SUCCESS) {                                              \
        printf("%s:%d Error after CL call: %d (%s)\n", __FILE__, __LINE__,\
        err, cl_code_to_string(err));                                     \
        return ret;                                                       \
    }                                                                     \
    } while (0)

cl_device_id get_device() {
    cl_platform_id platform;
    cl_device_id device;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, nullptr);
    RETURN_ON_CL_FAILURE(err, nullptr);

    size_t platform_name_len;
    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0 , nullptr,
                            &platform_name_len);
    RETURN_ON_CL_FAILURE(err, nullptr);

    std::string platform_name(platform_name_len, ' ');
    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, platform_name.size(),
                            const_cast<char*>(platform_name.data()), nullptr);
    RETURN_ON_CL_FAILURE(err, nullptr);

    std::cout << "Platform: " << platform_name << std::endl;

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, nullptr);
    RETURN_ON_CL_FAILURE(err, nullptr);

    size_t device_name_len;
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &device_name_len);
    RETURN_ON_CL_FAILURE(err, nullptr);

    std::string device_name(device_name_len, ' ');
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, device_name.size(),
                          const_cast<char*>(device_name.data()), nullptr);
    RETURN_ON_CL_FAILURE(err, nullptr);

    std::cout << "Device: " << device_name << std::endl;

    return device;
}

cl_device_id gDevice;

int main(int argc, char* argv[]) {

    ::testing::InitGoogleTest(&argc, argv);

    // Select device
    gDevice = get_device();
    if (gDevice == nullptr) {
        std::cerr << "Couldn't find an OpenCL device\n";
        std::exit(EXIT_FAILURE);
    }

    return RUN_ALL_TESTS();
}
