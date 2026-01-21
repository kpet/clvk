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

#pragma once

#define CL_TARGET_OPENCL_VERSION 300
#define CL_USE_DEPRECATED_OPENCL_1_0_APIS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include "CL/cl.h"
#include "CL/cl_ext.h"
#include "CL/cl_half.h"
#include "CL/cl_icd.h"

// cl_ext_buffer_device_address extension definitions
// These will be provided by OpenCL-Headers once the extension is released
#ifndef cl_ext_buffer_device_address
#define cl_ext_buffer_device_address 1

typedef cl_ulong cl_mem_device_address_ext;

#define CL_MEM_DEVICE_PRIVATE_ADDRESS_EXT           0x4200
#define CL_MEM_DEVICE_ADDRESS_EXT                   0x4201
#define CL_KERNEL_EXEC_INFO_DEVICE_PTRS_EXT         0x11B8

typedef cl_int (CL_API_CALL *clSetKernelArgDevicePointerEXT_fn)(
    cl_kernel kernel,
    cl_uint arg_index,
    cl_mem_device_address_ext arg_value);

#endif // cl_ext_buffer_device_address
