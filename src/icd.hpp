// Copyright 2020 The clvk authors.
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

#include "cl_headers.hpp"

extern cl_icd_dispatch gDispatchTable;

namespace clvk {
struct icd_object {
    cl_icd_dispatch* m_dispatch = &gDispatchTable;
};
} // namespace clvk

struct _cl_platform_id : clvk::icd_object {};
struct _cl_device_id : clvk::icd_object {};
struct _cl_context : clvk::icd_object {};
struct _cl_command_queue : clvk::icd_object {};
struct _cl_program : clvk::icd_object {};
struct _cl_kernel : clvk::icd_object {};
struct _cl_mem : clvk::icd_object {};
struct _cl_sampler : clvk::icd_object {};
struct _cl_event : clvk::icd_object {};
struct _cl_semaphore_khr : clvk::icd_object {};
struct _cl_command_buffer_khr : clvk::icd_object {};
