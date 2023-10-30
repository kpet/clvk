// Copyright 2021 The clvk authors.
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

#include "device.hpp"
#include "log.hpp"

#include <vulkan/vulkan.h>

extern "C" {
void CL_API_CALL clvk_override_device_max_compute_work_group_count(
    cl_device_id device, uint32_t x, uint32_t y, uint32_t z) {
#ifdef CLVK_UNIT_TESTING_ENABLED
    cvk_debug_fn("x: %u, y: %u, z: %u\n", x, y, z);
    assert(device != nullptr && icd_downcast(device)->is_valid());
    auto& vklimits = icd_downcast(device)->vulkan_limits_writable();

    vklimits.maxComputeWorkGroupCount[0] = x;
    vklimits.maxComputeWorkGroupCount[1] = y;
    vklimits.maxComputeWorkGroupCount[2] = z;
#endif
}

void CL_API_CALL clvk_restore_device_properties(cl_device_id device) {
#ifdef CLVK_UNIT_TESTING_ENABLED
    cvk_debug_fn("device: %p\n", (void*)device);
    assert(device != nullptr && icd_downcast(device)->is_valid());

    icd_downcast(device)->restore_device_properties();
#endif
}

void CL_API_CALL clvk_override_printf_buffer_size(uint32_t size) {
#ifdef CLVK_UNIT_TESTING_ENABLED
    auto printf_buffer_size =
        (config_value<uint32_t>*)&config.printf_buffer_size;
    printf_buffer_size->value = size;
    printf_buffer_size->set = true;
#endif
}
}
