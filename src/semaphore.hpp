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

#pragma once

#include "cl_headers.hpp"
#include "context.hpp"
#include "device.hpp"
#include "icd.hpp"
#include "objects.hpp"
#include "utils.hpp"

#include <vector>

struct cvk_semaphore : public _cl_semaphore_khr,
                       api_object<object_magic::semaphore> {
    cvk_semaphore(cvk_context* context, cl_semaphore_type_khr type,
                  std::vector<cl_device_id>&& devices,
                  std::vector<cl_semaphore_properties_khr>&& properties)
        : api_object(context), m_type(type), m_devices(std::move(devices)),
          m_properties(std::move(properties)), m_semaphore(VK_NULL_HANDLE) {}

    CHECK_RETURN cl_int init();

    virtual ~cvk_semaphore() {
        if (m_semaphore != VK_NULL_HANDLE) {
            auto vkdev = m_context->device()->vulkan_device();
            vkDestroySemaphore(vkdev, m_semaphore, nullptr);
        }
    }
    cl_semaphore_type_khr type() const { return m_type; }
    const std::vector<cl_semaphore_properties_khr>& properties() const {
        return m_properties;
    }
    const std::vector<cl_device_id>& devices() const { return m_devices; }
    bool can_be_used_with_device(const cvk_device* device) const {
        for (auto devapi : m_devices) {
            auto dev = static_cast<cvk_device*>(devapi);
            if (device == dev) {
                return true;
            }
        }
        if (m_context->has_device(device)) {
            return true;
        }
        return false;
    }
    bool requires_payload() const {
        return false; // Binary semaphores are the only type supported and they
                      // don't require a payload
    }
    cl_semaphore_payload_khr payload() const {
        return 0; // TODO return 1 when signaled
    }

private:
    cl_semaphore_type_khr m_type;
    std::vector<cl_device_id> m_devices;
    std::vector<cl_semaphore_properties_khr> m_properties;
    VkSemaphore m_semaphore;
};

static inline cvk_semaphore* icd_downcast(cl_semaphore_khr sem) {
    return static_cast<cvk_semaphore*>(sem);
}
