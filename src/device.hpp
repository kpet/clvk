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

#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "cl_headers.hpp"
#include "utils.hpp"

typedef struct _cl_device_id cvk_device;

typedef struct _cl_device_id {

    _cl_device_id(VkPhysicalDevice pd) : m_pdev(pd) {
        vkGetPhysicalDeviceProperties(m_pdev, &m_properties);
        vkGetPhysicalDeviceMemoryProperties(m_pdev, &m_mem_properties);
    }

    static cvk_device* create(VkPhysicalDevice pdev);

    virtual ~_cl_device_id() {
        vkDestroyDevice(m_dev, nullptr);
    }

    const VkPhysicalDeviceLimits& vulkan_limits() const { return m_properties.limits; }
    const char* name() const { return m_properties.deviceName; }
    uint32_t vendor_id() const { return m_properties.vendorID; }


    CHECK_RETURN uint32_t memory_type_index() const {
        uint32_t memoryTypeIndex = VK_MAX_MEMORY_TYPES;

        for (uint32_t k = 0; k < m_mem_properties.memoryTypeCount; k++) {
            if ((VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT & m_mem_properties.memoryTypes[k].propertyFlags) &&
                (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT & m_mem_properties.memoryTypes[k].propertyFlags)) {
                memoryTypeIndex = k;
                break;
            }
        }

        return memoryTypeIndex;
    }

    uint64_t actual_memory_size() const {
        uint32_t type = memory_type_index();
        auto memprop = m_mem_properties.memoryTypes[type];
        return m_mem_properties.memoryHeaps[memprop.heapIndex].size;
    }

    uint64_t memory_size() const {
        return std::min(max_alloc_size() * 4, actual_memory_size());
    }

    size_t max_alloc_size() const {
        size_t max_buffer_size = m_properties.limits.maxStorageBufferRange;
        return std::min(max_buffer_size, actual_memory_size());
    }

    cl_uint mem_base_addr_align() const {
        return m_mem_base_addr_align;
    }

    std::string version_string() const {
        std::string ret = "Vulkan v";
        ret += vulkan_version_string(m_properties.apiVersion);
        ret += " driver " + std::to_string(m_properties.driverVersion);

        return ret;
    }

    cl_device_type type() const {
        cl_device_type ret;

        switch (m_properties.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            ret = CL_DEVICE_TYPE_GPU;
            break;
        case VK_PHYSICAL_DEVICE_TYPE_CPU:
            ret = CL_DEVICE_TYPE_CPU;
            break;
        case VK_PHYSICAL_DEVICE_TYPE_OTHER:
        default:
            ret = CL_DEVICE_TYPE_CUSTOM;
            break;
        }

        return ret;
    }

    cl_bool has_host_unified_memory() const {
        switch (m_properties.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_CPU:
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            return CL_TRUE;
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        case VK_PHYSICAL_DEVICE_TYPE_OTHER:
        default:
            return CL_FALSE;
        }
    }

    VkQueue vulkan_queue_allocate() {
        VkQueue queue = VK_NULL_HANDLE;
        vkGetDeviceQueue(m_dev, m_vulkan_queue_family, m_vulkan_queue_alloc_index, &queue);

        // Simple round-robin allocation for now
        m_vulkan_queue_alloc_index++;
        if (m_vulkan_queue_alloc_index == m_vulkan_num_queues) {
            m_vulkan_queue_alloc_index = 0;
        }

        return queue;
    }

    uint32_t vulkan_queue_family() const {
        return m_vulkan_queue_family;
    }

    cl_device_fp_config fp_config(cl_device_info fptype) const {
        if (fptype == CL_DEVICE_SINGLE_FP_CONFIG) {
            return CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN;
        }

        if ((fptype == CL_DEVICE_DOUBLE_FP_CONFIG) && m_features.shaderFloat64) {
            return CL_FP_FMA | CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO |
                   CL_FP_ROUND_TO_INF | CL_FP_INF_NAN | CL_FP_DENORM;
        }

        return 0;
    }

    VkPhysicalDevice vulkan_physical_device() const {
        return m_pdev;
    }

    VkDevice vulkan_device() const {
        return m_dev;
    }

private:
    bool init_queues();
    bool init_extensions();
    void init_features();
    bool init();

    VkPhysicalDevice m_pdev;
    VkPhysicalDeviceProperties m_properties;
    VkPhysicalDeviceMemoryProperties m_mem_properties;
    VkPhysicalDeviceFeatures m_features;
    VkDevice m_dev;
    std::vector<const char*> m_vulkan_device_extensions;
    cl_uint m_mem_base_addr_align;

    uint32_t m_vulkan_num_queues;
    uint32_t m_vulkan_queue_alloc_index;
    uint32_t m_vulkan_queue_family;

} cvk_device;

typedef struct _cl_platform_id {
    std::vector<cvk_device*> devices;
} cvk_platform;

