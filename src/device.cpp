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


#include "device.hpp"
#include "memory.hpp"
#include "utils.hpp"

cvk_device* cvk_device::create(VkPhysicalDevice pdev)
{
    cvk_device *device = new cvk_device(pdev);

    if (!device->init()) {
        delete device;
        return nullptr;
    }

    return device;
}

bool cvk_device::init_queues(uint32_t *num_queues, uint32_t *queue_family)
{
    // Get number of queue families
    uint32_t num_families;
    vkGetPhysicalDeviceQueueFamilyProperties(m_pdev, &num_families, nullptr);

    cvk_info_fn("physical device (%s) has %u queue families:",
             vulkan_physical_device_type_string(m_properties.deviceType).c_str(),
             num_families);

    // Get their properties
    std::vector<VkQueueFamilyProperties> families(num_families);
    vkGetPhysicalDeviceQueueFamilyProperties(m_pdev, &num_families, families.data());

    // Look for suitable queues
    bool found_queues = false;
    *num_queues = 0;
    for (uint32_t i = 0; i < num_families; i++) {

        cvk_info_fn("queue family %u: %2u queues | %s", i,
                families[i].queueCount,
                vulkan_queue_flags_string(families[i].queueFlags).c_str()
        );

        if (!found_queues && (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            *queue_family = i;
            *num_queues = families[i].queueCount;
            found_queues = true;
        }
    }

    if (!found_queues) {
        cvk_error("Could not find a suitable queue family for this device");
        return false;
    }

    // Initialise the queue allocator
    m_vulkan_queue_alloc_index = 0;

    return true;
}

bool cvk_device::init_extensions()
{
    uint32_t numext;
    VkResult res = vkEnumerateDeviceExtensionProperties(m_pdev, nullptr, &numext, nullptr);
    CVK_VK_CHECK_ERROR_RET(res, false, "Failed to get the number of device extension properties");

    cvk_info("%u device extension properties reported.", numext);

    std::vector<VkExtensionProperties> extensions(numext);
    res = vkEnumerateDeviceExtensionProperties(m_pdev, nullptr, &numext, extensions.data());
    CVK_VK_CHECK_ERROR_RET(res, false, "Could not enumerate device extension properties");

    m_vulkan_device_extensions = {
        "VK_KHR_storage_buffer_storage_class",
    };

    const std::vector<const char *> desired_extensions = {
        "VK_KHR_16bit_storage",
        "VK_KHR_variable_pointers",
    };

    for (size_t i = 0; i < numext; i++) {
        cvk_info("Found extension %s, spec version %u",
                 extensions[i].extensionName,
                 extensions[i].specVersion);

        for (auto de : desired_extensions) {
            if (!strcmp(de, extensions[i].extensionName)) {
                m_vulkan_device_extensions.push_back(de);
                cvk_info_fn("found extension %s, enabling", de);
            }
        }
    }

    return true;
}

void cvk_device::init_features()
{
    // Query supported features.
    VkPhysicalDeviceFeatures2 supported_features;
    VkPhysicalDeviceVariablePointerFeatures supported_features_variable_pointers;
    supported_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    supported_features.pNext = &supported_features_variable_pointers;
    vkGetPhysicalDeviceFeatures2(m_pdev, &supported_features);

    memset(&m_features, 0, sizeof(m_features));
    memset(&m_features_variable_pointers, 0,
           sizeof(m_features_variable_pointers));

    if (supported_features.features.shaderInt16) {
        m_features.shaderInt16 = VK_TRUE;
    }
    if (supported_features.features.shaderInt64) {
        m_features.shaderInt64 = VK_TRUE;
    }
    if (supported_features.features.shaderFloat64) {
        m_features.shaderFloat64 = VK_TRUE;
    }
    if (supported_features.features.shaderStorageImageWriteWithoutFormat) {
        m_features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    }

    if (supported_features_variable_pointers.variablePointers) {
        m_features_variable_pointers.variablePointers = VK_TRUE;
    }
    if (supported_features_variable_pointers.variablePointersStorageBuffer) {
        m_features_variable_pointers.variablePointersStorageBuffer = VK_TRUE;
    }
}

bool cvk_device::init()
{
    cvk_info_fn("Initialising device %s", m_properties.deviceName);

    uint32_t num_queues, queue_family;
    if (!init_queues(&num_queues, &queue_family)) {
        return false;
    }

    if (!init_extensions()) {
        return false;
    }

    init_features();

    // Give all queues the same priority
    std::vector<float> queuePriorities(num_queues, 1.0f);

    VkDeviceQueueCreateInfo queueCreateInfo = {
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        nullptr,
        0, //flags
        queue_family,
        num_queues, // queueCount
        queuePriorities.data()
    };


    // Create logical device
    const VkDeviceCreateInfo createInfo = {
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, // sType
        nullptr, // pNext
        0, // flags
        1, // queueCreateInfoCount
        &queueCreateInfo, // pQueueCreateInfos,
        0, // enabledLayerCount
        nullptr, // ppEnabledLayerNames
        static_cast<uint32_t>(m_vulkan_device_extensions.size()), // enabledExtensionCount
        m_vulkan_device_extensions.data(), // ppEnabledExtensionNames
        &m_features, // pEnabledFeatures
    };

    VkResult res = vkCreateDevice(m_pdev, &createInfo, nullptr, &m_dev);
    CVK_VK_CHECK_ERROR_RET(res, false, "Failed to create a device");

    // Construct the queue wrappers now that our queues exist
    for (auto i = 0U; i < num_queues; i++) {
        VkQueue queue;

        vkGetDeviceQueue(m_dev, queue_family, i, &queue);
        m_vulkan_queues.emplace_back(cvk_vulkan_queue_wrapper(queue, queue_family));
    }

    // Work out the required alignment for buffers
    const VkBufferCreateInfo bufferCreateInfo = {
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, // sType
        nullptr, // pNext
        0, // flags
        1, // size
        cvk_buffer::USAGE_FLAGS, // usage
        VK_SHARING_MODE_EXCLUSIVE,
        0, // queueFamilyIndexCount
        nullptr, // pQueueFamilyIndices
    };

    VkBuffer buffer;
    res = vkCreateBuffer(m_dev, &bufferCreateInfo, nullptr, &buffer);
    CVK_VK_CHECK_ERROR_RET(res, false, "Failed to create a buffer");

    VkMemoryRequirements memreqs;
    vkGetBufferMemoryRequirements(m_dev, buffer, &memreqs);

    uint32_t alignment_bits = memreqs.alignment * 8;
    // The OpenCL spec requires at least 1024 bits (long16's alignment)
    m_mem_base_addr_align = std::max(alignment_bits, 1024u);
    vkDestroyBuffer(m_dev, buffer, nullptr);

    // Print relevant device limits
    const VkPhysicalDeviceLimits& limits = vulkan_limits();
    cvk_info_fn("device's resources per stage limits:");
    cvk_info_fn("    total = %u", limits.maxPerStageResources);
    cvk_info_fn("    uniform buffers = %u", limits.maxPerStageDescriptorUniformBuffers);
    cvk_info_fn("    storage buffers = %u", limits.maxPerStageDescriptorStorageBuffers);
    cvk_info_fn("device's max buffer size = %s", pretty_size(limits.maxStorageBufferRange).c_str());

    // Print memoy information
    cvk_info_fn("device has %u memory types:", m_mem_properties.memoryTypeCount);
    for (uint32_t i = 0; i < m_mem_properties.memoryTypeCount; i++) {
        VkMemoryType memtype = m_mem_properties.memoryTypes[i];
        auto heapsize = m_mem_properties.memoryHeaps[memtype.heapIndex].size;
        cvk_info_fn("    %u: heap = %u, %s | %s", i, memtype.heapIndex,
            pretty_size(heapsize).c_str(),
            vulkan_memory_property_flags_string(memtype.propertyFlags).c_str());
    }
    
    cvk_info_fn("device has %u memory heaps:", m_mem_properties.memoryHeapCount);
    for (uint32_t i = 0; i < m_mem_properties.memoryHeapCount; i++) {
        VkMemoryHeap memheap = m_mem_properties.memoryHeaps[i];
        cvk_info_fn("    %u: %s | %s", i,
            pretty_size(memheap.size).c_str(),
            vulkan_memory_property_flags_string(memheap.flags).c_str());
    }

    return true;
}

bool cvk_device::supports_capability(spv::Capability capability) const {
    switch (capability) {
    // Capabilities required by all Vulkan implementations:
    case spv::CapabilityShader:
        return true;
    // Optional capabilities:
    case spv::CapabilityFloat64:
        return m_features.shaderFloat64;
    case spv::CapabilityInt16:
        return m_features.shaderInt16;
    case spv::CapabilityInt64:
        return m_features.shaderInt64;
    case spv::CapabilityStorageImageWriteWithoutFormat:
        return m_features.shaderStorageImageWriteWithoutFormat;
    case spv::CapabilityVariablePointers:
        return m_features_variable_pointers.variablePointers;
    case spv::CapabilityVariablePointersStorageBuffer:
        return m_features_variable_pointers.variablePointersStorageBuffer;
    // Capabilities that have not yet been mapped to Vulkan features:
    default:
        cvk_warn_fn("Capability %d not yet mapped to a feature.", capability);
        return false;
    }
}
