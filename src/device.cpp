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
#include "init.hpp"
#include "log.hpp"
#include "memory.hpp"

cvk_device* cvk_device::create(cvk_platform* platform, VkInstance instance,
                               VkPhysicalDevice pdev) {
    cvk_device* device = new cvk_device(platform, pdev);

    if (!device->init(instance)) {
        delete device;
        return nullptr;
    }

    return device;
}

void cvk_device::init_driver_behaviors(VkInstance instance) {

    cvk_info("Initialising driver behaviors");

    // Disable all driver-specific behaviors by default
    m_driver_behaviors = 0;

    if (is_vulkan_extension_enabled(VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME) ||
        m_properties.apiVersion >= VK_MAKE_VERSION(1, 2, 0)) {

        // Get driver properties
        VkPhysicalDeviceProperties2KHR properties;
        properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR;
        properties.pNext = &m_driver_properties;
        m_driver_properties.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES_KHR;
        m_driver_properties.pNext = nullptr;
        auto func =
            GET_INSTANCE_PROC(instance, vkGetPhysicalDeviceProperties2KHR);
        if (!func) {
            cvk_fatal(
                "Failed to get pointer to vkGetPhysicalDeviceProperties2KHR()");
        }
        func(m_pdev, &properties);

        // Log basic information about the target Vulkan device
        cvk_info("  driverName = %s", m_driver_properties.driverName);
        cvk_info("  driverInfo = %s", m_driver_properties.driverInfo);
        cvk_info("  conformanceVersion = %d.%d.%d.%d",
                 m_driver_properties.conformanceVersion.major,
                 m_driver_properties.conformanceVersion.minor,
                 m_driver_properties.conformanceVersion.subminor,
                 m_driver_properties.conformanceVersion.patch);

        // Select behaviors based on the target Vulkan device and driver version
        if (m_driver_properties.driverID == VK_DRIVER_ID_ARM_PROPRIETARY_KHR) {
            // Workaround for resource management bug on Mali GPUs.
            // TODO: Make this conditional on the driver version when this is
            // fixed in the driver.
            m_driver_behaviors |= use_reset_command_buffer_bit;
        }
    } else {
        cvk_warn("The VK_KHR_driver_properties extension is not supported.");
        cvk_warn("Using default Vulkan driver behaviors.");
    }

    // List driver behaviors
    cvk_info("Driver behaviors:");
#define PRINT_BEHAVIOR(name)                                                   \
    cvk_info("  %s = %s", #name, (m_driver_behaviors & name) ? "true" : "false")
    PRINT_BEHAVIOR(use_reset_command_buffer_bit);
#undef PRINT_BEHAVIOR
}

bool cvk_device::init_queues(uint32_t* num_queues, uint32_t* queue_family) {
    // Get number of queue families
    uint32_t num_families;
    vkGetPhysicalDeviceQueueFamilyProperties(m_pdev, &num_families, nullptr);

    cvk_info(
        "Physical device (%s) has %u queue families:",
        vulkan_physical_device_type_string(m_properties.deviceType).c_str(),
        num_families);

    // Get their properties
    std::vector<VkQueueFamilyProperties> families(num_families);
    vkGetPhysicalDeviceQueueFamilyProperties(m_pdev, &num_families,
                                             families.data());

    // Look for suitable queues
    bool found_queues = false;
    *num_queues = 0;
    for (uint32_t i = 0; i < num_families; i++) {

        cvk_info("  queue family %u: %2u queues | %s", i,
                 families[i].queueCount,
                 vulkan_queue_flags_string(families[i].queueFlags).c_str());

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

bool cvk_device::init_extensions() {
    uint32_t numext;
    VkResult res =
        vkEnumerateDeviceExtensionProperties(m_pdev, nullptr, &numext, nullptr);
    CVK_VK_CHECK_ERROR_RET(
        res, false, "Failed to get the number of device extension properties");

    cvk_info("%u extensions are supported", numext);

    std::vector<VkExtensionProperties> extensions(numext);
    res = vkEnumerateDeviceExtensionProperties(m_pdev, nullptr, &numext,
                                               extensions.data());
    CVK_VK_CHECK_ERROR_RET(res, false,
                           "Could not enumerate device extension properties");

    if (m_properties.apiVersion < VK_MAKE_VERSION(1, 1, 0)) {
        m_vulkan_device_extensions.push_back(
            VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME);
    }

    const std::vector<const char*> desired_extensions = {
        VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
        VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
        VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME,
        VK_KHR_VARIABLE_POINTERS_EXTENSION_NAME,
        VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
        VK_KHR_UNIFORM_BUFFER_STANDARD_LAYOUT_EXTENSION_NAME,
        VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME,
    };

    for (size_t i = 0; i < numext; i++) {
        cvk_info("  %s, spec version %u", extensions[i].extensionName,
                 extensions[i].specVersion);

        for (auto name : desired_extensions) {
            if (!strcmp(name, extensions[i].extensionName)) {
                m_vulkan_device_extensions.push_back(name);
                cvk_info("    ENABLING");
                break;
            }
        }
    }

    return true;
}

void cvk_device::init_features(VkInstance instance) {

    cvk_info("Initialising features");

    // Query supported features.
    m_features_ubo_stdlayout.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES_KHR;
    m_features_float16_int8.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR;
    m_features_variable_pointer.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES_KHR;
    m_features_8bit_storage.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR;
    m_features_16bit_storage.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR;

    std::vector<std::pair<const char*, VkBaseOutStructure*>>
        extension_features = {
#define EXTFEAT(EXT, FEAT) {EXT, reinterpret_cast<VkBaseOutStructure*>(FEAT)}
            EXTFEAT(VK_KHR_UNIFORM_BUFFER_STANDARD_LAYOUT_EXTENSION_NAME,
                    &m_features_ubo_stdlayout),
            EXTFEAT(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
                    &m_features_float16_int8),
            EXTFEAT(VK_KHR_VARIABLE_POINTERS_EXTENSION_NAME,
                    &m_features_variable_pointer),
            EXTFEAT(VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
                    &m_features_8bit_storage),
            EXTFEAT(VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
                    &m_features_16bit_storage),
#undef EXTFEAT
        };

    VkBaseOutStructure* pNext = nullptr;
    for (auto& ext_feat : extension_features) {
        auto ext = ext_feat.first;
        if (is_vulkan_extension_enabled(ext)) {
            auto feat = ext_feat.second;
            feat->pNext = pNext;
            pNext = feat;
        }
    }

    VkPhysicalDeviceFeatures2 supported_features;
    supported_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    supported_features.pNext = pNext;

    if (m_properties.apiVersion < VK_MAKE_VERSION(1, 1, 0)) {
        // Use the extension on Vulkan 1.0 platforms
        auto func = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2KHR>(
            vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2KHR"));
        if (!func) {
            cvk_fatal(
                "Failed to get pointer to vkGetPhysicalDeviceFeatures2KHR()");
        }
        func(m_pdev, &supported_features);
    } else {
        vkGetPhysicalDeviceFeatures2(m_pdev, &supported_features);
    }

    // Log supported features
    cvk_info("8-bit storage: SSBO = %d, UBO = %d, Push constants = %d",
             m_features_8bit_storage.storageBuffer8BitAccess,
             m_features_8bit_storage.uniformAndStorageBuffer8BitAccess,
             m_features_8bit_storage.storagePushConstant8);
    cvk_info("16-bit storage: SSBO = %d, UBO = %d, Push constants = %d",
             m_features_16bit_storage.storageBuffer16BitAccess,
             m_features_16bit_storage.uniformAndStorageBuffer16BitAccess,
             m_features_16bit_storage.storagePushConstant16);

    // Selectively enable core features.
    if (supported_features.features.shaderInt16) {
        m_features.features.shaderInt16 = VK_TRUE;
    }
    if (supported_features.features.shaderInt64) {
        m_features.features.shaderInt64 = VK_TRUE;
    }
    if (supported_features.features.shaderFloat64) {
        m_features.features.shaderFloat64 = VK_TRUE;
    }
    if (supported_features.features.shaderStorageImageWriteWithoutFormat) {
        m_features.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    }

    // All queried extended features are enabled when supported.
    m_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    m_features.pNext = pNext;
}

void cvk_device::build_extension_ils_list() {

    m_extensions = {
        // Start with required extensions
        MAKE_NAME_VERSION(1, 0, 0, "cl_khr_global_int32_base_atomics"),
        MAKE_NAME_VERSION(1, 0, 0, "cl_khr_global_int32_extended_atomics"),
        MAKE_NAME_VERSION(1, 0, 0, "cl_khr_local_int32_base_atomics"),
        MAKE_NAME_VERSION(1, 0, 0, "cl_khr_local_int32_extended_atomics"),
        MAKE_NAME_VERSION(1, 0, 0, "cl_khr_byte_addressable_store"),

        // Add always supported extensions
        MAKE_NAME_VERSION(1, 0, 0, "cl_khr_extended_versioning"),
        MAKE_NAME_VERSION(1, 0, 0, "cl_khr_create_command_queue"),
#ifndef CLSPV_ONLINE_COMPILER
        MAKE_NAME_VERSION(1, 0, 0, "cl_khr_il_program"),
#endif
        MAKE_NAME_VERSION(1, 0, 0, "cl_khr_spirv_no_integer_wrap_decoration"),
    };

    // Build extension string
    for (auto& ext : m_extensions) {
        m_extension_string += ext.name;
        m_extension_string += " ";
    }

    // Build list of ILs
    m_ils = {
        MAKE_NAME_VERSION(1, 0, 0, "SPIR-V"),
    };

    for (auto& il : m_ils) {
        m_ils_string += il.name;
        m_ils_string += "_";
        m_ils_string += std::to_string(CL_VERSION_MAJOR_KHR(il.version));
        m_ils_string += ".";
        m_ils_string += std::to_string(CL_VERSION_MINOR_KHR(il.version));
        m_ils_string += " ";
    }
}

bool cvk_device::create_vulkan_queues_and_device(uint32_t num_queues,
                                                 uint32_t queue_family) {
    cvk_info("Creating Vulkan device and queues");
    // Give all queues the same priority
    std::vector<float> queuePriorities(num_queues, 1.0f);

    VkDeviceQueueCreateInfo queueCreateInfo = {
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        nullptr,
        0, // flags
        queue_family,
        num_queues, // queueCount
        queuePriorities.data()};

    // Create logical device
    const VkDeviceCreateInfo createInfo = {
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, // sType
        &m_features,                          // pNext
        0,                                    // flags
        1,                                    // queueCreateInfoCount
        &queueCreateInfo,                     // pQueueCreateInfos,
        0,                                    // enabledLayerCount
        nullptr,                              // ppEnabledLayerNames
        static_cast<uint32_t>(
            m_vulkan_device_extensions.size()), // enabledExtensionCount
        m_vulkan_device_extensions.data(),      // ppEnabledExtensionNames
        nullptr,                                // pEnabledFeatures
    };

    VkResult res = vkCreateDevice(m_pdev, &createInfo, nullptr, &m_dev);
    CVK_VK_CHECK_ERROR_RET(res, false, "Failed to create a device");

    // Construct the queue wrappers now that our queues exist
    for (auto i = 0U; i < num_queues; i++) {
        VkQueue queue;

        vkGetDeviceQueue(m_dev, queue_family, i, &queue);
        m_vulkan_queues.emplace_back(
            cvk_vulkan_queue_wrapper(queue, queue_family));
    }

    return true;
}

bool cvk_device::init_time_management(VkInstance instance) {

    if (is_vulkan_extension_enabled(
            VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME)) {
        auto func = GET_INSTANCE_PROC(
            instance, vkGetPhysicalDeviceCalibrateableTimeDomainsEXT);
        uint32_t num_time_domains;
        VkResult res;
        res = func(m_pdev, &num_time_domains, nullptr);
        if (res != VK_SUCCESS) {
            cvk_error(
                "Can't get number of available calibrateable time domains");
            return false;
        }

        cvk_info("Device supports %u calibrateable time domains",
                 num_time_domains);

        std::vector<VkTimeDomainEXT> supported_time_domains(num_time_domains);
        res = func(m_pdev, &num_time_domains, supported_time_domains.data());
        if (res != VK_SUCCESS) {
            cvk_error("Can't get list of available calibrateable time domains");
            return false;
        }

        bool has_device = false;
        bool has_monotonic = false;
        for (auto td : supported_time_domains) {
            cvk_info("  %s",
                     vulkan_calibrateable_time_domain_string(td).c_str());
            if (td == VK_TIME_DOMAIN_CLOCK_MONOTONIC_EXT) {
                has_monotonic = true;
            }

            if (td == VK_TIME_DOMAIN_DEVICE_EXT) {
                has_device = true;
            }
        }

        if (has_device && has_monotonic) {
            m_has_timer_support = true;
            m_vkfns.vkGetCalibratedTimestampsEXT =
                GET_INSTANCE_PROC(instance, vkGetCalibratedTimestampsEXT);
        }
    }

    if (!m_has_timer_support) {
        cvk_warn("This device does not support VK_EXT_calibrated_timestamps or "
                 "it does not support the required "
                 "VK_TIME_DOMAIN_CLOCK_MONOTONIC_EXT and "
                 "VK_TIME_DOMAIN_DEVICE_EXT time domains");
        cvk_warn("clGetHostTimer and clGetDeviceAndHostTimer will not work");
    }

    return true;
}

void cvk_device::log_limits_and_memory_information() {
    // Print relevant device limits
    const VkPhysicalDeviceLimits& limits = vulkan_limits();
    cvk_info("Device's resources per stage limits:");
    cvk_info("    total = %u", limits.maxPerStageResources);
    cvk_info("    uniform buffers = %u",
             limits.maxPerStageDescriptorUniformBuffers);
    cvk_info("    storage buffers = %u",
             limits.maxPerStageDescriptorStorageBuffers);
    cvk_info("Device's max buffer size = %s",
             pretty_size(limits.maxStorageBufferRange).c_str());
    cvk_info("Device's max push constant size = %s",
             pretty_size(limits.maxPushConstantsSize).c_str());

    // Print memoy information
    cvk_info("Device has %u memory types:", m_mem_properties.memoryTypeCount);
    for (uint32_t i = 0; i < m_mem_properties.memoryTypeCount; i++) {
        VkMemoryType memtype = m_mem_properties.memoryTypes[i];
        auto heapsize = m_mem_properties.memoryHeaps[memtype.heapIndex].size;
        cvk_info(
            "    %u: heap = %u, %s | %s", i, memtype.heapIndex,
            pretty_size(heapsize).c_str(),
            vulkan_memory_property_flags_string(memtype.propertyFlags).c_str());
    }

    cvk_info("Device has %u memory heaps:", m_mem_properties.memoryHeapCount);
    for (uint32_t i = 0; i < m_mem_properties.memoryHeapCount; i++) {
        VkMemoryHeap memheap = m_mem_properties.memoryHeaps[i];
        cvk_info("    %u: %s | %s", i, pretty_size(memheap.size).c_str(),
                 vulkan_memory_property_flags_string(memheap.flags).c_str());
    }
}

bool cvk_device::init(VkInstance instance) {
    cvk_info("Initialising device %s", m_properties.deviceName);

    uint32_t num_queues, queue_family;
    if (!init_queues(&num_queues, &queue_family)) {
        return false;
    }

    if (!init_extensions()) {
        return false;
    }

    init_driver_behaviors(instance);

    init_features(instance);

    build_extension_ils_list();

    if (!init_time_management(instance)) {
        return false;
    }

    if (!create_vulkan_queues_and_device(num_queues, queue_family)) {
        return false;
    }

    log_limits_and_memory_information();

    return true;
}

bool cvk_device::supports_capability(spv::Capability capability) const {
    switch (capability) {
    // Capabilities required by all Vulkan implementations:
    case spv::CapabilityShader:
    case spv::CapabilitySampled1D:
    case spv::CapabilityImage1D:
    case spv::CapabilityImageQuery:
        return true;
    // Optional capabilities:
    case spv::CapabilityFloat16:
        return m_features_float16_int8.shaderFloat16;
    case spv::CapabilityFloat64:
        return m_features.features.shaderFloat64;
    case spv::CapabilityInt8:
        return m_features_float16_int8.shaderInt8;
    case spv::CapabilityInt16:
        return m_features.features.shaderInt16;
    case spv::CapabilityInt64:
        return m_features.features.shaderInt64;
    case spv::CapabilityStorageImageWriteWithoutFormat:
        return m_features.features.shaderStorageImageWriteWithoutFormat;
    case spv::CapabilityVariablePointers:
        return m_features_variable_pointer.variablePointers;
    case spv::CapabilityVariablePointersStorageBuffer:
        return m_features_variable_pointer.variablePointersStorageBuffer;
    // Capabilities that have not yet been mapped to Vulkan features:
    default:
        cvk_warn_fn("Capability %d not yet mapped to a feature.", capability);
        return false;
    }
}

cl_int cvk_device::get_device_host_timer(cl_ulong* device_timestamp,
                                         cl_ulong* host_timestamp) const {
    auto vkdev = vulkan_device();

    uint64_t timestamps[2];
    uint64_t max_deviation;
    VkCalibratedTimestampInfoEXT timestamp_infos[2] = {
        {VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT, nullptr,
         VK_TIME_DOMAIN_CLOCK_MONOTONIC_EXT},
        {VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT, nullptr,
         VK_TIME_DOMAIN_DEVICE_EXT}};

    uint32_t num_requested_timestamps;
    if (device_timestamp == nullptr) {
        num_requested_timestamps = 1;
    } else {
        num_requested_timestamps = 2;
    }

    auto res = m_vkfns.vkGetCalibratedTimestampsEXT(
        vkdev, num_requested_timestamps, timestamp_infos, timestamps,
        &max_deviation);
    if (res != VK_SUCCESS) {
        return CL_OUT_OF_RESOURCES;
    }

    *host_timestamp = timestamps[0];
    if (device_timestamp != nullptr) {
        *device_timestamp = timestamp_to_ns(timestamps[1]);
    }

    return CL_SUCCESS;
}
