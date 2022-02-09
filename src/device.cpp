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

#include <fstream>

#include "device.hpp"
#include "init.hpp"
#include "log.hpp"
#include "memory.hpp"

#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif

constexpr VkMemoryPropertyFlags cvk_device::buffer_supported_memory_types[];
constexpr VkMemoryPropertyFlags cvk_device::image_supported_memory_types[];

cvk_device* cvk_device::create(cvk_platform* platform, VkInstance instance,
                               VkPhysicalDevice pdev) {
    cvk_device* device = new cvk_device(platform, pdev);

    if (!device->init(instance)) {
        delete device;
        return nullptr;
    }

    return device;
}

void cvk_device::init_vulkan_properties(VkInstance instance) {

    cvk_info("Getting Vulkan device properties");

    m_device_id_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES_KHR;
    m_driver_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES_KHR;
    m_pci_bus_info_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT;

#define VER_EXT_PROP(ver, ext, prop)                                           \
    {ver, ext, reinterpret_cast<VkBaseOutStructure*>(&prop)}
    std::vector<std::tuple<uint32_t, const char*, VkBaseOutStructure*>>
        coreversion_extension_properties = {
            VER_EXT_PROP(VK_MAKE_VERSION(1, 1, 0), nullptr,
                         m_device_id_properties),
            VER_EXT_PROP(VK_MAKE_VERSION(1, 2, 0),
                         VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME,
                         m_driver_properties),
            VER_EXT_PROP(0, VK_EXT_PCI_BUS_INFO_EXTENSION_NAME,
                         m_pci_bus_info_properties),
        };
#undef VER_EXT_PROP

    VkBaseOutStructure* pNext = nullptr;
    for (auto& ver_ext_prop : coreversion_extension_properties) {
        auto corever = std::get<0>(ver_ext_prop);
        auto ext = std::get<1>(ver_ext_prop);
        auto prop = std::get<2>(ver_ext_prop);
        if ((corever != 0) && (m_properties.apiVersion >= corever)) {
            prop->pNext = pNext;
            pNext = prop;
        } else if ((ext != nullptr) && is_vulkan_extension_enabled(ext)) {
            prop->pNext = pNext;
            pNext = prop;
        }
    }

    VkPhysicalDeviceProperties2KHR properties;
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR;
    properties.pNext = pNext;

    auto func = GET_INSTANCE_PROC(instance, vkGetPhysicalDeviceProperties2KHR);
    if (!func) {
        cvk_fatal(
            "Failed to get pointer to vkGetPhysicalDeviceProperties2KHR()");
    }
    func(m_pdev, &properties);
}

void cvk_device::init_opencl_properties() {
    // Set default values for all properties.
    m_global_mem_cache_size = 0;
    m_num_compute_units = 1;

    // Set correct property values for known devices.
    // These values can be obtained from the native OpenCL driver.
    if (!strncmp(m_properties.deviceName, "Mali-", 5)) {
#ifdef __ANDROID__
        // Find out which SoC this is.
        char soc[PROP_VALUE_MAX + 1];
        int len = __system_property_get("ro.hardware", soc);
        if (len == 0) {
            cvk_warn("Unable to query 'ro.hardware' system property, some "
                     "device properties will be incorrect.");
            return;
        }

        if (!strcmp(soc, "exynos9820")) {
            m_global_mem_cache_size = 262144;
            m_num_compute_units = 12;
        } else if (!strcmp(soc, "exynos990")) {
            m_global_mem_cache_size = 262144;
            m_num_compute_units = 11;
        } else {
            cvk_warn("Unrecognized 'ro.hardware' value '%s', some device "
                     "properties will be incorrect.",
                     soc);
        }
#else
        cvk_warn("Unrecognized Mali device, some device properties will be "
                 "incorrect.");
#endif
    } else if (!strcmp(m_properties.deviceName, "Adreno (TM) 615")) {
        m_global_mem_cache_size = 65536;
        m_num_compute_units = 1;
    } else if (!strcmp(m_properties.deviceName, "Adreno (TM) 620")) {
        m_global_mem_cache_size = 65536;
        m_num_compute_units = 1;
    } else if (!strcmp(m_properties.deviceName, "Adreno (TM) 630")) {
        m_global_mem_cache_size = 131072;
        m_num_compute_units = 2;
    } else if (!strcmp(m_properties.deviceName, "Adreno (TM) 640")) {
        m_global_mem_cache_size = 131072;
        m_num_compute_units = 2;
    } else {
        cvk_warn("Unrecognized device '%s', some device properties will be "
                 "incorrect.",
                 m_properties.deviceName);
    }
}

void cvk_device::init_driver_behaviors() {

    cvk_info("Initialising driver behaviors");

    // Disable all driver-specific behaviors by default
    m_driver_behaviors = 0;

    if (is_vulkan_extension_enabled(VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME) ||
        m_properties.apiVersion >= VK_MAKE_VERSION(1, 2, 0)) {

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
        VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
        VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
        VK_KHR_UNIFORM_BUFFER_STANDARD_LAYOUT_EXTENSION_NAME,
        VK_KHR_VARIABLE_POINTERS_EXTENSION_NAME,
        VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME,
        VK_EXT_PCI_BUS_INFO_EXTENSION_NAME,
        VK_KHR_SPIRV_1_4_EXTENSION_NAME,
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
    if (supported_features.features.shaderStorageImageReadWithoutFormat) {
        m_features.features.shaderStorageImageReadWithoutFormat = VK_TRUE;
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
        MAKE_NAME_VERSION(1, 0, 0, "cl_khr_il_program"),
        MAKE_NAME_VERSION(1, 0, 0, "cl_khr_spirv_no_integer_wrap_decoration"),
        MAKE_NAME_VERSION(1, 0, 0, "cl_arm_non_uniform_work_group_size"),
        MAKE_NAME_VERSION(1, 0, 0, "cl_khr_suggested_local_work_size"),
    };

    if (m_properties.apiVersion >= VK_MAKE_VERSION(1, 1, 0)) {
        m_extensions.push_back(
            MAKE_NAME_VERSION(1, 0, 0, "cl_khr_device_uuid"));
    }

    // Enable cl_khr_fp16 if we have 16-bit storage and shaderFloat16
    if ((is_vulkan_extension_enabled(VK_KHR_16BIT_STORAGE_EXTENSION_NAME) &&
         m_features_16bit_storage.storageBuffer16BitAccess) &&
        (is_vulkan_extension_enabled(
             VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME) &&
         m_features_float16_int8.shaderFloat16)) {
        m_has_fp16_support = true;
        m_extensions.push_back(MAKE_NAME_VERSION(1, 0, 0, "cl_khr_fp16"));
    }

    // Enable 8-bit integer support if possible
    if ((is_vulkan_extension_enabled(VK_KHR_8BIT_STORAGE_EXTENSION_NAME) &&
         m_features_8bit_storage.storageBuffer8BitAccess) &&
        (is_vulkan_extension_enabled(
             VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME) &&
         m_features_float16_int8.shaderInt8)) {
        m_has_int8_support = true;
    }

    // Report cl_khr_pci_bus_info if VK_EXT_pci_bus_info is supported
    if (is_vulkan_extension_enabled(VK_EXT_PCI_BUS_INFO_EXTENSION_NAME)) {
        m_extensions.push_back(
            MAKE_NAME_VERSION(1, 0, 0, "cl_khr_pci_bus_info"));
    }

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
        m_ils_string += std::to_string(CL_VERSION_MAJOR(il.version));
        m_ils_string += ".";
        m_ils_string += std::to_string(CL_VERSION_MINOR(il.version));
        m_ils_string += " ";
    }

    // Build list of supported OpenCL C versions
    m_opencl_c_versions = {
        MAKE_NAME_VERSION(1, 0, 0, "OpenCL C"),
        MAKE_NAME_VERSION(1, 1, 0, "OpenCL C"),
        MAKE_NAME_VERSION(1, 2, 0, "OpenCL C"),
        MAKE_NAME_VERSION(3, 0, 0, "OpenCL C"),
    };

    // Build list of supported OpenCL C features
    m_opencl_c_features = {
        MAKE_NAME_VERSION(3, 0, 0, "__opencl_c_images"),
    };
    if (m_features.features.shaderInt64) {
        m_opencl_c_features.push_back(
            MAKE_NAME_VERSION(3, 0, 0, "__opencl_c_int64"));
    }
    if (m_features.features.shaderFloat64) {
        m_has_fp64_support = true;
        m_opencl_c_features.push_back(
            MAKE_NAME_VERSION(3, 0, 0, "__opencl_c_fp64"));
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
    m_vulkan_queues.reserve(num_queues);
    for (auto i = 0U; i < num_queues; i++) {
        VkQueue queue;

        vkGetDeviceQueue(m_dev, queue_family, i, &queue);
        m_vulkan_queues.emplace_back(queue, queue_family);
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
        cvk_warn("Command queue profiling will suffer from limitations");
    }

    return true;
}

// Returns the pipeline cache file path for a given SPIR-V SHA-1 hash.
// If pipeline cache serialization is not enabled, an empty string is returned.
std::string
cvk_device::get_pipeline_cache_filename(const cvk_sha1_hash& sha1) const {
    const char* cache_dir = getenv("CLVK_CACHE_DIR");
    if (cache_dir == nullptr) {
        return "";
    }

    // The pipeline cache file path is:
    // ${CLVK_CACHE_DIR}/clvk-pipeline-cache.<UUID>.<SHA1>.bin
    std::string cache_path = cache_dir;
    cache_path += "/";
    cache_path += "clvk-pipeline-cache.";
    cache_path += to_hex_string(m_properties.pipelineCacheUUID, VK_UUID_SIZE);
    cache_path += ".";
    cache_path += to_hex_string(reinterpret_cast<const uint8_t*>(sha1.data()),
                                SHA1_DIGEST_NUM_BYTES);
    cache_path += ".bin";
    return cache_path;
}

bool cvk_device::get_pipeline_cache(const std::vector<uint32_t>& spirv,
                                    VkPipelineCache& pipeline_cache) {

    std::lock_guard<std::mutex> lock(m_pipeline_cache_mutex);

    pipeline_cache = VK_NULL_HANDLE;

    // Compute SHA-1 hash of the SPIR-V binary
    cvk_sha1_hash sha1 =
        cvk_sha1(spirv.data(), spirv.size() * sizeof(uint32_t));

    // Check the in-memory cache of pipeline caches
    if (m_pipeline_caches.count(sha1)) {
        pipeline_cache = m_pipeline_caches.at(sha1);
        return true;
    }

    std::vector<char> cache_data;

    // Load pipeline cache data from file if this is enabled
    std::string cache_path = get_pipeline_cache_filename(sha1);
    if (!cache_path.empty()) {
        cvk_info("Looking for pipeline cache at %s", cache_path.c_str());
        std::ifstream cache_file(cache_path, std::ios::in | std::ios::binary);
        if (cache_file.is_open()) {
            // Get the size of the pipeline cache file
            cache_file.seekg(0, std::ios::end);
            uint32_t size = cache_file.tellg();
            cache_file.seekg(0, std::ios::beg);

            // Load the pipeline cache data into memory
            cache_data.resize(size);
            cache_file.read(cache_data.data(), size);
            if (!cache_file.good()) {
                cvk_warn("Failed to read pipeline cache data");
                cache_data.clear();
            }
        } else {
            cvk_warn("Failed to open pipeline cache file");
        }
    }

    // Create pipeline cache
    VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
        nullptr,           // pNext
        0,                 // flags
        cache_data.size(), // initialDataSize
        cache_data.data(), // pInitialData
    };

    VkResult res = vkCreatePipelineCache(m_dev, &pipelineCacheCreateInfo,
                                         nullptr, &pipeline_cache);
    if (res != VK_SUCCESS) {
        cvk_error("Could not create pipeline cache.");
        return false;
    }

    // Add pipeline cache to the in-memory cache
    m_pipeline_caches[sha1] = pipeline_cache;

    return cache_data.size() != 0;
}

void cvk_device::save_pipeline_cache(
    const cvk_sha1_hash& sha1, const VkPipelineCache& pipeline_cache) const {
    VkResult res;

    std::string cache_path = get_pipeline_cache_filename(sha1);
    if (cache_path.empty()) {
        return;
    }

    // Retrieve the pipeline cache data from the Vulkan implementation
    size_t size;
    res = vkGetPipelineCacheData(m_dev, pipeline_cache, &size, nullptr);
    if (res != VK_SUCCESS) {
        cvk_error("Failed to retrieve pipeline cache size");
        return;
    }
    std::vector<char> cache_data(size);
    res =
        vkGetPipelineCacheData(m_dev, pipeline_cache, &size, cache_data.data());
    if (res != VK_SUCCESS) {
        cvk_error("Failed to retrieve pipeline cache data");
        return;
    }

    cvk_info("Writing %lu bytes of pipeline cache data to file", size);

    // Write the pipeline cache data to file
    std::ofstream cache_file(cache_path, std::ios::out | std::ios::binary);
    if (!cache_file.is_open()) {
        cvk_error("Failed to open pipeline cache file for writing: %s",
                  cache_path.c_str());
        return;
    }
    cache_file.write(cache_data.data(), size);
    if (!cache_file.good()) {
        cvk_error("Failed to write pipeline cache data");
    }
}

void cvk_device::init_spirv_environment() {
    if (m_properties.apiVersion < VK_MAKE_VERSION(1, 1, 0)) {
        m_vulkan_spirv_env = SPV_ENV_VULKAN_1_0;
    } else if (m_properties.apiVersion < VK_MAKE_VERSION(1, 2, 0)) {
        if (is_vulkan_extension_enabled(VK_KHR_SPIRV_1_4_EXTENSION_NAME)) {
            m_vulkan_spirv_env = SPV_ENV_VULKAN_1_1_SPIRV_1_4;
        } else {
            m_vulkan_spirv_env = SPV_ENV_VULKAN_1_1;
        }
    } else {
        // Assume 1.2
        m_vulkan_spirv_env = SPV_ENV_VULKAN_1_2;
    }
    cvk_info("Vulkan SPIR-V environment: %s",
             spvTargetEnvDescription(m_vulkan_spirv_env));
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
    cvk_info("Device's max uniform buffer size = %s",
             pretty_size(limits.maxUniformBufferRange).c_str());
    cvk_info("Device's max push constant size = %s",
             pretty_size(limits.maxPushConstantsSize).c_str());
    cvk_info("Device's execution capabilities:");
    cvk_info("    Max work-group count: {%u,%u,%u}",
             limits.maxComputeWorkGroupCount[0],
             limits.maxComputeWorkGroupCount[1],
             limits.maxComputeWorkGroupCount[2]);
    cvk_info("    Max invocations per work-group: %u",
             limits.maxComputeWorkGroupInvocations);
    cvk_info("    Max work-group size: {%u,%u,%u}",
             limits.maxComputeWorkGroupSize[0],
             limits.maxComputeWorkGroupSize[1],
             limits.maxComputeWorkGroupSize[2]);

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
    cvk_info("  API Version: %s",
             vulkan_version_string(m_properties.apiVersion).c_str());

    uint32_t num_queues, queue_family;
    if (!init_queues(&num_queues, &queue_family)) {
        return false;
    }

    if (!init_extensions()) {
        return false;
    }

    init_vulkan_properties(instance);

    init_opencl_properties();

    init_driver_behaviors();

    init_features(instance);

    build_extension_ils_list();

    if (!init_time_management(instance)) {
        return false;
    }

    if (!create_vulkan_queues_and_device(num_queues, queue_family)) {
        return false;
    }

    init_spirv_environment();

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
    case spv::CapabilityStorageImageReadWithoutFormat:
        return m_features.features.shaderStorageImageReadWithoutFormat;
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

void cvk_device::select_work_group_size(
    const std::array<uint32_t, 3>& global_size,
    std::array<uint32_t, 3>& local_size) const {

    // Start at (1,1,1), which is always valid.
    local_size = {1, 1, 1};

    // Cap the total work-group size to the Vulkan device's limit.
    uint32_t max_size = m_properties.limits.maxComputeWorkGroupInvocations;

    // Further cap the total size to 64, as this is expected to be a
    // reasonable size on many devices.
    max_size = std::min(max_size, UINT32_C(64));

    // TODO: We should also take into account the total number of
    // work-groups that would be launched, to ensure the device is fully
    // utilized for smaller global work sizes.

    // Increase the work-group size until we hit device limits.
    bool changed;
    do {
        changed = false;

        // Alternate between increasing the X and Y dimensions.
        // TODO: Consider increasing the Z dimension as well?
        for (int i = 0; i < 2; i++) {
            // Double the dimension if we can.
            // TODO: Allow non power-of-two sizes?
            // TODO: Allow non-uniform sizes if supported?
            std::array<uint32_t, 3> new_local_size = local_size;
            new_local_size[i] *= 2;
            if (global_size[i] % new_local_size[i] == 0 &&
                new_local_size[i] <=
                    m_properties.limits.maxComputeWorkGroupCount[i] &&
                new_local_size[0] * new_local_size[1] * new_local_size[2] <=
                    max_size) {
                local_size = new_local_size;
                changed = true;
            }
        }
    } while (changed);
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

cl_ulong cvk_device::device_timer_to_host(cl_ulong dev, cl_ulong sync_dev,
                                          cl_ulong sync_host) const {
    if (sync_host > sync_dev) {
        return (sync_host - sync_dev) + dev;
    } else {
        return dev - (sync_dev - sync_host);
    }
}
