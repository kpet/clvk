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

#include <array>
#include <fstream>
#include <functional>
#include <iterator>
#include <sstream>

#include "config.hpp"
#include "device.hpp"
#include "init.hpp"
#include "log.hpp"
#include "memory.hpp"

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
    m_subgroup_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    m_subgroup_size_control_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;
    m_float_controls_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES;
    m_integer_dot_product_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES;

    //--- Get maxMemoryAllocationSize for figuring out the  max single buffer
    // allocation size and default init when the extension is not supported
    m_maintenance3_properties.maxMemoryAllocationSize =
        std::numeric_limits<VkDeviceSize>::max();
    m_maintenance3_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES;
    //---

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
            VER_EXT_PROP(VK_MAKE_VERSION(1, 1, 0), nullptr,
                         m_subgroup_properties),
            VER_EXT_PROP(VK_MAKE_VERSION(1, 3, 0), nullptr,
                         m_subgroup_size_control_properties),
            VER_EXT_PROP(VK_MAKE_VERSION(1, 1, 0), nullptr,
                         m_maintenance3_properties),
            VER_EXT_PROP(VK_MAKE_VERSION(1, 2, 0), nullptr,
                         m_float_controls_properties),
            VER_EXT_PROP(VK_MAKE_VERSION(1, 3, 0), nullptr,
                         m_integer_dot_product_properties),
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

void cvk_device::init_clvk_runtime_behaviors() {
#define SET_DEVICE_PROPERTY(option, print)                                     \
    do {                                                                       \
        if (config.option.set) {                                               \
            m_##option = config.option;                                        \
        } else {                                                               \
            m_##option = m_clvk_properties->get_##option();                    \
        }                                                                      \
        print(option);                                                         \
    } while (0)

#define PRINT_U(option) cvk_info_fn(#option ": %u", m_##option);
#define SET_DEVICE_PROPERTY_U(option) SET_DEVICE_PROPERTY(option, PRINT_U)

#define PRINT_S(option) cvk_info_fn(#option ": %s", m_##option.c_str());
#define SET_DEVICE_PROPERTY_S(option) SET_DEVICE_PROPERTY(option, PRINT_S)

    SET_DEVICE_PROPERTY_U(max_cmd_batch_size);
    SET_DEVICE_PROPERTY_U(max_first_cmd_batch_size);
    SET_DEVICE_PROPERTY_U(max_cmd_group_size);
    SET_DEVICE_PROPERTY_U(max_first_cmd_group_size);

    SET_DEVICE_PROPERTY_U(physical_addressing);
    SET_DEVICE_PROPERTY_S(spirv_arch);

    SET_DEVICE_PROPERTY_U(preferred_subgroup_size);

#undef PRINT_U
#undef PRINT_S
#undef SET_DEVICE_PROPERTY_U
#undef SET_DEVICE_PROPERTY_S
#undef SET_DEVICE_PROPERTY
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

static bool queue_flags_contains_compute(VkQueueFlags flags) {
    return flags & VK_QUEUE_COMPUTE_BIT;
}
static bool queue_flags_contains_compute_not_graphics(VkQueueFlags flags) {
    if (flags & VK_QUEUE_GRAPHICS_BIT) {
        return false;
    }
    return queue_flags_contains_compute(flags);
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

    std::array<std::function<bool(VkQueueFlags)>, 2> queue_flags_tests = {
        queue_flags_contains_compute_not_graphics,
        queue_flags_contains_compute};

    // Look for suitable queues
    bool found_queues = false;
    *num_queues = 0;
    for (auto queue_flags_test : queue_flags_tests) {
        for (uint32_t i = 0; i < num_families; i++) {

            cvk_info("  queue family %u: %2u queues | %s", i,
                     families[i].queueCount,
                     vulkan_queue_flags_string(families[i].queueFlags).c_str());

            if (!found_queues && queue_flags_test(families[i].queueFlags)) {
                *queue_family = i;
                *num_queues = families[i].queueCount;
                found_queues = true;
            }
        }
    }

    if (!found_queues) {
        cvk_error("Could not find a suitable queue family for this device");
        return false;
    }
    cvk_info(
        "  selecting queue %u: %2u queues | %s", *queue_family, *num_queues,
        vulkan_queue_flags_string(families[*queue_family].queueFlags).c_str());

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

    std::vector<const char*> desired_extensions{
        VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
        VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
        VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME,
        VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
        VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
        VK_KHR_UNIFORM_BUFFER_STANDARD_LAYOUT_EXTENSION_NAME,
        VK_KHR_VARIABLE_POINTERS_EXTENSION_NAME,
        VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME,
        VK_EXT_PCI_BUS_INFO_EXTENSION_NAME,
        VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
        VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME,
        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME,
    };

    if (m_properties.apiVersion < VK_MAKE_VERSION(1, 2, 0)) {
        desired_extensions.push_back(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
        desired_extensions.push_back(
            VK_KHR_SHADER_SUBGROUP_EXTENDED_TYPES_EXTENSION_NAME);
    }

    if (m_properties.apiVersion < VK_MAKE_VERSION(1, 1, 0)) {
        desired_extensions.push_back(VK_KHR_MAINTENANCE3_EXTENSION_NAME);
    }

    if (m_properties.apiVersion < VK_MAKE_VERSION(1, 1, 0)) {
        m_vulkan_device_extensions.push_back(
            VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME);
    } else {
        desired_extensions.push_back(
            VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME);
    }

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
    m_features_shader_subgroup_extended_types.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES_KHR;
    m_features_vulkan_memory_model.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES;
    m_features_buffer_device_address.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR;
    m_features_subgroup_size_control.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
    m_features_shader_integer_dot_product.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
    m_features_queue_global_priority.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GLOBAL_PRIORITY_QUERY_FEATURES_KHR;

    std::vector<std::tuple<uint32_t, const char*, VkBaseOutStructure*>>
        coreversion_extension_features = {
#define VER_EXT_FEAT(ver, ext, feat)                                           \
    {ver, ext, reinterpret_cast<VkBaseOutStructure*>(&feat)}
            VER_EXT_FEAT(VK_MAKE_VERSION(1, 2, 0),
                         VK_KHR_UNIFORM_BUFFER_STANDARD_LAYOUT_EXTENSION_NAME,
                         m_features_ubo_stdlayout),
            VER_EXT_FEAT(VK_MAKE_VERSION(1, 2, 0),
                         VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
                         m_features_float16_int8),
            VER_EXT_FEAT(VK_MAKE_VERSION(1, 1, 0),
                         VK_KHR_VARIABLE_POINTERS_EXTENSION_NAME,
                         m_features_variable_pointer),
            VER_EXT_FEAT(VK_MAKE_VERSION(1, 2, 0),
                         VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
                         m_features_8bit_storage),
            VER_EXT_FEAT(VK_MAKE_VERSION(1, 1, 0),
                         VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
                         m_features_16bit_storage),
            VER_EXT_FEAT(VK_MAKE_VERSION(1, 2, 0),
                         VK_KHR_SHADER_SUBGROUP_EXTENDED_TYPES_EXTENSION_NAME,
                         m_features_shader_subgroup_extended_types),
            VER_EXT_FEAT(VK_MAKE_VERSION(1, 3, 0),
                         VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,
                         m_features_subgroup_size_control),
            VER_EXT_FEAT(VK_MAKE_VERSION(1, 2, 0),
                         VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME,
                         m_features_vulkan_memory_model),
            VER_EXT_FEAT(VK_MAKE_VERSION(1, 2, 0),
                         VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
                         m_features_buffer_device_address),
            VER_EXT_FEAT(VK_MAKE_VERSION(1, 3, 0), nullptr,
                         m_features_shader_integer_dot_product),
            VER_EXT_FEAT(0, VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME,
                         m_features_queue_global_priority),

#undef VER_EXT_FEAT
        };

    VkBaseOutStructure* pNext = nullptr;
    for (auto& ver_ext_feat : coreversion_extension_features) {
        auto corever = std::get<0>(ver_ext_feat);
        auto ext = std::get<1>(ver_ext_feat);
        auto feat = std::get<2>(ver_ext_feat);
        if ((corever != 0) && (m_properties.apiVersion >= corever)) {
            feat->pNext = pNext;
            pNext = feat;
        } else if ((ext != nullptr) && is_vulkan_extension_enabled(ext)) {
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
    cvk_info(
        "subgroup extended types: %d",
        m_features_shader_subgroup_extended_types.shaderSubgroupExtendedTypes);

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

void cvk_device::init_command_pointers(VkInstance instance) {
    // Buffer device address
    if (m_properties.apiVersion >= VK_MAKE_VERSION(1, 2, 0)) {
        m_vkfns.vkGetBufferDeviceAddressKHR =
            GET_INSTANCE_PROC(instance, vkGetBufferDeviceAddress);
    } else if (is_vulkan_extension_enabled(
                   VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)) {
        m_vkfns.vkGetBufferDeviceAddressKHR =
            GET_INSTANCE_PROC(instance, vkGetBufferDeviceAddressKHR);
    }
}

void cvk_device::init_compiler_options() {
    m_device_compiler_options = "";

    if (!devices_support_images()) {
        m_device_compiler_options += " -images=0 ";
    }

    // 8-bit storage capability restrictions.
    if (device_8bit_storage_features().storageBuffer8BitAccess == VK_FALSE) {
        m_device_compiler_options += " -no-8bit-storage=ssbo ";
    }
    if (device_8bit_storage_features().uniformAndStorageBuffer8BitAccess ==
        VK_FALSE) {
        m_device_compiler_options += " -no-8bit-storage=ubo ";
    }
    if (device_8bit_storage_features().storagePushConstant8 == VK_FALSE) {
        m_device_compiler_options += " -no-8bit-storage=pushconstant ";
    }

    // 16-bit storage capability restrictions.
    if (device_16bit_storage_features().storageBuffer16BitAccess == VK_FALSE) {
        m_device_compiler_options += " -no-16bit-storage=ssbo ";
    }
    if (device_16bit_storage_features().uniformAndStorageBuffer16BitAccess ==
        VK_FALSE) {
        m_device_compiler_options += " -no-16bit-storage=ubo ";
    }
    if (device_16bit_storage_features().storagePushConstant16 == VK_FALSE) {
        m_device_compiler_options += " -no-16bit-storage=pushconstant ";
    }
    std::vector<std::string> roundingModeRTE;
    if (m_float_controls_properties.shaderRoundingModeRTEFloat16 &&
        supports_fp16()) {
        roundingModeRTE.push_back("16");
    }
    if (m_float_controls_properties.shaderRoundingModeRTEFloat32) {
        roundingModeRTE.push_back("32");
    }
    if (m_float_controls_properties.shaderRoundingModeRTEFloat64 &&
        supports_fp64()) {
        roundingModeRTE.push_back("64");
    }
    if (roundingModeRTE.size() > 0) {
        m_device_compiler_options += " -rounding-mode-rte=";
        for (unsigned i = 0; i < roundingModeRTE.size(); i++) {
            if (i != 0) {
                m_device_compiler_options += ",";
            }
            m_device_compiler_options += roundingModeRTE[i];
        }
        m_device_compiler_options += " ";
    }

    // Types support
    if (!supports_fp16()) {
        m_device_compiler_options += " -fp16=0 ";
    }
    if (!supports_fp64()) {
        m_device_compiler_options += " -fp64=0 ";
    }
    if (supports_int8()) {
        m_device_compiler_options += " -rewrite-packed-structs ";
    } else {
        m_device_compiler_options += " -int8=0 ";
    }
    if (supports_ubo_stdlayout()) {
        m_device_compiler_options += " -std430-ubo-layout ";
    }
    if (supports_non_uniform_decoration()) {
        m_device_compiler_options += " -decorate-nonuniform ";
    }

    // Device specific options
    m_device_compiler_options +=
        " " + m_clvk_properties->get_compile_options() + " ";

    m_device_compiler_options += " -arch=" + m_spirv_arch + " ";

    if (m_physical_addressing) {
        m_device_compiler_options += " -physical-storage-buffers ";
    }

    if (supports_dot_product()) {
        m_device_compiler_options += " -cl-arm-integer-dot-product ";
    }

    // Builtin options
    auto parse_builtins = [](std::string s) {
        std::set<std::string> builtins;
        size_t pos = 0;
        size_t comma = s.find(',', pos);
        while (comma != std::string::npos) {
            builtins.insert(s.substr(pos, comma - pos));
            pos = comma + 1;
            comma = s.find(',', pos);
        }
        builtins.insert(s.substr(pos));
        return builtins;
    };
    auto clspv_library_builtins =
        parse_builtins(config.clspv_library_builtins());
    auto native_builtins = m_clvk_properties->get_native_builtins();
    auto clspv_native_builtins = parse_builtins(config.clspv_native_builtins());
    native_builtins.insert(clspv_native_builtins.begin(),
                           clspv_native_builtins.end());

    std::string builtin_list = "";
    for (const auto& builtin : native_builtins) {
        if (clspv_library_builtins.count(builtin) > 0) {
            continue;
        }
        builtin_list += builtin + ",";
    }
    if (builtin_list != "") {
        m_device_compiler_options +=
            " --use-native-builtins=" + builtin_list + " ";
    }

    // Select target SPIR-V version
    m_device_compiler_options += " -spv-version=";
    switch (vulkan_spirv_env()) {
    default:
    case SPV_ENV_VULKAN_1_0:
        m_device_compiler_options += "1.0 ";
        break;
    case SPV_ENV_VULKAN_1_1:
        m_device_compiler_options += "1.3 ";
        break;
    case SPV_ENV_VULKAN_1_1_SPIRV_1_4:
        m_device_compiler_options += "1.4 ";
        break;
    case SPV_ENV_VULKAN_1_2:
        m_device_compiler_options += "1.5 ";
        break;
    case SPV_ENV_VULKAN_1_3:
        m_device_compiler_options += "1.6 ";
        break;
    }

    // Limits
    m_device_compiler_options +=
        " -max-pushconstant-size=" +
        std::to_string(vulkan_max_push_constants_size()) + " ";
    m_device_compiler_options +=
        " -max-ubo-size=" + std::to_string(vulkan_max_uniform_buffer_range()) +
        " ";
    m_device_compiler_options += " -global-offset ";
    m_device_compiler_options += " -long-vector ";
    m_device_compiler_options += " -module-constants-in-storage-buffer ";
    m_device_compiler_options += " -cl-arm-non-uniform-work-group-size ";
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
#ifdef ENABLE_SPIRV_IL
        MAKE_NAME_VERSION(1, 0, 0, "cl_khr_il_program"),
#endif
        MAKE_NAME_VERSION(1, 0, 0, "cl_khr_spirv_no_integer_wrap_decoration"),
        MAKE_NAME_VERSION(1, 0, 0, "cl_arm_non_uniform_work_group_size"),
        MAKE_NAME_VERSION(1, 0, 0, "cl_arm_printf"),
        MAKE_NAME_VERSION(1, 0, 0, "cl_khr_suggested_local_work_size"),
        MAKE_NAME_VERSION(1, 0, 0, "cl_khr_3d_image_writes"),
        // MAKE_NAME_VERSION(0, 9, 0, "cl_khr_semaphore"),
        MAKE_NAME_VERSION(1, 0, 0, "cl_khr_spirv_linkonce_odr"),
    };

    if (m_properties.apiVersion >= VK_MAKE_VERSION(1, 1, 0)) {
        m_extensions.push_back(
            MAKE_NAME_VERSION(1, 0, 0, "cl_khr_device_uuid"));
        VkSubgroupFeatureFlags required_subgroup_ops =
            VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
        if ((m_subgroup_properties.supportedOperations &
             required_subgroup_ops) == required_subgroup_ops &&
            (m_features_shader_subgroup_extended_types
                 .shaderSubgroupExtendedTypes == VK_TRUE)) {
            m_has_subgroups_support = true;
        }
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

    if ((m_properties.apiVersion >= VK_MAKE_VERSION(1, 3, 0) ||
         is_vulkan_extension_enabled(
             VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME)) &&
        m_features_subgroup_size_control.subgroupSizeControl &&
        (m_subgroup_size_control_properties.requiredSubgroupSizeStages &
         VK_SHADER_STAGE_COMPUTE_BIT)) {
        m_extensions.push_back(
            MAKE_NAME_VERSION(1, 0, 0, "cl_intel_required_subgroup_size"));
        m_has_subgroup_size_selection = true;
    }

    if (supports_dot_product()) {
        if (supports_int8()) {
            m_extensions.push_back(MAKE_NAME_VERSION(
                2, 0, 0, CL_KHR_INTEGER_DOT_PRODUCT_EXTENSION_NAME));
            m_extensions.push_back(
                MAKE_NAME_VERSION(1, 0, 0, "cl_arm_integer_dot_product_int8"));
            m_extensions.push_back(MAKE_NAME_VERSION(
                1, 0, 0, "cl_arm_integer_dot_product_accumulate_int8"));
            m_extensions.push_back(MAKE_NAME_VERSION(
                1, 0, 0,
                "cl_arm_integer_dot_product_accumulate_saturate_int8"));
        }
        m_extensions.push_back(MAKE_NAME_VERSION(
            1, 0, 0, "cl_arm_integer_dot_product_accumulate_int16"));
    }

    auto split_string = [](std::string input, char delimiter) {
        std::vector<std::string> outputs;
        size_t pos = 0;
        while ((pos = input.find(delimiter)) != std::string::npos) {
            outputs.push_back(input.substr(0, pos));
            input.erase(0, pos + 1);
        }
        if (input.size() > 0) {
            outputs.push_back(input);
        }
        return outputs;
    };
    auto config_extensions = split_string(config.device_extensions(), ',');
    for (auto& config_extension : config_extensions) {
        cl_name_version extension;
        extension.version = CL_MAKE_VERSION(0, 0, 0);
        memcpy(extension.name, config_extension.c_str(),
               std::min(config_extension.size(),
                        (size_t)CL_NAME_VERSION_MAX_NAME_SIZE));
        m_extensions.push_back(extension);
    }

    auto config_extensions_masked =
        split_string(config.device_extensions_masked(), ',');
    for (auto& config_extension_masked : config_extensions_masked) {
        for (auto it = m_extensions.begin(); it != m_extensions.end(); it++) {
            if (strcmp(config_extension_masked.c_str(), it->name) == 0) {
                m_extensions.erase(it);
                break;
            }
        }
    }

    // Build extension string
    for (auto& ext : m_extensions) {
        m_extension_string += ext.name;
        m_extension_string += " ";
    }
    cvk_info_fn("extensions: '%s'", m_extension_string.c_str());

    // Build list of ILs
    m_ils = {
#ifdef ENABLE_SPIRV_IL
        MAKE_NAME_VERSION(1, 0, 0, "SPIR-V"),
#endif
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
        MAKE_NAME_VERSION(3, 0, 0, "__opencl_c_3d_image_writes"),
        // TODO(#216) re-enable when clspv ready
        // MAKE_NAME_VERSION(3, 0, 0, "__opencl_c_generic_address_space"),
    };
    if (supports_read_write_images()) {
        m_opencl_c_features.push_back(
            MAKE_NAME_VERSION(3, 0, 0, "__opencl_c_read_write_images"));
    }
    if (supports_atomic_order_acq_rel()) {
        m_opencl_c_features.push_back(
            MAKE_NAME_VERSION(3, 0, 0, "__opencl_c_atomic_order_acq_rel"));
    }
    if (supports_atomic_scope_device()) {
        m_opencl_c_features.push_back(
            MAKE_NAME_VERSION(3, 0, 0, "__opencl_c_atomic_scope_device"));
    }
    if (supports_subgroups()) {
        m_opencl_c_features.push_back(
            MAKE_NAME_VERSION(3, 0, 0, "__opencl_c_subgroups"));
    }
    if (m_features.features.shaderInt64) {
        m_opencl_c_features.push_back(
            MAKE_NAME_VERSION(3, 0, 0, "__opencl_c_int64"));
    }
    if (m_features.features.shaderFloat64) {
        m_has_fp64_support = true;
        m_opencl_c_features.push_back(
            MAKE_NAME_VERSION(3, 0, 0, "__opencl_c_fp64"));
    }
    if (supports_dot_product()) {
        if (supports_int8()) {
            m_opencl_c_features.push_back(MAKE_NAME_VERSION(
                3, 0, 0, "__opencl_c_integer_dot_product_input_4x8bit"));
        }
        m_opencl_c_features.push_back(MAKE_NAME_VERSION(
            3, 0, 0, "__opencl_c_integer_dot_product_input_4x8bit_packed"));
    }
}

bool cvk_device::create_vulkan_queues_and_device(uint32_t num_queues,
                                                 uint32_t queue_family) {
    cvk_info("Creating Vulkan device and queues");
    // Give all queues the same priority
    std::vector<float> queuePriorities(num_queues, 1.0f);
    void* pNext = nullptr;
    VkDeviceQueueGlobalPriorityCreateInfoKHR globalPriorityCreateInfo;
    cvk_info_fn("queue global priority: %u",
                m_features_queue_global_priority.globalPriorityQuery);
    if (m_features_queue_global_priority.globalPriorityQuery) {
        const uint32_t num_priorities = 4;
        static const VkQueueGlobalPriorityKHR priorities[num_priorities] = {
            VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR,
            VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR,
            VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR,
            VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR};
        uint32_t queue_priority =
            std::min(config.queue_global_priority(), num_priorities - 1);

        globalPriorityCreateInfo.sType =
            VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR;
        globalPriorityCreateInfo.pNext = nullptr;
        globalPriorityCreateInfo.globalPriority = priorities[queue_priority];
        pNext = &globalPriorityCreateInfo;

        cvk_info_fn("setting queue global priority to '%u': '%s' (%u)",
                    queue_priority,
                    queue_global_priority_to_string(
                        globalPriorityCreateInfo.globalPriority),
                    globalPriorityCreateInfo.globalPriority);
    }

    VkDeviceQueueCreateInfo queueCreateInfo = {
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        pNext,
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
    if (config.cache_dir().empty()) {
        return "";
    }

    // The pipeline cache file path is:
    // ${CLVK_CACHE_DIR}/clvk-pipeline-cache.<UUID>.<SHA1>.bin
    std::string cache_path = config.cache_dir;
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
    } else if (m_properties.apiVersion < VK_MAKE_VERSION(1, 3, 0)) {
        m_vulkan_spirv_env = SPV_ENV_VULKAN_1_2;
    } else {
        // Assume 1.3
        m_vulkan_spirv_env = SPV_ENV_VULKAN_1_3;
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

    init_clvk_runtime_behaviors();

    init_vulkan_properties(instance);

    init_driver_behaviors();

    init_features(instance);

    init_command_pointers(instance);

    build_extension_ils_list();

    if (!init_time_management(instance)) {
        return false;
    }

    if (!create_vulkan_queues_and_device(num_queues, queue_family)) {
        return false;
    }

    init_spirv_environment();

    log_limits_and_memory_information();

    // Must be done last as it relies on info set up in several of the above.
    init_compiler_options();

    return true;
}

std::string cvk_device::vendor() const {
    // Is this a Khronos vendor ID?
    if (m_properties.vendorID > 0xFFFF) {
        return vulkan_vendor_id_string(
            static_cast<VkVendorId>(m_properties.vendorID));
    }

    // If not, we are looking at a PCI Vendor ID.
    // TODO support looking it up in the PCI DB.

    return m_clvk_properties->vendor();
}

bool cvk_device::supports_capability(spv::Capability capability) const {
    switch (capability) {
    // Capabilities required by all Vulkan implementations:
    case spv::CapabilityShader:
    case spv::CapabilitySampled1D:
    case spv::CapabilityImage1D:
    case spv::CapabilityImageQuery:
    case spv::CapabilityImageBuffer:
    case spv::CapabilitySampledBuffer:
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
    case spv::CapabilityGroupNonUniform:
        return m_subgroup_properties.supportedOperations &
               VK_SUBGROUP_FEATURE_BASIC_BIT;
    case spv::CapabilityGroupNonUniformVote:
        return m_subgroup_properties.supportedOperations &
               VK_SUBGROUP_FEATURE_VOTE_BIT;
    case spv::CapabilityGroupNonUniformArithmetic:
        return m_subgroup_properties.supportedOperations &
               VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
    case spv::CapabilityGroupNonUniformBallot:
        return m_subgroup_properties.supportedOperations &
               VK_SUBGROUP_FEATURE_BALLOT_BIT;
    case spv::CapabilityVulkanMemoryModel:
        return m_features_vulkan_memory_model.vulkanMemoryModel;
    case spv::CapabilityShaderNonUniform:
        return supports_non_uniform_decoration();
    case spv::CapabilityPhysicalStorageBufferAddresses:
        return m_features_buffer_device_address.bufferDeviceAddress;
    case spv::CapabilityRoundingModeRTE:
        return m_float_controls_properties.shaderRoundingModeRTEFloat32 ||
               m_float_controls_properties.shaderRoundingModeRTEFloat16 ||
               m_float_controls_properties.shaderRoundingModeRTEFloat64;
    case spv::CapabilityDotProduct:
    case spv::CapabilityDotProductInput4x8BitPacked:
        return supports_dot_product();
    case spv::CapabilityDotProductInput4x8Bit:
    case spv::CapabilityDotProductInputAll:
        return supports_dot_product() && supports_int8();
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
        cvk_error_fn("vkGetCalibratedTimestampsEXT failed %d %s", res,
                     vulkan_error_string(res));
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
