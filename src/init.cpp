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

#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "config.hpp"
#include "device.hpp"
#include "init.hpp"
#include "log.hpp"
#include "memory.hpp"
#include "objects.hpp"
#include "queue.hpp"
#include "tracing.hpp"

static VkBool32 VKAPI_PTR debugCallback(
    VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objectType,
    uint64_t object, size_t location, int32_t messageCode,
    const char* pLayerPrefix, const char* pMessage, void* pUserData) {
    UNUSED(objectType);
    UNUSED(object);
    UNUSED(location);
    UNUSED(messageCode);
    UNUSED(pLayerPrefix);
    UNUSED(pUserData);

    if (flags & VK_DEBUG_REPORT_ERROR_BIT_EXT) {
        cvk_error_group(loggroup::validation, "%s", pMessage);
    } else if ((flags & VK_DEBUG_REPORT_WARNING_BIT_EXT) ||
               (flags & VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT)) {
        cvk_warn_group(loggroup::validation, "%s", pMessage);
    } else if (flags & VK_DEBUG_REPORT_INFORMATION_BIT_EXT) {
        cvk_info_group(loggroup::validation, "%s", pMessage);
    } else if (flags & VK_DEBUG_REPORT_DEBUG_BIT_EXT) {
        cvk_debug_group(loggroup::validation, "%s", pMessage);
    } else {
        cvk_error_group(loggroup::validation, "%s", pMessage);
    }

    return VK_FALSE;
}

void clvk_global_state::init_vulkan() {
    VkResult res;

    // Handle validation layers config
    const char* validation_layers[] = {
        "VK_LAYER_KHRONOS_validation",
        "VK_LAYER_LUNARG_standard_validation",
    };

    if (config.validation_layers) {
        cvk_info("Enabling validation layers.");
    }

    // Discover, log and select layers
    uint32_t numLayerProperties;
    res = vkEnumerateInstanceLayerProperties(&numLayerProperties, nullptr);
    CVK_VK_CHECK_FATAL(res, "Could not query layers");
    cvk_info("%u layers visible", numLayerProperties);

    std::vector<VkLayerProperties> layerProperties(numLayerProperties);
    res = vkEnumerateInstanceLayerProperties(&numLayerProperties,
                                             layerProperties.data());
    CVK_VK_CHECK_FATAL(res, "Could not query layers");

    std::vector<const char*> enabledLayers;
    bool validation_layers_found = false;
    for (uint32_t i = 0; i < numLayerProperties; i++) {
        cvk_info("  %s, spec version %s, impl version %u",
                 layerProperties[i].layerName,
                 vulkan_version_string(layerProperties[i].specVersion).c_str(),
                 layerProperties[i].implementationVersion);
        if (config.validation_layers) {
            for (auto dl : validation_layers) {
                if (!strcmp(layerProperties[i].layerName, dl)) {
                    cvk_info("    ENABLING");
                    enabledLayers.push_back(dl);
                    validation_layers_found = true;
                }
            }
        }
    }

    if (config.validation_layers && !validation_layers_found) {
        cvk_warn("Validation layers are enabled but none have been found");
    }

    // Print extension info
    uint32_t numExtensionProperties;
    res = vkEnumerateInstanceExtensionProperties(
        nullptr, &numExtensionProperties, nullptr);
    CVK_VK_CHECK_FATAL(res, "Could not query extensions");
    cvk_info("%u extensions are supported", numExtensionProperties);

    std::vector<VkExtensionProperties> extensionProperties(
        numExtensionProperties);
    res = vkEnumerateInstanceExtensionProperties(
        nullptr, &numExtensionProperties, extensionProperties.data());
    CVK_VK_CHECK_FATAL(res, "Could not query extensions");

    std::vector<const char*> enabledExtensions;

    const std::vector<const char*> desired_extensions = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_EXT_DEBUG_REPORT_EXTENSION_NAME,
        VK_KHR_EXTERNAL_FENCE_CAPABILITIES_EXTENSION_NAME,
    };

    for (size_t i = 0; i < numExtensionProperties; i++) {
        cvk_info("  %s, spec version %u", extensionProperties[i].extensionName,
                 extensionProperties[i].specVersion);

        for (auto name : desired_extensions) {
            if (!strcmp(name, extensionProperties[i].extensionName)) {
                enabledExtensions.push_back(name);
                cvk_info("    ENABLING");
                break;
            }
        }
    }

    m_debug_report_enabled =
        std::find(enabledExtensions.begin(), enabledExtensions.end(),
                  VK_EXT_DEBUG_REPORT_EXTENSION_NAME) !=
        enabledExtensions.end();

    // Create the instance
    VkApplicationInfo appInfo = {
        VK_STRUCTURE_TYPE_APPLICATION_INFO,
        nullptr,
        "clvk",
        0, // TODO applicationVersion
        "clvk",
        0, // TODO engineVersion
        VK_MAKE_VERSION(1, 3, 0)
    };

    VkInstanceCreateInfo info = {
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,      // sType
        nullptr,                                     // pNext
        0,                                           // flags
        &appInfo,                                    // pApplicationInfo
        static_cast<uint32_t>(enabledLayers.size()), // enabledLayerCount
        enabledLayers.data(),                        // ppEnabledLayerNames
        static_cast<uint32_t>(
            enabledExtensions.size()), // enabledExtensionCount
        enabledExtensions.data(),      // ppEnabledExtensionNames
    };

    res = vkCreateInstance(&info, nullptr, &m_vulkan_instance);
    CVK_VK_CHECK_FATAL(res, "Could not create the instance");
    cvk_info("Created the VkInstance");

    // Create debug callback
    VkDebugReportCallbackCreateInfoEXT callbackInfo = {
        VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT, // sType
        NULL,                                                    // pNext
        VK_DEBUG_REPORT_ERROR_BIT_EXT |                          // flags
            VK_DEBUG_REPORT_DEBUG_BIT_EXT |
            VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT |
            VK_DEBUG_REPORT_INFORMATION_BIT_EXT |
            VK_DEBUG_REPORT_WARNING_BIT_EXT,
        &debugCallback, // pfnCallback
        NULL           // pUserData
    };

    if (m_debug_report_enabled) {
        auto func =
            CVK_VK_GET_INSTANCE_PROC(this, vkCreateDebugReportCallbackEXT);

        res = func(m_vulkan_instance, &callbackInfo, nullptr,
                   &m_vulkan_debug_callback);
        CVK_VK_CHECK_FATAL(res, "Can't setup debug callback");
    } else {
        cvk_warn("VK_EXT_debug_report not enabled");
    }
}

void clvk_global_state::term_vulkan() {
    if (m_debug_report_enabled) {
        auto func =
            CVK_VK_GET_INSTANCE_PROC(this, vkDestroyDebugReportCallbackEXT);
        func(m_vulkan_instance, m_vulkan_debug_callback, nullptr);
    }
    vkDestroyInstance(m_vulkan_instance, nullptr);
}

void clvk_global_state::init_platform() {

    m_platform = new cvk_platform();

    uint32_t numDevices;
    VkResult res =
        vkEnumeratePhysicalDevices(m_vulkan_instance, &numDevices, nullptr);
    CVK_VK_CHECK_FATAL(res, "Could not enumerate physical devices");
    cvk_info("Found %u physical devices", numDevices);

    std::vector<VkPhysicalDevice> physicalDevices(numDevices);
    res = vkEnumeratePhysicalDevices(m_vulkan_instance, &numDevices,
                                     physicalDevices.data());
    CVK_VK_CHECK_FATAL(res, "Could not enumerate physical devices");

    for (uint32_t i = 0; i < numDevices; ++i) {
        if (!m_platform->create_device(m_vulkan_instance, physicalDevices[i])) {
            cvk_error("Could not create CL device from Vulkan device!");
        }
    }

    auto num_devices = m_platform->devices().size();
    if (num_devices == 0) {
        cvk_fatal("Could not initialise any device!");
    } else {
        cvk_info("Initialised %zu devices", num_devices);
    }
}

void clvk_global_state::term_platform() { delete m_platform; }

void clvk_global_state::init_executors() {
    m_thread_pool = new cvk_executor_thread_pool();
}

void clvk_global_state::term_executors() { delete m_thread_pool; }

clvk_global_state::clvk_global_state() {
    // Init the configuration using environment variables before logging so
    // we can enable full logging support before parsing configuration files.
    init_config_from_env_only();
    init_logging();
    init_config();
    cvk_info("Starting initialisation");
    init_tracing();
    init_vulkan();
    init_platform();
    init_executors();
    cvk_info("Initialisation complete");
}

clvk_global_state::~clvk_global_state() {
    if (config.destroy_global_state) {
        term_executors();
        term_platform();
        term_vulkan();
        term_tracing();
        term_logging();
    }
}

static clvk_global_state* gGlobalState;
static std::once_flag gInitOnceFlag;

static void destroy_global_state() { delete gGlobalState; }

static void init_global_state() {
    gGlobalState = new clvk_global_state();
#ifndef WIN32
    if (atexit(destroy_global_state) != 0) {
        cvk_fatal(
            "Could not register global state destructor using atexit()\n");
    }
#endif
}

clvk_global_state* get_or_init_global_state() {
    std::call_once(gInitOnceFlag, init_global_state);
    return gGlobalState;
}
