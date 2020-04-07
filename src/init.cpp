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

#include "device.hpp"
#include "log.hpp"
#include "memory.hpp"
#include "objects.hpp"

VkInstance gVkInstance;
cvk_platform* gPlatform;
bool gDebugReportEnabled = false;
bool gQueueProfilingUsesTimestampQueries = false;

#ifndef CLSPV_ONLINE_COMPILER
std::string gCLSPVPath = DEFAULT_CLSPV_BINARY_PATH;
std::string gLLVMSPIRVPath = DEFAULT_LLVMSPIRV_BINARY_PATH;
#endif
std::string gCLSPVOptions;

static VkDebugReportCallbackEXT gVkDebugCallback;

VkBool32 VKAPI_CALL debugCallback(VkDebugReportFlagsEXT flags,
                                  VkDebugReportObjectTypeEXT objectType,
                                  uint64_t object, size_t location,
                                  int32_t messageCode, const char* pLayerPrefix,
                                  const char* pMessage, void* pUserData) {
    UNUSED(objectType);
    UNUSED(object);
    UNUSED(location);
    UNUSED(messageCode);
    UNUSED(pLayerPrefix);
    UNUSED(pUserData);

    if (flags & VK_DEBUG_REPORT_ERROR_BIT_EXT) {
        cvk_error("%s", pMessage);
    } else if ((flags & VK_DEBUG_REPORT_WARNING_BIT_EXT) ||
               (flags & VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT)) {
        cvk_warn("%s", pMessage);
    } else if (flags & VK_DEBUG_REPORT_INFORMATION_BIT_EXT) {
        cvk_info("%s", pMessage);
    } else if (flags & VK_DEBUG_REPORT_DEBUG_BIT_EXT) {
        cvk_debug("%s", pMessage);
    } else {
        cvk_error("%s", pMessage);
    }

    return VK_FALSE;
}

static void init_vulkan() {
    VkResult res;

    // Print layer info
    uint32_t numLayerProperties;
    res = vkEnumerateInstanceLayerProperties(&numLayerProperties, nullptr);
    CVK_VK_CHECK_FATAL(res, "Could not query layers");
    cvk_info("%u layers visible", numLayerProperties);

    std::vector<VkLayerProperties> layerProperties(numLayerProperties);
    res = vkEnumerateInstanceLayerProperties(&numLayerProperties,
                                             layerProperties.data());
    CVK_VK_CHECK_FATAL(res, "Could not query layers");

    for (uint32_t i = 0; i < numLayerProperties; i++) {
        cvk_info("  %s, spec version %s, impl version %u",
                 layerProperties[i].layerName,
                 vulkan_version_string(layerProperties[i].specVersion).c_str(),
                 layerProperties[i].implementationVersion);
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

    for (size_t i = 0; i < numExtensionProperties; i++) {
        cvk_info("  %s, spec version %u", extensionProperties[i].extensionName,
                 extensionProperties[i].specVersion);

        if (!strcmp(extensionProperties[i].extensionName,
                    VK_EXT_DEBUG_REPORT_EXTENSION_NAME)) {
            enabledExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
            gDebugReportEnabled = true;
        } else if (
            !strcmp(extensionProperties[i].extensionName,
                    VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)) {
            enabledExtensions.push_back(
                VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
        }
    }

    // Handle validation layers
    std::vector<const char*> enabledLayers;

    char* enable_validation_layers = getenv("CLVK_VALIDATION_LAYERS");
    if (enable_validation_layers != nullptr) {
        int value = atoi(enable_validation_layers);
        if (value == 1) {
            enabledLayers.push_back("VK_LAYER_LUNARG_standard_validation");
            enabledLayers.push_back("VK_LAYER_KHRONOS_validation");
            cvk_info("Enabling validation layers.");
        }
    }

    // Create the instance
    VkInstanceCreateInfo info = {
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,      // sType
        nullptr,                                     // pNext
        0,                                           // flags
        nullptr,                                     // pApplicationInfo
        static_cast<uint32_t>(enabledLayers.size()), // enabledLayerCount
        enabledLayers.data(),                        // ppEnabledLayerNames
        static_cast<uint32_t>(
            enabledExtensions.size()), // enabledExtensionCount
        enabledExtensions.data(),      // ppEnabledExtensionNames
    };

    res = vkCreateInstance(&info, nullptr, &gVkInstance);
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
        debugCallback, // pfnCallback
        NULL           // pUserData
    };

    if (gDebugReportEnabled) {
        CVK_VK_GET_INSTANCE_PROC(vkCreateDebugReportCallbackEXT);

        res = fnvkCreateDebugReportCallbackEXT(gVkInstance, &callbackInfo,
                                               nullptr, &gVkDebugCallback);
        CVK_VK_CHECK_FATAL(res, "Can't setup debug callback");
    } else {
        cvk_warn("VK_EXT_debug_report not enabled");
    }
}

static void term_vulkan() {
    if (gDebugReportEnabled) {
        CVK_VK_GET_INSTANCE_PROC(vkDestroyDebugReportCallbackEXT);
        fnvkDestroyDebugReportCallbackEXT(gVkInstance, gVkDebugCallback,
                                          nullptr);
    }
    vkDestroyInstance(gVkInstance, nullptr);
}

static void init_options() {
#ifndef CLSPV_ONLINE_COMPILER
    char* llvmspirv_binary = getenv("CLVK_LLVMSPIRV_BIN");
    if (llvmspirv_binary != nullptr) {
        gLLVMSPIRVPath = llvmspirv_binary;
    }
    char* clspv_binary = getenv("CLVK_CLSPV_BIN");
    if (clspv_binary != nullptr) {
        gCLSPVPath = clspv_binary;
    }
#endif
    auto clspv_options = getenv("CLVK_CLSPV_OPTIONS");
    if (clspv_options != nullptr) {
        gCLSPVOptions = clspv_options;
    }
    auto profiling = getenv("CLVK_QUEUE_PROFILING_USE_TIMESTAMP_QUERIES");
    if (profiling != nullptr) {
        int val = atoi(profiling);
        if (val != 0) {
            gQueueProfilingUsesTimestampQueries = true;
        }
    }
}

static void init_platform() {
    gPlatform = new cvk_platform();

    uint32_t numDevices;
    VkResult res =
        vkEnumeratePhysicalDevices(gVkInstance, &numDevices, nullptr);
    CVK_VK_CHECK_FATAL(res, "Could not enumerate physical devices");
    cvk_info("Found %u physical devices", numDevices);

    std::vector<VkPhysicalDevice> physicalDevices(numDevices);
    res = vkEnumeratePhysicalDevices(gVkInstance, &numDevices,
                                     physicalDevices.data());
    CVK_VK_CHECK_FATAL(res, "Could not enumerate physical devices");

    for (uint32_t i = 0; i < numDevices; ++i) {
        if (!gPlatform->create_device(physicalDevices[i])) {
            cvk_error("Could not create CL device from Vulkan device!");
        }
    }

    auto num_devices = gPlatform->devices().size();
    if (num_devices == 0) {
        cvk_fatal("Could not initialise any device!");
    } else {
        cvk_info("Initialised %zu devices", num_devices);
    }
}

static void term_platform() { delete gPlatform; }

class clvk_initializer {
public:
    clvk_initializer() {
        init_logging();
        cvk_info("Starting initialisation");
        init_options();
        init_vulkan();
        init_platform();
        cvk_info("Initialisation complete");
    }

    ~clvk_initializer() {
        term_platform();
        term_vulkan();
        term_logging();
    }
} gInitializer;
