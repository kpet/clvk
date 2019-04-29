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
#include "memory.hpp"
#include "objects.hpp"
#include "utils.hpp"

VkInstance gVkInstance;
cvk_platform *gPlatform;
loglevel gLoggingLevel = loglevel::fatal;
bool gLoggingColour = true;
bool gDebugReportEnabled = false;

#ifndef CLSPV_ONLINE_COMPILER
std::string gCLSPVPath = DEFAULT_CLSPV_BINARY_PATH;
#endif

static VkDebugReportCallbackEXT gVkDebugCallback;

VkBool32 debugCallback(
    VkDebugReportFlagsEXT                       flags,
    VkDebugReportObjectTypeEXT                  objectType,
    uint64_t                                    object,
    size_t                                      location,
    int32_t                                     messageCode,
    const char*                                 pLayerPrefix,
    const char*                                 pMessage,
    void*                                       pUserData
){
    UNUSED(objectType);
    UNUSED(object);
    UNUSED(location);
    UNUSED(messageCode);
    UNUSED(pLayerPrefix);
    UNUSED(pUserData);

    if (flags & VK_DEBUG_REPORT_ERROR_BIT_EXT) {
        cvk_error("%s", pMessage);
    } else if ((flags & VK_DEBUG_REPORT_WARNING_BIT_EXT) || (flags & VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT)) {
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

static void init_vulkan()
{
    VkResult res;

    // Print layer info
    uint32_t numLayerProperties;
    res = vkEnumerateInstanceLayerProperties(&numLayerProperties, nullptr);
    CVK_VK_CHECK_FATAL(res, "Could not query layers");
    cvk_info("%u layer properties are available", numLayerProperties);

    std::vector<VkLayerProperties> layerProperties(numLayerProperties);
    res = vkEnumerateInstanceLayerProperties(&numLayerProperties, layerProperties.data());
    CVK_VK_CHECK_FATAL(res, "Could not query layers");

    for (uint32_t i = 0; i < numLayerProperties; i++) {
        cvk_info("Found layer %s, spec version %s, impl version %u",
                layerProperties[i].layerName,
                vulkan_version_string(layerProperties[i].specVersion).c_str(),
                layerProperties[i].implementationVersion);
    }

    // Print extension info
    uint32_t numExtensionProperties;
    res = vkEnumerateInstanceExtensionProperties(nullptr, &numExtensionProperties, nullptr);
    CVK_VK_CHECK_FATAL(res, "Could not query extensions");
    cvk_info("%u extension properties are available", numExtensionProperties);

    std::vector<VkExtensionProperties> extensionProperties(numExtensionProperties);
    res = vkEnumerateInstanceExtensionProperties(nullptr, &numExtensionProperties, extensionProperties.data());
    CVK_VK_CHECK_FATAL(res, "Could not query extensions");

    std::vector<const char*> enabledExtensions = {
        "VK_KHR_get_physical_device_properties2",
    };

    for (size_t i = 0; i < numExtensionProperties; i++) {
        cvk_info("Found extension %s, spec version %u",
                extensionProperties[i].extensionName,
                extensionProperties[i].specVersion);

        if (!strcmp(extensionProperties[i].extensionName, "VK_EXT_debug_report")) {
            enabledExtensions.push_back("VK_EXT_debug_report");
            gDebugReportEnabled = true;
        }
    }

    // Handle validation layers
    std::vector<const char*> enabledLayers;

    char *enable_validation_layers = getenv("CVK_VALIDATION_LAYERS");
    if (enable_validation_layers != nullptr) {
        int value = atoi(enable_validation_layers);
        if (value == 1) {
            enabledLayers.push_back("VK_LAYER_LUNARG_standard_validation");
        }
    }

    // Create the instance
    VkInstanceCreateInfo info = {
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, // sType
        nullptr, // pNext
        0, // flags
        nullptr, // pApplicationInfo
        static_cast<uint32_t>(enabledLayers.size()), // enabledLayerCount
        enabledLayers.data(), // ppEnabledLayerNames
        static_cast<uint32_t>(enabledExtensions.size()), // enabledExtensionCount
        enabledExtensions.data(), // ppEnabledExtensionNames
    };

    res = vkCreateInstance(&info, nullptr, &gVkInstance);
    CVK_VK_CHECK_FATAL(res, "Could not create the instance");
    cvk_info("Created the VkInstance");

    // Create debug callback
    VkDebugReportCallbackCreateInfoEXT callbackInfo = {
            VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,    // sType
            NULL,                                                       // pNext
            VK_DEBUG_REPORT_ERROR_BIT_EXT |                             // flags
            VK_DEBUG_REPORT_DEBUG_BIT_EXT |
            VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT |
            VK_DEBUG_REPORT_INFORMATION_BIT_EXT |
            VK_DEBUG_REPORT_WARNING_BIT_EXT,
            debugCallback,                                        // pfnCallback
            NULL                                                        // pUserData
    };

    if (gDebugReportEnabled) {
        CVK_VK_GET_INSTANCE_PROC(vkCreateDebugReportCallbackEXT);

        res = fnvkCreateDebugReportCallbackEXT(gVkInstance, &callbackInfo, nullptr, &gVkDebugCallback);
        CVK_VK_CHECK_FATAL(res, "Can't setup debug callback");
    }
    else {
        cvk_warn("VK_EXT_debug_report not enabled");
    }
}

static void term_vulkan()
{
    if (gDebugReportEnabled) {
        CVK_VK_GET_INSTANCE_PROC(vkDestroyDebugReportCallbackEXT);
        fnvkDestroyDebugReportCallbackEXT(gVkInstance, gVkDebugCallback, nullptr);
    }
    vkDestroyInstance(gVkInstance, nullptr);
}

static void init_logging()
{
    char *logging = getenv("CVK_LOG");
    if (logging) {
        loglevel setting = static_cast<loglevel>(atoi(logging));
        if ((setting < loglevel::fatal) || (setting > loglevel::debug)) {
            setting = loglevel::error;
        }
        gLoggingLevel = setting;
    }

    char *logging_colour = getenv("CVK_LOG_COLOUR");
    if (logging_colour) {
        int val = atoi(logging_colour);
        if (val == 0) {
            gLoggingColour = false;
        }
    }
}

static void init_compiler()
{
#ifndef CLSPV_ONLINE_COMPILER
    char *clspv_binary = getenv("CVK_CLSPV_BIN");
    if (clspv_binary != nullptr) {
        gCLSPVPath = clspv_binary;
    }
#endif
}

static void init_platform()
{
    gPlatform = new cvk_platform();

    uint32_t numDevices;
    VkResult res = vkEnumeratePhysicalDevices(gVkInstance, &numDevices, nullptr);
    CVK_VK_CHECK_FATAL(res, "Could not enumerate physical devices");
    cvk_info("Found %u physical devices", numDevices);

    std::vector<VkPhysicalDevice> physicalDevices(numDevices);
    res = vkEnumeratePhysicalDevices(gVkInstance, &numDevices, physicalDevices.data());
    CVK_VK_CHECK_FATAL(res, "Could not enumerate physical devices");

    for (uint32_t i = 0; i < numDevices; ++i) {

        auto dev = cvk_device::create(physicalDevices[i]);

        if (dev != nullptr) {
            gPlatform->devices.push_back(dev);
        }
    }

    if (gPlatform->devices.size() == 0) {
        cvk_fatal("Could not initialise any device!");
    } else {
        cvk_info("Initialised %zu devices", gPlatform->devices.size());
    }
}

static void term_platform()
{
    for (auto d : gPlatform->devices) {
        delete d;
    }
}


class clvk_initializer {
public:
    clvk_initializer()
    {
        init_logging();
        cvk_info("Starting initialisation");
        init_compiler();
        init_vulkan();
        init_platform();
        cvk_info("Initialisation complete");
    }

    ~clvk_initializer()
    {
        term_platform();
        term_vulkan();
    }
} gInitializer;
