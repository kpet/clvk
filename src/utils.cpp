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

#include "utils.hpp"
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

#ifdef __APPLE__
#include <unistd.h>
#endif

#ifdef WIN32
#include <Windows.h>
#include <io.h>
#endif

char* cvk_mkdtemp(std::string& tmpl)
{
#ifdef WIN32
    if (_mktemp_s(&tmpl.front(), tmpl.size() + 1) != 0) {
        return nullptr;
    }

    if (!CreateDirectory(tmpl.c_str(), nullptr)) {
        return nullptr;
    }

    return &tmpl.front();
#else
    return mkdtemp(&tmpl.front());
#endif
}

const char* vulkan_error_string(VkResult result) {
    switch(result) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_NOT_READY: return "VK_NOT_READY";
    case VK_TIMEOUT: return "VK_TIMEOUT";
    case VK_EVENT_SET: return "VK_EVENT_SET";
    case VK_EVENT_RESET: return "VK_EVENT_RESET";
    case VK_INCOMPLETE: return "VK_INCOMPLETE";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
    case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
    case VK_ERROR_SURFACE_LOST_KHR: return "VK_ERROR_FRAGMENTED_POOL";
    case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR: return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
    case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";
    case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
    case VK_ERROR_INCOMPATIBLE_DISPLAY_KHR: return "VK_ERROR_INCOMPATIBLE_DISPLAY_KHR";
    case VK_ERROR_VALIDATION_FAILED_EXT: return "VK_ERROR_VALIDATION_FAILED_EXT";
    case VK_ERROR_INVALID_SHADER_NV: return "VK_ERROR_INVALID_SHADER_NV";
    default: return "Unknown vulkan error";
    }
}

std::string vulkan_memory_property_flags_string(VkMemoryPropertyFlags flags) {
    std::string str;

    if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        str += "HOST_VISIBLE ";
    }

    if (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
        str += "HOST_COHERENT ";
    }

    if (flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) {
        str += "HOST_CACHED ";
    }

    if (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
        str += "DEVICE_LOCAL ";
    }

    if (flags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) {
        str += "LAZILY_ALLOCATED ";
    }

    return str;
}

std::string vulkan_queue_flags_string(VkQueueFlags flags) {
    std::string str;

    if (flags & VK_QUEUE_GRAPHICS_BIT) {
        str += "GRAPHICS ";
    }

    if (flags & VK_QUEUE_COMPUTE_BIT) {
        str += "COMPUTE ";
    }

    if (flags & VK_QUEUE_TRANSFER_BIT) {
        str += "TRANSFER ";
    }

    if (flags & VK_QUEUE_SPARSE_BINDING_BIT) {
        str += "SPARSE_BINDING ";
    }

    return str;
}

std::string vulkan_physical_device_type_string(VkPhysicalDeviceType type) {
    switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_OTHER:
        return "Other";
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        return "Integrated GPU";
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        return "Discrete GPU";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        return "Virtual GPU";
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
        return "CPU";
    default:
        return "Unknown";
    }
}

std::string pretty_size(uint64_t size) {
    std::string ret;

    static const char *units[] = {" B","kB", "MB", "GB", "TB"};
    int unit = 0;
    float fsize = size;
    while (fsize > 1024) {
        unit++;
        fsize /= 1024;
    }

    ret = std::to_string(fsize) + " " + units[unit];

    return ret;
}

static const char colourRed[] = "\e[0;31m";
static const char colourYellow[] = "\e[0;33m";
static const char colourReset[] = "\e[0m";

void cvk_log(loglevel level, const char *fmt, ...) {

    if (level > gLoggingLevel) {
        return;
    }

    const char *colourCode = nullptr;

    if (gLoggingColour) {

        switch (level) {
        case loglevel::fatal:
        case loglevel::error:
            colourCode = colourRed;
            break;
        case loglevel::warn:
            colourCode = colourYellow;
            break;
        case loglevel::info:
        case loglevel::debug:
            break;
        }

        if (colourCode != nullptr) {
            printf("%s", colourCode);
        }
    }

    printf("[CLVK] ");

    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);

    if ((gLoggingColour) && (colourCode != nullptr)) {
        printf("%s", colourReset);
    }

    if (level == loglevel::fatal) {
        exit(EXIT_FAILURE);
    }
}
