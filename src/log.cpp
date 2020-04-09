// Copyright 2020 The clvk authors.
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

#include "log.hpp"

#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#ifndef _MSC_VER
#include <unistd.h>
#endif

static int gLoggingLevel;
static bool gLoggingColour;
static FILE* gLoggingFile;

void init_logging() {
    char* logging = getenv("CLVK_LOG");
    if (logging) {
        loglevel setting = static_cast<loglevel>(atoi(logging));
        if ((setting < loglevel::fatal) || (setting > loglevel::debug)) {
            // FIXME handle all errors
            fprintf(stderr, "FATAL: Unknown log level '%s'.\n", logging);
            exit(EXIT_FAILURE);
        }
        gLoggingLevel = setting;
    } else {
        gLoggingLevel = loglevel::fatal;
    }

    char* logging_dest = getenv("CLVK_LOG_DEST");
    if (logging_dest) {

        std::string val(logging_dest);

        if (val == "stdout") {
            gLoggingFile = stdout;
        } else if (val == "stderr") {
            gLoggingFile = stderr;
        } else if (val.rfind("file:", 0) == 0) {

            val.erase(0, strlen("file:"));

            gLoggingFile = fopen(val.c_str(), "w+");

            if (gLoggingFile == nullptr) {
                fprintf(stderr, "FATAL: Could not open log file '%s': %s.\n",
                        val.c_str(), strerror(errno));
                exit(EXIT_FAILURE);
            }
        } else {
            fprintf(stderr, "FATAL: Unknown log destination '%s'.\n",
                    val.c_str());
            exit(EXIT_FAILURE);
        }
    } else {
        gLoggingFile = stderr;
    }

    bool isTTY = isatty(fileno(gLoggingFile));
    if (isTTY) {
        gLoggingColour = true;
    } else {
        gLoggingColour = false;
    }

    char* logging_colour = getenv("CLVK_LOG_COLOUR");
    if (logging_colour) {
        int val = atoi(logging_colour);
        // FIXME handle errors
        if (val == 0) {
            gLoggingColour = false;
        } else {
            gLoggingColour = true;
        }
    }
}

void term_logging() {
    if ((gLoggingFile != stdout) && (gLoggingFile != stderr)) {
        fclose(gLoggingFile);
    }
}

bool cvk_log_level_enabled(loglevel level) {
    return gLoggingLevel >= level;
}

static const char colourRed[] = "\e[0;31m";
static const char colourYellow[] = "\e[0;33m";
static const char colourReset[] = "\e[0m";

void cvk_log(loglevel level, const char* fmt, ...) {

    if (!cvk_log_level_enabled(level)) {
        return;
    }

    const char* colourCode = nullptr;

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
            fprintf(gLoggingFile, "%s", colourCode);
        }
    }

    fprintf(gLoggingFile, "[CLVK] ");

    va_list args;
    va_start(args, fmt);
    vfprintf(gLoggingFile, fmt, args);
    va_end(args);

    if ((gLoggingColour) && (colourCode != nullptr)) {
        fprintf(gLoggingFile, "%s", colourReset);
    }

    if (level == loglevel::fatal) {
        exit(EXIT_FAILURE);
    }
}

std::string pretty_size(uint64_t size) {
    std::string ret;

    static const char* units[] = {" B", "kB", "MB", "GB", "TB"};
    int unit = 0;
    float fsize = size;
    while (fsize > 1024) {
        unit++;
        fsize /= 1024;
    }

    ret = std::to_string(fsize) + " " + units[unit];

    return ret;
}

const char* vulkan_error_string(VkResult result) {
    switch (result) {
#define CASE(X)                                                                \
    case X:                                                                    \
        return #X;
        CASE(VK_SUCCESS)
        CASE(VK_NOT_READY)
        CASE(VK_TIMEOUT)
        CASE(VK_EVENT_SET)
        CASE(VK_EVENT_RESET)
        CASE(VK_INCOMPLETE)
        CASE(VK_ERROR_OUT_OF_HOST_MEMORY)
        CASE(VK_ERROR_OUT_OF_DEVICE_MEMORY)
        CASE(VK_ERROR_INITIALIZATION_FAILED)
        CASE(VK_ERROR_DEVICE_LOST)
        CASE(VK_ERROR_MEMORY_MAP_FAILED)
        CASE(VK_ERROR_LAYER_NOT_PRESENT)
        CASE(VK_ERROR_EXTENSION_NOT_PRESENT)
        CASE(VK_ERROR_FEATURE_NOT_PRESENT)
        CASE(VK_ERROR_INCOMPATIBLE_DRIVER)
        CASE(VK_ERROR_TOO_MANY_OBJECTS)
        CASE(VK_ERROR_FORMAT_NOT_SUPPORTED)
        CASE(VK_ERROR_FRAGMENTED_POOL)
        CASE(VK_ERROR_SURFACE_LOST_KHR)
        CASE(VK_ERROR_NATIVE_WINDOW_IN_USE_KHR)
        CASE(VK_SUBOPTIMAL_KHR)
        CASE(VK_ERROR_OUT_OF_DATE_KHR)
        CASE(VK_ERROR_INCOMPATIBLE_DISPLAY_KHR)
        CASE(VK_ERROR_VALIDATION_FAILED_EXT)
        CASE(VK_ERROR_INVALID_SHADER_NV)
        CASE(VK_ERROR_INVALID_EXTERNAL_HANDLE)
        CASE(VK_ERROR_FRAGMENTATION_EXT)
        CASE(VK_ERROR_NOT_PERMITTED_EXT)
        CASE(VK_ERROR_OUT_OF_POOL_MEMORY_KHR)
    default:
        return "Unknown vulkan error";
#undef CASE
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
