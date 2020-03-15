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

#pragma once

#include <string>

#include <vulkan/vulkan.h>

enum loglevel
{
    fatal = 0,
    error = 1,
    warn = 2,
    info = 3,
    debug = 4
};

#ifndef _MSC_VER
#define CHECK_PRINTF(index, first) __attribute__((format(printf, index, first)))
#else
#define CHECK_PRINTF(index, first)
#endif

void init_logging();
void term_logging();
void cvk_log(loglevel level, const char* fmt, ...) CHECK_PRINTF(2, 3);
bool cvk_log_level_enabled(loglevel level);

#define cvk_fatal(fmt, ...) cvk_log(loglevel::fatal, fmt "\n", ##__VA_ARGS__)
#define cvk_error(fmt, ...) cvk_log(loglevel::error, fmt "\n", ##__VA_ARGS__)
#define cvk_warn(fmt, ...) cvk_log(loglevel::warn, fmt "\n", ##__VA_ARGS__)
#define cvk_info(fmt, ...) cvk_log(loglevel::info, fmt "\n", ##__VA_ARGS__)
#define cvk_debug(fmt, ...) cvk_log(loglevel::debug, fmt "\n", ##__VA_ARGS__)

#define cvk_fatal_fn(fmt, ...) cvk_fatal("%s: " fmt, __func__, ##__VA_ARGS__)
#define cvk_error_fn(fmt, ...) cvk_error("%s: " fmt, __func__, ##__VA_ARGS__)
#define cvk_warn_fn(fmt, ...) cvk_warn("%s: " fmt, __func__, ##__VA_ARGS__)
#define cvk_info_fn(fmt, ...) cvk_info("%s: " fmt, __func__, ##__VA_ARGS__)
#define cvk_debug_fn(fmt, ...) cvk_debug("%s: " fmt, __func__, ##__VA_ARGS__)

const char* vulkan_error_string(VkResult result);
std::string pretty_size(uint64_t size);
std::string vulkan_memory_property_flags_string(VkMemoryPropertyFlags flags);
std::string vulkan_queue_flags_string(VkQueueFlags flags);
std::string vulkan_physical_device_type_string(VkPhysicalDeviceType type);
std::string vulkan_calibrateable_time_domain_string(VkTimeDomainEXT td);

static inline std::string vulkan_version_string(uint32_t version) {
    std::string ret = std::to_string(VK_VERSION_MAJOR(version));
    ret += "." + std::to_string(VK_VERSION_MINOR(version));
    ret += "." + std::to_string(VK_VERSION_PATCH(version));
    return ret;
}
