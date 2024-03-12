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

#include <spirv/unified1/spirv.hpp>
#include <string>

#include "cl_headers.hpp"
#include <vulkan/vulkan.h>

enum loglevel
{
    fatal = 0,
    error = 1,
    warn = 2,
    info = 3,
    debug = 4
};

enum loggroup : uint64_t
{
    refcounting = (1ULL << 0),
    api = (1ULL << 1),
    event = (1ULL << 2),
    validation = (1ULL << 3),
    none = (1ULL << 63),
    all = ~0ULL
};

#ifndef _MSC_VER
#define CHECK_PRINTF(index, first) __attribute__((format(printf, index, first)))
#else
#define CHECK_PRINTF(index, first)
#endif

void init_logging();
void term_logging();
void cvk_log(uint64_t group_mask, loglevel level, const char* fmt, ...)
    CHECK_PRINTF(3, 4);
bool cvk_log_level_enabled(loglevel level);
bool cvk_log_group_enabled(uint64_t group_mask);

#define cvk_fatal(fmt, ...)                                                    \
    cvk_log(loggroup::none, loglevel::fatal, fmt "\n", ##__VA_ARGS__)
#define cvk_error(fmt, ...)                                                    \
    cvk_log(loggroup::none, loglevel::error, fmt "\n", ##__VA_ARGS__)
#define cvk_warn(fmt, ...)                                                     \
    cvk_log(loggroup::none, loglevel::warn, fmt "\n", ##__VA_ARGS__)
#define cvk_info(fmt, ...)                                                     \
    cvk_log(loggroup::none, loglevel::info, fmt "\n", ##__VA_ARGS__)
#define cvk_debug(fmt, ...)                                                    \
    cvk_log(loggroup::none, loglevel::debug, fmt "\n", ##__VA_ARGS__)

#define cvk_fatal_fn(fmt, ...) cvk_fatal("%s: " fmt, __func__, ##__VA_ARGS__)
#define cvk_error_fn(fmt, ...) cvk_error("%s: " fmt, __func__, ##__VA_ARGS__)
#define cvk_warn_fn(fmt, ...) cvk_warn("%s: " fmt, __func__, ##__VA_ARGS__)
#define cvk_info_fn(fmt, ...) cvk_info("%s: " fmt, __func__, ##__VA_ARGS__)
#define cvk_debug_fn(fmt, ...) cvk_debug("%s: " fmt, __func__, ##__VA_ARGS__)

#define cvk_fatal_group(mask, fmt, ...)                                        \
    cvk_log(mask, loglevel::fatal, fmt "\n", ##__VA_ARGS__)
#define cvk_error_group(mask, fmt, ...)                                        \
    cvk_log(mask, loglevel::error, fmt "\n", ##__VA_ARGS__)
#define cvk_warn_group(mask, fmt, ...)                                         \
    cvk_log(mask, loglevel::warn, fmt "\n", ##__VA_ARGS__)
#define cvk_info_group(mask, fmt, ...)                                         \
    cvk_log(mask, loglevel::info, fmt "\n", ##__VA_ARGS__)
#define cvk_debug_group(mask, fmt, ...)                                        \
    cvk_log(mask, loglevel::debug, fmt "\n", ##__VA_ARGS__)

#define cvk_fatal_group_fn(mask, fmt, ...)                                     \
    cvk_fatalgroup(mask, "%s: " fmt, __func__, ##__VA_ARGS__)
#define cvk_error_group_fn(mask, fmt, ...)                                     \
    cvk_error_group(mask, "%s: " fmt, __func__, ##__VA_ARGS__)
#define cvk_warn_group_fn(mask, fmt, ...)                                      \
    cvk_warn_group(mask, "%s: " fmt, __func__, ##__VA_ARGS__)
#define cvk_info_group_fn(mask, fmt, ...)                                      \
    cvk_info_group(mask, "%s: " fmt, __func__, ##__VA_ARGS__)
#define cvk_debug_group_fn(mask, fmt, ...)                                     \
    cvk_debug_group(mask, "%s: " fmt, __func__, ##__VA_ARGS__)

const char* vulkan_error_string(VkResult result);
std::string pretty_size(uint64_t size);
std::string vulkan_memory_property_flags_string(VkMemoryPropertyFlags flags);
std::string vulkan_queue_flags_string(VkQueueFlags flags);
std::string vulkan_physical_device_type_string(VkPhysicalDeviceType type);
std::string vulkan_calibrateable_time_domain_string(VkTimeDomainEXT td);
std::string vulkan_format_features_string(VkFormatFeatureFlags flags);
std::string vulkan_vendor_id_string(VkVendorId vid);

static inline std::string vulkan_version_string(uint32_t version) {
    std::string ret = std::to_string(VK_VERSION_MAJOR(version));
    ret += "." + std::to_string(VK_VERSION_MINOR(version));
    ret += "." + std::to_string(VK_VERSION_PATCH(version));
    return ret;
}

std::string cl_channel_order_to_string(cl_channel_order order);
std::string cl_channel_type_to_string(cl_channel_type type);

const char* cl_command_type_to_string(cl_command_type type);
const char* cl_device_type_to_string(cl_device_type type);
const char* cl_command_execution_status_to_string(cl_int status);

const char* spirv_capability_to_string(spv::Capability capability);

const char* queue_global_priority_to_string(VkQueueGlobalPriorityKHR priority);
