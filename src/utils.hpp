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

#pragma once

#include <cassert>
#include <string>

#include <vulkan/vulkan.h>

#ifndef _MSC_VER
#define CHECK_RETURN __attribute__((warn_unused_result))
#define CHECK_PRINTF(index, first) __attribute__((format(printf, index, first)))
#else
#define CHECK_RETURN
#define CHECK_PRINTF(index, first)
#endif

enum loglevel
{
    fatal = 0,
    error = 1,
    warn = 2,
    info = 3,
    debug = 4
};

char* cvk_mkdtemp(std::string& tmpl);

void cvk_log(loglevel level, const char* fmt, ...) CHECK_PRINTF(2, 3);

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
std::string vulkan_memory_property_flags_string(VkMemoryPropertyFlags flags);
std::string vulkan_queue_flags_string(VkQueueFlags flags);
std::string vulkan_physical_device_type_string(VkPhysicalDeviceType type);

static inline std::string vulkan_version_string(uint32_t version) {
    std::string ret = std::to_string(VK_VERSION_MAJOR(version));
    ret += "." + std::to_string(VK_VERSION_MINOR(version));
    ret += "." + std::to_string(VK_VERSION_PATCH(version));
    return ret;
}

#define CVK_VK_CHECK_INTERNAL(logfn, res, msg)                                 \
    do {                                                                       \
        if (res != VK_SUCCESS) {                                               \
            logfn(msg " : %s", vulkan_error_string(res));                      \
        }                                                                      \
    } while (0);

#define CVK_VK_CHECK_INTERNAL_RET(logfn, res, ret, msg)                        \
    do {                                                                       \
        if (res != VK_SUCCESS) {                                               \
            logfn(msg " : %s", vulkan_error_string(res));                      \
            return ret;                                                        \
        }                                                                      \
    } while (0);

#define CVK_VK_CHECK_FATAL(res, msg) CVK_VK_CHECK_INTERNAL(cvk_fatal, res, msg)
#define CVK_VK_CHECK_ERROR(res, msg) CVK_VK_CHECK_INTERNAL(cvk_error, res, msg)
#define CVK_VK_CHECK_ERROR_RET(res, ret, msg)                                  \
    CVK_VK_CHECK_INTERNAL_RET(cvk_error, res, ret, msg)

#define CVK_ASSERT(cond) assert(cond)

#define CVK_VK_GET_INSTANCE_PROC(name)                                         \
    PFN_##name fn##name = reinterpret_cast<PFN_##name>(                        \
        vkGetInstanceProcAddr(gVkInstance, #name))

#define UNUSED(X) ((void)(X))

std::string pretty_size(uint64_t size);

static inline void* pointer_offset(const void* ptr, size_t offset) {
    auto ptrint = reinterpret_cast<uintptr_t>(ptr);
    return reinterpret_cast<void*>(ptrint + offset);
}
