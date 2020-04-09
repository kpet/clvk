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

#include "log.hpp"

#include <cassert>

#include <vulkan/vulkan.h>

#ifndef _MSC_VER
#define CHECK_RETURN __attribute__((warn_unused_result))
#else
#define CHECK_RETURN
#endif

char* cvk_mkdtemp(std::string& tmpl);

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

static inline void* pointer_offset(const void* ptr, size_t offset) {
    auto ptrint = reinterpret_cast<uintptr_t>(ptr);
    return reinterpret_cast<void*>(ptrint + offset);
}
