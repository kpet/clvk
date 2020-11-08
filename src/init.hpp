// Copyright 2019 The clvk authors.
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

#include <memory>
#include <string>

#include <vulkan/vulkan.h>

extern std::string gCLSPVPath;
extern std::string gLLVMSPIRVPath;
extern std::string gCLSPVOptions;
extern bool gQueueProfilingUsesTimestampQueries;
extern bool gKeepTemporaries;

struct cvk_platform;
struct cvk_executor_thread_pool;

class clvk_global_state {
public:
    clvk_global_state();
    ~clvk_global_state();

    cvk_platform* platform() const { return m_platform; }

    VkInstance vulkan_instance() const { return m_vulkan_instance; }

    PFN_vkVoidFunction get_instance_proc(const char* name) {
        return vkGetInstanceProcAddr(m_vulkan_instance, name);
    }

    cvk_executor_thread_pool* thread_pool() { return m_thread_pool; }

private:
    void init_vulkan();
    void term_vulkan();
    void init_platform();
    void term_platform();
    void init_executors();
    void term_executors();

    cvk_executor_thread_pool* m_thread_pool;
    cvk_platform* m_platform;
    VkInstance m_vulkan_instance;
    bool m_debug_report_enabled{};
    VkDebugReportCallbackEXT m_vulkan_debug_callback;
};

#define GET_INSTANCE_PROC(instance, name)                                      \
    reinterpret_cast<PFN_##name>(vkGetInstanceProcAddr(instance, #name))

#define CVK_VK_GET_INSTANCE_PROC(state, name)                                  \
    reinterpret_cast<PFN_##name>(state->get_instance_proc(#name))

extern clvk_global_state* get_or_init_global_state();
