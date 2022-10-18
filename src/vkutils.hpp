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

#include <mutex>
#include <vector>

#include <vulkan/vulkan.h>

#include "tracing.hpp"
#include "utils.hpp"

struct cvk_vulkan_queue_wrapper {
    cvk_vulkan_queue_wrapper(VkQueue queue, uint32_t family)
        : m_queue(queue), m_queue_family(family) {}

    cvk_vulkan_queue_wrapper(cvk_vulkan_queue_wrapper&& other) {
        m_queue = other.m_queue;
        m_queue_family = other.m_queue_family;
    }

    ~cvk_vulkan_queue_wrapper() {
        cvk_debug("Queue %p has made %llu submissions.", m_queue,
                  (unsigned long long)m_num_submissions);
    }

    CHECK_RETURN VkResult submit(VkCommandBuffer command_buffer) {
        std::lock_guard<std::mutex> lock(m_lock);

        VkSubmitInfo submitInfo = {
            VK_STRUCTURE_TYPE_SUBMIT_INFO,
            nullptr,
            0,       // waitSemaphoreCOunt
            nullptr, // pWaitSemaphores
            nullptr, // pWaitDstStageMask
            1,       // commandBufferCount
            &command_buffer,
            0,       // signalSemaphoreCount
            nullptr, // pSignalSemaphores
        };

        TRACE_BEGIN("vkQueueSubmit");
        auto ret = vkQueueSubmit(m_queue, 1, &submitInfo, VK_NULL_HANDLE);
        TRACE_END();
        if (ret != VK_SUCCESS) {
            cvk_error_fn("could not submit work to queue: %s",
                         vulkan_error_string(ret));
        }

        m_num_submissions++;

        return ret;
    }

    CHECK_RETURN VkResult submit(const std::vector<VkCommandBuffer>& cmdbufs) {
        std::lock_guard<std::mutex> lock(m_lock);

        VkSubmitInfo submitInfo = {
            VK_STRUCTURE_TYPE_SUBMIT_INFO,
            nullptr,
            0,                                     // waitSemaphoreCOunt
            nullptr,                               // pWaitSemaphores
            nullptr,                               // pWaitDstStageMask
            static_cast<uint32_t>(cmdbufs.size()), // commandBufferCount
            cmdbufs.data(),
            0,       // signalSemaphoreCount
            nullptr, // pSignalSemaphores
        };

        TRACE_BEGIN("vkQueueSubmit");
        auto ret = vkQueueSubmit(m_queue, 1, &submitInfo, VK_NULL_HANDLE);
        TRACE_END();
        if (ret != VK_SUCCESS) {
            cvk_error_fn("could not submit work to queue: %s",
                         vulkan_error_string(ret));
        }

        return ret;
    }

    CHECK_RETURN VkResult wait_idle() {
        std::lock_guard<std::mutex> lock(m_lock);

        TRACE_BEGIN("vkQueueWaitIdle");
        auto ret = vkQueueWaitIdle(m_queue);
        TRACE_END();

        if (ret != VK_SUCCESS) {
            cvk_error_fn("could not wait for queue to become idle: %s",
                         vulkan_error_string(ret));
        }

        return ret;
    }

    uint32_t queue_family() { return m_queue_family; }

private:
    std::mutex m_lock;
    VkQueue m_queue;
    uint32_t m_queue_family;
    uint64_t m_num_submissions{};
};
