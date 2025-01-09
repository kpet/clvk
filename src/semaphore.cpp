// Copyright 2022 The clvk authors.
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

#include "semaphore.hpp"
#include "tracing.hpp"

cl_int cvk_semaphore::init() {

    auto vkdev = m_context->device()->vulkan_device();

    VkSemaphoreTypeCreateInfo timelineCreateInfo;
    timelineCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timelineCreateInfo.pNext = NULL;
    timelineCreateInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timelineCreateInfo.initialValue = m_next_value;

    VkSemaphoreCreateInfo info = {
        VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &timelineCreateInfo,
        0 // flags
    };

    auto res = vkCreateSemaphore(vkdev, &info, nullptr, &m_semaphore);
    if (res != VK_SUCCESS) {
        return CL_OUT_OF_RESOURCES;
    }

    return CL_SUCCESS;
}

void cvk_semaphore::notify(uint64_t value) {
    std::unique_lock<std::mutex> lock(m_lock);
    if (value <= m_current_value) {
        return;
    }

    VkSemaphoreSignalInfo signalInfo;
    signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    signalInfo.pNext = NULL;
    signalInfo.semaphore = m_semaphore;
    signalInfo.value = value;
    VkResult res =
        vkSignalSemaphore(m_context->device()->vulkan_device(), &signalInfo);
    if (res != VK_SUCCESS) {
        cvk_error("vkSignalSemaphore failed (%d %s)", res,
                  vulkan_error_string(res));
    }
    m_current_value = value;
}

bool cvk_semaphore::wait(uint64_t value) {
    std::unique_lock<std::mutex> lock(m_lock);
    VkResult res = VK_SUCCESS;
    do {
        if (value <= m_current_value) {
            return true;
        }
        VkSemaphoreWaitInfo waitInfo;
        waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        waitInfo.pNext = NULL;
        waitInfo.flags = 0;
        waitInfo.semaphoreCount = 1;
        waitInfo.pSemaphores = &m_semaphore;
        waitInfo.pValues = &value;

        TRACE_BEGIN("vkWaitSemaphore", "semaphore", (uintptr_t)this,
                    "semaphore-value", value);
        lock.unlock();
        res = vkWaitSemaphores(m_context->device()->vulkan_device(), &waitInfo,
                               1000000);
        lock.lock();
        TRACE_END();

        if (res != VK_TIMEOUT && res != VK_SUCCESS) {
            cvk_error("vkWaitSemaphores failed (%d %s)", res,
                      vulkan_error_string(res));
            return false;
        }
    } while (res != VK_SUCCESS);
    m_current_value = std::max(m_current_value, value);
    return true;
}

bool cvk_semaphore::poll(uint64_t value) {
    uint64_t counter_value;
    TRACE_BEGIN("vkPollSemaphore", "semaphore", (uintptr_t)this,
                "semaphore-value", value);
    do {
        VkResult res = vkGetSemaphoreCounterValue(
            m_context->device()->vulkan_device(), m_semaphore, &counter_value);
        if (res != VK_SUCCESS) {
            cvk_error("vkGetSemaphoreCounterValue failed (%d %s)", res,
                      vulkan_error_string(res));
            return false;
        }
    } while (counter_value < value);
    m_lock.lock();
    m_current_value = std::max(m_current_value, counter_value);
    m_lock.unlock();
    TRACE_END();

    return true;
}

bool cvk_semaphore::poll_once(uint64_t value) {
    uint64_t counter_value;
    TRACE_BEGIN("vkPollOnceSemaphore", "semaphore", (uintptr_t)this,
                "semaphore-value", value);
    VkResult res = vkGetSemaphoreCounterValue(
        m_context->device()->vulkan_device(), m_semaphore, &counter_value);
    if (res != VK_SUCCESS) {
        cvk_error("vkGetSemaphoreCounterValue failed (%d %s)", res,
                  vulkan_error_string(res));
        return false;
    }
    m_lock.lock();
    m_current_value = std::max(m_current_value, counter_value);
    m_lock.unlock();
    TRACE_END();

    return m_current_value >= value;
}
