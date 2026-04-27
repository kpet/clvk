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

#include "event.hpp"
#include "queue.hpp"

static const cl_profiling_info status_to_profiling_info[4] = {
    CL_PROFILING_COMMAND_END,
    CL_PROFILING_COMMAND_START,
    CL_PROFILING_COMMAND_SUBMIT,
    CL_PROFILING_COMMAND_QUEUED,
};

cvk_event_command::cvk_event_command(cvk_context* ctx, cvk_command_queue* queue,
                                     cl_command_type type)
    : cvk_event(ctx, queue), m_query_pool(VK_NULL_HANDLE) {
    m_command_type = type;
    if (type == CL_COMMAND_USER) {
        m_status = CL_SUBMITTED;
    } else {
        m_status = CL_QUEUED;
    }
}

cvk_event_command::~cvk_event_command() {
    if (m_query_pool != VK_NULL_HANDLE) {
        auto vkdev = m_queue->device()->vulkan_device();
        vkDestroyQueryPool(vkdev, m_query_pool, nullptr);
    }
}

void cvk_event_command::set_status(cl_int status) {
    cvk_debug_group(loggroup::event,
                    "cvk_event::set_status: event = %p, status = %d", this,
                    status);
    std::lock_guard<std::mutex> lock(m_lock);

    CVK_ASSERT(status < m_status);
    m_status = status;

    if (m_queue && m_queue->has_property(CL_QUEUE_PROFILING_ENABLE) &&
        status >= CL_COMPLETE && status <= CL_QUEUED) {
        cl_profiling_info pinfo = status_to_profiling_info[status];
        // profiling could have already been set. In particular in the
        // case of the command_batch
        if (get_profiling_info(pinfo) == 0) {
            if (m_query_pool != VK_NULL_HANDLE &&
                pinfo == CL_PROFILING_COMMAND_END) {
                auto err = set_profiling_info_end_from_query_pool();
                if (err != CL_SUCCESS) {
                    m_status = err;
                }
            } else if (m_query_pool != VK_NULL_HANDLE &&
                       pinfo == CL_PROFILING_COMMAND_START) {
                return;
            } else {
                set_profiling_info_from_monotonic_clock(pinfo);
            }
        }
    }

    if (completed() || terminated()) {

        for (auto& type_cb : m_callbacks) {
            for (auto& cb : type_cb.second) {
                execute_callback(cb);
            }
        }

        m_cv.notify_all();
    }
}

cl_int cvk_event_command::set_profiling_info_end_from_query_pool() {
    // If it has already been set, don't override it
    if (get_profiling_info(CL_PROFILING_COMMAND_END) != 0) {
        return CL_SUCCESS;
    }
    cl_ulong start_dev, end_dev;
    auto perr = get_timestamp_query_results(&start_dev, &end_dev);
    if (perr != CL_COMPLETE) {
        return perr;
    }
    cl_ulong start_host, end_host;
    perr = m_queue->device()->device_timer_to_host(start_dev, start_host);
    if (perr != CL_SUCCESS) {
        return perr;
    }
    perr = m_queue->device()->device_timer_to_host(end_dev, end_host);
    if (perr != CL_SUCCESS) {
        return perr;
    }
    set_profiling_info(CL_PROFILING_COMMAND_START, start_host);
    set_profiling_info(CL_PROFILING_COMMAND_END, end_host);
    return CL_SUCCESS;
}

cl_int cvk_event_command::get_timestamp_query_results(cl_ulong* start,
                                                      cl_ulong* end) {
    uint64_t timestamps[NUM_POOL_QUERIES_PER_COMMAND];
    auto dev = m_queue->device();
    auto res = vkGetQueryPoolResults(
        dev->vulkan_device(), m_query_pool, 0, NUM_POOL_QUERIES_PER_COMMAND,
        sizeof(timestamps), timestamps, sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (res != VK_SUCCESS) {
        cvk_error_fn("vkGetQueryPoolResults failed %d %s", res,
                     vulkan_error_string(res));
        return CL_OUT_OF_RESOURCES;
    }

    auto ts_start_raw = timestamps[POOL_QUERY_CMD_START];
    auto ts_end_raw = timestamps[POOL_QUERY_CMD_END];

    *start = dev->timestamp_to_ns(ts_start_raw);
    *end = dev->timestamp_to_ns(ts_end_raw);

    return CL_COMPLETE;
}
