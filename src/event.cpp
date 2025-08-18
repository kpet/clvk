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

cvk_event_command::cvk_event_command(cvk_context* ctx, cvk_command* cmd,
                                     cvk_command_queue* queue)
    : cvk_event(ctx, queue), m_cmd(cmd) {
    if (cmd == nullptr) {
        m_status = CL_SUBMITTED;
        m_command_type = CL_COMMAND_USER;
    } else {
        m_status = CL_QUEUED;
        m_command_type = cmd->type();
    }
}

void cvk_event_command::set_status(cl_int status) {
    cvk_debug_group(loggroup::event,
                    "cvk_event::set_status: event = %p, status = %d", this,
                    status);
    std::lock_guard<std::mutex> lock(m_lock);

    CVK_ASSERT(status < m_status);
    m_status = status;

    if (m_queue && m_queue->has_property(CL_QUEUE_PROFILING_ENABLE) && m_cmd &&
        status >= CL_COMPLETE && status <= CL_QUEUED) {
        cl_profiling_info pinfo = status_to_profiling_info[status];
        // profiling could have already been set. In particular in the
        // case of the command_batch
        if (get_profiling_info(pinfo) == 0) {
            auto err = m_cmd->set_profiling_info(pinfo);
            if (err != CL_SUCCESS) {
                m_status = err;
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
