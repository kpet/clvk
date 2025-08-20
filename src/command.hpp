// Copyright 2025 The clvk authors.
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

#include "cl_headers.hpp"
#include "memory.hpp"
#include "objects.hpp"
#include "queue.hpp"

struct cvk_api_command_buffer : public _cl_command_buffer_khr,
                                api_object<object_magic::command_buffer> {

    cvk_api_command_buffer(const std::vector<cvk_command_queue*>& queues,
                           std::vector<cl_command_properties_khr>&& properties)
        : api_object(queues[0]->context()), m_sync_point(1),
          m_properties(std::move(properties)),
          m_state(CL_COMMAND_BUFFER_STATE_RECORDING_KHR) {
        for (auto cq : queues) {
            m_queues.emplace_back(cq);
        }
    }

    const std::vector<cvk_command_queue_holder>& queues() const {
        return m_queues;
    }
    const std::vector<cl_command_properties_khr>& properties() const {
        return m_properties;
    }
    cl_command_buffer_state_khr state() { return get_updated_state(); }

    cl_int add_command(cvk_command* command,
                       cl_uint num_sync_points_in_wait_list,
                       const cl_sync_point_khr* sync_point_wait_list,
                       cl_sync_point_khr* sync_point);

    cl_int finalize();

    cl_int enqueue(const std::vector<cvk_command_queue*>& queues,
                   cl_uint num_events_in_wait_list,
                   const cl_event* event_wait_list, cl_event* event);

private:
    cl_command_buffer_state_khr get_updated_state();

    cl_sync_point_khr m_sync_point;
    std::vector<cvk_command_queue_holder> m_queues;
    std::vector<cl_command_properties_khr> m_properties;
    std::unordered_map<cvk_command_queue*, std::vector<cvk_command*>>
        m_commands;
    cl_command_buffer_state_khr m_state;
    std::mutex m_lock;
    cvk_event_holder last_enqueue_event;
};

static inline cvk_api_command_buffer*
icd_downcast(cl_command_buffer_khr cmdbuf) {
    return static_cast<cvk_api_command_buffer*>(cmdbuf);
}
