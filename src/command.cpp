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

#include "command.hpp"

cl_int cvk_api_command_buffer::add_command(
    cvk_command* command, cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point) {
    UNUSED(num_sync_points_in_wait_list);
    UNUSED(sync_point_wait_list);
    std::lock_guard lock(m_lock);
    if (m_state != CL_COMMAND_BUFFER_STATE_RECORDING_KHR) {
        return CL_INVALID_OPERATION;
    }
    m_commands[command->queue()].push_back(command);
    if (sync_point != nullptr) {
        *sync_point = m_sync_point;
    }
    m_sync_point++;
    return CL_SUCCESS;
}

cl_int cvk_api_command_buffer::finalize() {
    std::lock_guard lock(m_lock);
    if (m_state != CL_COMMAND_BUFFER_STATE_RECORDING_KHR) {
        return CL_INVALID_OPERATION;
    }
    m_state = CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR;
    return CL_SUCCESS;
}

cl_int
cvk_api_command_buffer::enqueue(const std::vector<cvk_command_queue*>& queues,
                                cl_uint num_events_in_wait_list,
                                const cl_event* event_wait_list,
                                cl_event* event) {
    std::unique_lock<std::mutex> lock(m_lock);
    if (m_state != CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR) {
        return CL_INVALID_OPERATION;
    }
    std::vector<cvk_command_queue*> queues_to_enqueue;
    if (queues.size() == 0) {
        for (auto queue : m_queues) {
            queues_to_enqueue.push_back(queue);
        }
    } else {
        for (auto queue : queues) {
            queues_to_enqueue.push_back(queue);
            if (m_commands.count(queue) != m_commands.count(m_queues[0])) {
                for (auto cmd : m_commands[m_queues[0]]) {
                    m_commands[queue].push_back(cmd->clone(queue));
                }
            }
        }
    }

    cl_int err = CL_SUCCESS;
    for (auto queue : queues_to_enqueue) {
        for (auto cmd : m_commands[queue]) {
            cmd->reset_event();
            cmd->retain();
        }
        unsigned nb_commands = m_commands[queue].size();
        std::vector<cl_event> events;
        events.resize(nb_commands);
        for (unsigned i = 0; i < nb_commands; i++) {
            auto cmd = m_commands[queue][i];
            err = queue->enqueue_command_with_deps(cmd, num_events_in_wait_list,
                                                   event_wait_list, &events[i]);
            if (err != CL_SUCCESS) {
                return err;
            }
        }
        if (nb_commands == 0) {
            nb_commands = 1;
            events.resize(nb_commands);
            auto cmd =
                new cvk_command_dep(queue, CL_COMMAND_COMMAND_BUFFER_KHR);
            err = queue->enqueue_command_with_deps(cmd, num_events_in_wait_list,
                                                   event_wait_list, &events[0]);
            if (err != CL_SUCCESS) {
                return err;
            }
        }

        if (event != nullptr) {
            if (nb_commands == 1) {
                *event = events[0];
                icd_downcast(events[0])->retain();
            } else {
                *event = new cvk_event_combine(
                    queue->context(), CL_COMMAND_COMMAND_BUFFER_KHR, queue,
                    icd_downcast(events[0]),
                    icd_downcast(events[nb_commands - 1]));
            }
        }
        last_enqueue_event.reset(icd_downcast(events[nb_commands - 1]));

        for (unsigned i = 0; i < nb_commands; i++) {
            icd_downcast(events[i])->release();
        }
    }
    return err;
}
