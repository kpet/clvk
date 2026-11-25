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

#include <unordered_set>

#include "command.hpp"

cl_command_buffer_state_khr cvk_api_command_buffer::state() {
    std::lock_guard lock(m_lock);
    return m_state;
}

bool cvk_api_command_buffer::can_pre_record_commands(
    const std::vector<cvk_command*>& cmds) const {
    if (cmds.empty()) {
        return false;
    }
    for (auto cmd : cmds) {
        if (cmd->type() == CL_COMMAND_BARRIER) {
            continue;
        }
        if (!cmd->can_be_batched()) {
            return false;
        }
    }
    return true;
}

cl_int
cvk_api_command_buffer::build_recorded_cmdbuf_locked(cvk_command_queue* queue) {
    auto vk_cmdbuf = std::make_unique<cvk_command_buffer>(
        queue, VK_COMMAND_BUFFER_LEVEL_SECONDARY);
    // OpenCL cl_khr_command_buffer 1.0 allows pipelined/serialized re-enqueues
    // even without CL_COMMAND_BUFFER_SIMULTANEOUS_USE_KHR. That can record the
    // same secondary buffer multiple times into one primary batch or into a new
    // primary batch while an earlier submission is still pending, which Vulkan
    // requires VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT for
    // (VUID-vkCmdExecuteCommands-pCommandBuffers-00091/00092).
    if (!vk_cmdbuf->begin(VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT)) {
        return CL_OUT_OF_RESOURCES;
    }

    {
        cvk_command_pool_lock_holder pool_lock(queue);
        for (auto cmd : m_commands[queue]) {
            if (cmd->type() == CL_COMMAND_BARRIER) {
                VkMemoryBarrier memoryBarrier = {
                    VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                    VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
                vkCmdPipelineBarrier(*vk_cmdbuf,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     0, 1, &memoryBarrier, 0, nullptr, 0,
                                     nullptr);
                continue;
            }
            cl_int err =
                static_cast<cvk_command_batchable*>(cmd)->build_batchable_inner(
                    *vk_cmdbuf);
            if (err != CL_SUCCESS) {
                return err;
            }
        }
        if (!vk_cmdbuf->end()) {
            return CL_OUT_OF_RESOURCES;
        }
    }

    m_recorded_cmdbufs[queue] = std::move(vk_cmdbuf);
    return CL_SUCCESS;
}

cl_int cvk_api_command_buffer::add_command(
    cvk_command* command, cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point) {
    UNUSED(num_sync_points_in_wait_list);
    UNUSED(sync_point_wait_list);
    std::lock_guard lock(m_lock);
    if (m_state != CL_COMMAND_BUFFER_STATE_RECORDING_KHR) {
        command->release();
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

    auto primary_queue = m_queues[0];
    auto it = m_commands.find(primary_queue);
    if (it != m_commands.end() && !it->second.empty()) {
        std::unordered_set<cvk_mem*> unique_mems;
        for (auto cmd : it->second) {
            for (auto mem : cmd->memory_objects()) {
                if (unique_mems.insert(mem).second) {
                    m_memory_objects.push_back(mem);
                }
            }
        }
        m_can_pre_record = can_pre_record_commands(it->second);
        if (m_can_pre_record) {
            cl_int err = build_recorded_cmdbuf_locked(primary_queue);
            if (err != CL_SUCCESS) {
                m_can_pre_record = false;
                m_memory_objects.clear();
                return err;
            }
        }
    }

    m_state = CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR;
    return CL_SUCCESS;
}

cl_int cvk_api_command_buffer::resolve_target_queues_locked(
    const std::vector<cvk_command_queue*>& queues,
    std::vector<cvk_command_queue*>* target_queues) const {
    if (queues.empty()) {
        for (auto queue : m_queues) {
            target_queues->push_back(queue);
        }
    } else {
        *target_queues = queues;
    }

    for (auto queue : *target_queues) {
        if (queue->has_property(CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)) {
            return CL_INCOMPATIBLE_COMMAND_QUEUE_KHR;
        }
    }
    return CL_SUCCESS;
}

void cvk_api_command_buffer::ensure_commands_for_queue_locked(
    cvk_command_queue* queue) {
    if (m_commands.count(queue) != 0 || m_commands.count(m_queues[0]) == 0) {
        return;
    }
    for (auto cmd : m_commands[m_queues[0]]) {
        m_commands[queue].push_back(cmd->clone(queue));
    }
}

cl_int cvk_api_command_buffer::enqueue_single_command_locked(
    cvk_command_queue* queue, cvk_command* cmd, cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list, cl_event* event) {
    return queue->enqueue_command_with_deps(cmd, num_events_in_wait_list,
                                            event_wait_list, event);
}

cl_int cvk_api_command_buffer::enqueue_empty_locked(
    cvk_command_queue* queue, cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list, cl_event* event) {
    auto cmd = new cvk_command_dep(queue, CL_COMMAND_COMMAND_BUFFER_KHR);
    return enqueue_single_command_locked(queue, cmd, num_events_in_wait_list,
                                         event_wait_list, event);
}

cl_int cvk_api_command_buffer::enqueue_pre_recorded_locked(
    cvk_command_queue* queue, cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list, cl_event* event) {
    if (m_recorded_cmdbufs.count(queue) == 0) {
        cl_int err = build_recorded_cmdbuf_locked(queue);
        if (err != CL_SUCCESS) {
            return err;
        }
    }

    auto exec_cmd = new cvk_command_exec_cmdbuf(
        queue, this, m_recorded_cmdbufs[queue].get(), &m_commands[queue],
        &m_memory_objects);
    return enqueue_single_command_locked(
        queue, exec_cmd, num_events_in_wait_list, event_wait_list, event);
}

cl_int cvk_api_command_buffer::enqueue_cloned_commands_locked(
    cvk_command_queue* queue, cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list, cl_event* event) {
    const auto& cmds = m_commands[queue];
    const unsigned nb_commands = cmds.size();
    cl_event first_event = nullptr;
    cl_event last_event = nullptr;

    for (unsigned i = 0; i < nb_commands; i++) {
        auto cloned_cmd = cmds[i]->clone(queue);
        cl_uint deps_count = (i == 0) ? num_events_in_wait_list : 0;
        const cl_event* deps = (i == 0) ? event_wait_list : nullptr;
        cl_event* out_event = nullptr;
        if (event != nullptr) {
            if (i == 0) {
                out_event = &first_event;
            } else if (i == nb_commands - 1) {
                out_event = &last_event;
            }
        }
        cl_int err = queue->enqueue_command_with_deps(cloned_cmd, deps_count,
                                                      deps, out_event);
        if (err != CL_SUCCESS) {
            if (first_event != nullptr) {
                icd_downcast(first_event)->release();
            }
            if (last_event != nullptr) {
                icd_downcast(last_event)->release();
            }
            return err;
        }
    }

    if (event != nullptr) {
        cl_event end_event = (nb_commands == 1) ? first_event : last_event;
        *event = new cvk_event_combine(
            queue->context(), CL_COMMAND_COMMAND_BUFFER_KHR, queue,
            icd_downcast(first_event), icd_downcast(end_event));
        icd_downcast(first_event)->release();
        if (last_event != nullptr) {
            icd_downcast(last_event)->release();
        }
    }
    return CL_SUCCESS;
}

cl_int cvk_api_command_buffer::enqueue_on_queue_locked(
    cvk_command_queue* queue, cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list, cl_event* event) {
    ensure_commands_for_queue_locked(queue);

    if (m_commands[queue].empty()) {
        return enqueue_empty_locked(queue, num_events_in_wait_list,
                                    event_wait_list, event);
    }
    if (m_can_pre_record) {
        return enqueue_pre_recorded_locked(queue, num_events_in_wait_list,
                                           event_wait_list, event);
    }
    return enqueue_cloned_commands_locked(queue, num_events_in_wait_list,
                                          event_wait_list, event);
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

    std::vector<cvk_command_queue*> target_queues;
    cl_int err = resolve_target_queues_locked(queues, &target_queues);
    if (err != CL_SUCCESS) {
        return err;
    }

    for (auto queue : target_queues) {
        err = enqueue_on_queue_locked(queue, num_events_in_wait_list,
                                      event_wait_list, event);
        if (err != CL_SUCCESS) {
            return err;
        }
    }
    return CL_SUCCESS;
}
