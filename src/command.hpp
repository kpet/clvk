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

    cvk_api_command_buffer(
        const std::vector<cvk_command_queue*>& queues,
        cl_command_buffer_flags_khr flags,
        std::vector<cl_command_buffer_properties_khr>&& properties)
        : api_object(queues[0]->context()), m_sync_point(1), m_flags(flags),
          m_properties(std::move(properties)),
          m_state(CL_COMMAND_BUFFER_STATE_RECORDING_KHR),
          m_can_pre_record(false) {
        for (auto cq : queues) {
            m_queues.emplace_back(cq);
        }
    }

    ~cvk_api_command_buffer() {
        m_recorded_cmdbufs.clear();
        for (auto& [queue, commands] : m_commands) {
            for (auto cmd : commands) {
                cmd->release();
            }
        }
    }

    const std::vector<cvk_command_queue_holder>& queues() const {
        return m_queues;
    }
    const std::vector<cl_command_buffer_properties_khr>& properties() const {
        return m_properties;
    }
    bool is_simultaneous_use() const {
        return (m_flags & CL_COMMAND_BUFFER_SIMULTANEOUS_USE_KHR) != 0;
    }
    cl_command_buffer_state_khr state();

    cl_int add_command(cvk_command* command,
                       cl_uint num_sync_points_in_wait_list,
                       const cl_sync_point_khr* sync_point_wait_list,
                       cl_sync_point_khr* sync_point);

    cl_int finalize();

    cl_int enqueue(const std::vector<cvk_command_queue*>& queues,
                   cl_uint num_events_in_wait_list,
                   const cl_event* event_wait_list, cl_event* event);

private:
    bool can_pre_record_commands(const std::vector<cvk_command*>& cmds) const;
    cl_int build_recorded_cmdbuf_locked(cvk_command_queue* queue);
    cl_int resolve_target_queues_locked(
        const std::vector<cvk_command_queue*>& queues,
        std::vector<cvk_command_queue*>* target_queues) const;
    void ensure_commands_for_queue_locked(cvk_command_queue* queue);
    cl_int enqueue_single_command_locked(cvk_command_queue* queue,
                                         cvk_command* cmd,
                                         cl_uint num_events_in_wait_list,
                                         const cl_event* event_wait_list,
                                         cl_event* event);
    cl_int enqueue_empty_locked(cvk_command_queue* queue,
                                cl_uint num_events_in_wait_list,
                                const cl_event* event_wait_list,
                                cl_event* event);
    cl_int enqueue_pre_recorded_locked(cvk_command_queue* queue,
                                       cl_uint num_events_in_wait_list,
                                       const cl_event* event_wait_list,
                                       cl_event* event);
    cl_int enqueue_cloned_commands_locked(cvk_command_queue* queue,
                                          cl_uint num_events_in_wait_list,
                                          const cl_event* event_wait_list,
                                          cl_event* event);
    cl_int enqueue_on_queue_locked(cvk_command_queue* queue,
                                   cl_uint num_events_in_wait_list,
                                   const cl_event* event_wait_list,
                                   cl_event* event);

    cl_sync_point_khr m_sync_point;
    cl_command_buffer_flags_khr m_flags;
    std::vector<cvk_command_queue_holder> m_queues;
    std::vector<cl_command_buffer_properties_khr> m_properties;
    std::unordered_map<cvk_command_queue*, std::vector<cvk_command*>>
        m_commands;
    std::unordered_map<cvk_command_queue*, std::unique_ptr<cvk_command_buffer>>
        m_recorded_cmdbufs;
    std::vector<cvk_mem*> m_memory_objects;
    cl_command_buffer_state_khr m_state;
    bool m_can_pre_record;
    std::mutex m_lock;
};

using cvk_api_command_buffer_holder = refcounted_holder<cvk_api_command_buffer>;

struct cvk_command_exec_cmdbuf final : public cvk_command_batchable {
    cvk_command_exec_cmdbuf(cvk_command_queue* queue,
                            cvk_api_command_buffer* api_cmdbuf,
                            cvk_command_buffer* pre_recorded,
                            const std::vector<cvk_command*>* recorded_cmds,
                            const std::vector<cvk_mem*>* mem_objects)
        : cvk_command_batchable(CL_COMMAND_COMMAND_BUFFER_KHR, queue),
          m_api_cmdbuf(api_cmdbuf), m_pre_recorded(pre_recorded),
          m_recorded_cmds(recorded_cmds), m_mem_objects(mem_objects) {}

    ~cvk_command_exec_cmdbuf() override {
        // Free the primary command buffer (if built non-batched) before
        // releasing m_api_cmdbuf, which owns the referenced secondary buffer.
        reset_command_buffer();
    }

    CHECK_RETURN cl_int
    build_batchable_inner(cvk_command_buffer& cmdbuf) override final {
        VkCommandBuffer secondary = *m_pre_recorded;
        vkCmdExecuteCommands(cmdbuf, 1, &secondary);
        return CL_SUCCESS;
    }

    CHECK_RETURN cl_int do_post_action() override final {
        for (auto cmd : *m_recorded_cmds) {
            if (cmd->type() == CL_COMMAND_BARRIER) {
                continue;
            }
            cl_int err =
                static_cast<cvk_command_batchable*>(cmd)->do_post_action();
            if (err != CL_SUCCESS) {
                return err;
            }
        }
        return CL_SUCCESS;
    }

    const std::vector<cvk_mem*> memory_objects() const override final {
        return *m_mem_objects;
    }

    cvk_command* clone(cvk_command_queue* queue) override final {
        CVK_ASSERT(queue == m_queue);
        return new cvk_command_exec_cmdbuf(queue, m_api_cmdbuf, m_pre_recorded,
                                           m_recorded_cmds, m_mem_objects);
    }

private:
    cvk_api_command_buffer_holder m_api_cmdbuf;
    cvk_command_buffer* m_pre_recorded;
    const std::vector<cvk_command*>* m_recorded_cmds;
    const std::vector<cvk_mem*>* m_mem_objects;
};

static inline cvk_api_command_buffer*
icd_downcast(cl_command_buffer_khr cmdbuf) {
    return static_cast<cvk_api_command_buffer*>(cmdbuf);
}
