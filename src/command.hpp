
#pragma once

#include "cl_headers.hpp"
#include "objects.hpp"
#include "queue.hpp"

struct cvk_api_command_buffer : public _cl_command_buffer_khr,
                                api_object<object_magic::command_buffer> {

    cvk_api_command_buffer(
        const std::vector<cvk_command_queue*>& queues,
        std::vector<cl_command_buffer_properties_khr>&& properties)
        : api_object(queues[0]->context()), m_sync_point(0),
          m_properties(std::move(properties)),
          m_state(CL_COMMAND_BUFFER_STATE_RECORDING_KHR) {
        for (auto cq : queues) {
            m_queues.emplace_back(cq);
        }
    }

    const std::vector<cvk_queue_holder>& queues() const { return m_queues; }
    const std::vector<cl_command_buffer_properties_khr>& properties() const {
        return m_properties;
    }
    cl_command_buffer_state_khr state() const { return m_state; }

    cl_int add_command(cvk_command* command,
                       cl_uint num_sync_points_in_wait_list,
                       const cl_sync_point_khr* sync_point_wait_list,
                       cl_sync_point_khr* sync_point) {
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

    cl_int finalize() {
        std::lock_guard lock(m_lock);
        if (m_state != CL_COMMAND_BUFFER_STATE_RECORDING_KHR) {
            return CL_INVALID_OPERATION;
        }
        for (auto& queue_cmds : m_commands) {
            auto& cmds = queue_cmds.second;
            for (auto cmd : cmds) {
                if (!cmd->is_built_before_enqueue()) {
                    cvk_debug_fn("building command %p", cmd);
                    cl_int err =
                        static_cast<cvk_command_batchable*>(cmd)->build();
                    if (err != CL_SUCCESS) {
                        return err;
                    }
                }
            }
        }
        m_state = CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR;
        return CL_SUCCESS;
    }

    cl_int enqueue(const std::vector<cvk_command_queue*>& queues,
                   cl_uint num_events_in_wait_list,
                   const cl_event* event_wait_list, cl_event* event) {
        std::lock_guard lock(m_lock);
        if (m_state != CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR) {
            return CL_INVALID_OPERATION;
        }
        for (auto queue : m_queues) {
            // TODO are empty command buffers allowed?
            if (m_commands.count(queue) == 0) {
                auto cmd =
                    new cvk_command_dep(queue, CL_COMMAND_COMMAND_BUFFER_KHR);

                queue->enqueue_command_with_deps(cmd, num_events_in_wait_list,
                                                 event_wait_list, event);
            } else {
                std::vector<std::unique_ptr<cvk_command>> commands;

                for (auto cmd : m_commands[queue]) {
                    commands.emplace_back(cmd);
                }
                auto cmd = new cvk_command_combine(
                    queue, CL_COMMAND_COMMAND_BUFFER_KHR, std::move(commands));

                queue->enqueue_command_with_deps(cmd, CL_NON_BLOCKING,
                                                 num_events_in_wait_list,
                                                 event_wait_list, event);
            }
        }
        m_state = CL_COMMAND_BUFFER_STATE_PENDING_KHR;
        return CL_SUCCESS;
    }

private:
    cl_sync_point_khr m_sync_point;
    std::vector<cvk_queue_holder> m_queues;
    std::vector<cl_command_buffer_properties_khr> m_properties;
    std::unordered_map<cvk_command_queue*, std::vector<cvk_command*>>
        m_commands;
    cl_command_buffer_state_khr
        m_state; // FIXME reset state back to
                 // CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR after execution
    std::mutex m_lock;
};

static inline cvk_api_command_buffer*
icd_downcast(cl_command_buffer_khr cmdbuf) {
    return static_cast<cvk_api_command_buffer*>(cmdbuf);
}
