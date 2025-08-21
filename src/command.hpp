
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

    cl_int finalize() {
        std::lock_guard lock(m_lock);
        if (m_state != CL_COMMAND_BUFFER_STATE_RECORDING_KHR) {
            return CL_INVALID_OPERATION;
        }
        m_state = CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR;
        return CL_SUCCESS;
    }

    cl_int enqueue(const std::vector<cvk_command_queue*>& queues,
                   cl_uint num_events_in_wait_list,
                   const cl_event* event_wait_list, cl_event* event) {
        std::unique_lock<std::mutex> lock(m_lock);
        if (get_updated_state() != CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR) {
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
                err = queue->enqueue_command_with_deps(
                    cmd, num_events_in_wait_list, event_wait_list, &events[i]);
                if (err != CL_SUCCESS) {
                    return err;
                }
            }
            if (nb_commands == 0) {
                nb_commands = 1;
                events.resize(nb_commands);
                auto cmd =
                    new cvk_command_dep(queue, CL_COMMAND_COMMAND_BUFFER_KHR);
                err = queue->enqueue_command_with_deps(
                    cmd, num_events_in_wait_list, event_wait_list, &events[0]);
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
        m_state = CL_COMMAND_BUFFER_STATE_PENDING_KHR;
        return err;
    }

private:
    cl_command_buffer_state_khr get_updated_state() {
        if (m_state == CL_COMMAND_BUFFER_STATE_PENDING_KHR &&
            last_enqueue_event != nullptr && last_enqueue_event->completed()) {
            m_state = CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR;
        }
        return m_state;
    }
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
