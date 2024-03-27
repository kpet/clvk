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

#pragma once

#include "cl_headers.hpp"
#include "context.hpp"
#include "icd.hpp"
#include "objects.hpp"
#include "tracing.hpp"
#include "utils.hpp"

#include <mutex>
#include <unordered_map>

struct cvk_command;
struct cvk_command_queue;

using cvk_event_callback_pointer_type = void(CL_CALLBACK*)(
    cl_event event, cl_int event_command_exec_status, void* user_data);

struct cvk_event_callback {
    cvk_event_callback_pointer_type pointer;
    void* data;
};

struct cvk_event : public _cl_event, api_object<object_magic::event> {

    cvk_event(cvk_context* ctx, cvk_command_queue* queue)
        : api_object(ctx), m_command_type(0), m_queue(queue) {}

    virtual cl_int get_status() const = 0;

    bool completed() { return get_status() == CL_COMPLETE; }

    bool terminated() { return get_status() < 0; }

    virtual void set_status(cl_int status) = 0;

    virtual void register_callback(cl_int callback_type,
                                   cvk_event_callback_pointer_type ptr,
                                   void* user_data) = 0;

    cl_command_type command_type() const { return m_command_type; }

    bool is_user_event() const { return m_command_type == CL_COMMAND_USER; }

    cvk_command_queue* queue() const {
        CVK_ASSERT(!is_user_event());
        return m_queue;
    }

    virtual cl_int wait() = 0;

    virtual uint64_t get_profiling_info(cl_profiling_info pinfo) const = 0;

protected:
    cl_command_type m_command_type;
    cvk_command_queue* m_queue;
};

struct cvk_event_command : public cvk_event {

    cvk_event_command(cvk_context* ctx, cvk_command* cmd,
                      cvk_command_queue* queue);

    void set_status(cl_int status) override final;

    void register_callback(cl_int callback_type,
                           cvk_event_callback_pointer_type ptr,
                           void* user_data) override final {
        std::lock_guard<std::mutex> lock(m_lock);

        cvk_event_callback cb = {ptr, user_data};

        if (m_status <= callback_type) {
            execute_callback(cb);
        } else {
            m_callbacks[callback_type].push_back(cb);
        }
    }

    cl_int get_status() const override final { return m_status; }

    cl_int wait() override final {
        std::unique_lock<std::mutex> lock(m_lock);
        cvk_debug_group(loggroup::event,
                        "cvk_event::wait: event = %p, status = %d", this,
                        m_status);
        if ((m_status != CL_COMPLETE) && (m_status >= 0)) {
            TRACE_BEGIN_EVENT(command_type(), "queue", (uintptr_t)m_queue,
                              "command", (uintptr_t)m_cmd);
            m_cv.wait(lock);
            TRACE_END();
        }

        return m_status;
    }

    void set_profiling_info(cl_profiling_info pinfo, uint64_t val) {
        m_profiling_data[pinfo - CL_PROFILING_COMMAND_QUEUED] = val;
    }

    void copy_profiling_info(cl_profiling_info info, const cvk_event* event) {
        auto val = event->get_profiling_info(info);
        set_profiling_info(info, val);
    }

    uint64_t get_profiling_info(cl_profiling_info pinfo) const override final {
        return m_profiling_data[pinfo - CL_PROFILING_COMMAND_QUEUED];
    }

    static uint64_t sample_clock() {
        auto time_point = std::chrono::steady_clock::now();
        auto time_since_epoch = time_point.time_since_epoch();
        using ns = std::chrono::nanoseconds;
        return std::chrono::duration_cast<ns>(time_since_epoch).count();
    }

    void set_profiling_info_from_monotonic_clock(cl_profiling_info pinfo) {
        set_profiling_info(pinfo, sample_clock());
    }

private:
    void execute_callback(cvk_event_callback cb) {
        cb.pointer(this, m_status, cb.data);
    }

    std::mutex m_lock;
    std::condition_variable m_cv;
    cl_int m_status;
    cl_ulong m_profiling_data[4]{};
    cvk_command* m_cmd;
    std::unordered_map<cl_int, std::vector<cvk_event_callback>> m_callbacks;
};

struct cvk_event_combine : public cvk_event {

    cvk_event_combine(cvk_context* ctx, cl_command_type command_type,
                      cvk_command_queue* queue, cvk_event* start_event,
                      cvk_event* end_event)
        : cvk_event(ctx, queue), m_start_event(start_event),
          m_end_event(end_event) {
        m_command_type = command_type;
        start_event->retain();
        end_event->retain();
    }

    ~cvk_event_combine() {
        m_start_event->release();
        m_end_event->release();
    }

    void set_status(cl_int status) override final {
        UNUSED(status);
        CVK_ASSERT(false && "Should never be called");
    }

    void register_callback(cl_int callback_type,
                           cvk_event_callback_pointer_type ptr,
                           void* user_data) override final {
        if (callback_type == CL_COMPLETE) {
            m_end_event->register_callback(callback_type, ptr, user_data);
        } else {
            m_start_event->register_callback(callback_type, ptr, user_data);
        }
    }

    cl_int get_status() const override final {
        return std::min(m_end_event->get_status(), m_start_event->get_status());
    }

    cl_int wait() override final { return m_end_event->wait(); }

    uint64_t get_profiling_info(cl_profiling_info pinfo) const override final {
        if (pinfo == CL_PROFILING_COMMAND_END) {
            return m_end_event->get_profiling_info(pinfo);
        } else {
            return m_start_event->get_profiling_info(pinfo);
        }
    }

private:
    cvk_event* m_start_event;
    cvk_event* m_end_event;
};

using cvk_event_holder = refcounted_holder<cvk_event>;

static inline cvk_event* icd_downcast(cl_event event) {
    return static_cast<cvk_event*>(event);
}
