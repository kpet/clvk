// Copyright 2018 The clvk authors.
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

#include <memory>

#include "objects.hpp"
#include "kernel.hpp"

struct cvk_command;
typedef struct _cl_command_queue cvk_command_queue;
using cvk_command_queue_holder = refcounted_holder<cvk_command_queue>;

using cvk_event_callback_pointer_type = void (*) (cl_event event, cl_int event_command_exec_status, void *user_data);

struct cvk_event_callback {
    cvk_event_callback_pointer_type pointer;
    void *data;
};

typedef struct _cl_event : public api_object {

    _cl_event(cvk_context *ctx, cl_int status, cl_command_type type, cvk_command_queue *queue) :
        api_object(ctx),
        m_status(status),
        m_command_type(type),
        m_queue(queue) {}

    void set_status(cl_int status) {
        std::lock_guard<std::mutex> lock(m_lock);

        m_status = status;

        if ((m_status == CL_COMPLETE) || (m_status < 0)) {

            for (auto &type_cb : m_callbacks) {
                for (auto &cb : type_cb.second) {
                    cb.pointer(this, m_status, cb.data);
                }
            }

            m_cv.notify_all();
        }
    }

    void register_callback(cl_int callback_type, cvk_event_callback_pointer_type ptr, void *user_data) {
        cvk_event_callback cb = {ptr, user_data};
        m_callbacks[callback_type].push_back(cb);
    }

    cl_int get_status() const { return m_status; }
    cl_command_type command_type() const { return m_command_type; }

    bool is_user_event() const {
        return m_command_type == CL_COMMAND_USER;
    }

    cvk_command_queue* queue() const {
        CVK_ASSERT(!is_user_event());
        return m_queue;
    }

    cl_int wait() {
        std::unique_lock<std::mutex> lock(m_lock);
        cvk_debug("cvk_event::wait: event = %p, status = %d", this, m_status);
        if ((m_status != CL_COMPLETE) && (m_status >= 0)) {
            m_cv.wait(lock);
        }

        return m_status;
    }

    void set_profiling_info(cl_profiling_info pinfo, uint64_t val) {
        m_profiling_data[pinfo - CL_PROFILING_COMMAND_QUEUED] = val;
    }

    uint64_t get_profiling_info(cl_profiling_info pinfo) const {
        return m_profiling_data[pinfo - CL_PROFILING_COMMAND_QUEUED];
    }

    void set_profiling_info_from_monotonic_clock(cl_profiling_info pinfo) {
        auto ts_time_point = std::chrono::steady_clock::now();
        using ns = std::chrono::nanoseconds;
        uint64_t ts = std::chrono::duration_cast<ns>(ts_time_point.time_since_epoch()).count();
        set_profiling_info(pinfo, ts);
    }

private:
    std::mutex m_lock;
    std::condition_variable m_cv;
    cl_int m_status;
    cl_ulong m_profiling_data[4];
    cl_command_type m_command_type;
    cvk_command_queue *m_queue;
    std::unordered_map<cl_int, std::vector<cvk_event_callback>> m_callbacks;

} cvk_event;

struct cvk_command_group {
    std::deque<cvk_command*> commands;
};

struct cvk_executor_thread {

    cvk_executor_thread() : m_thread(nullptr), m_shutdown(false), m_profiling(false) {
        m_thread = std::make_unique<std::thread>(&cvk_executor_thread::executor, this);
    }

    void set_profiling(bool profiling) {
        m_profiling = profiling;
    }

    void send_group(std::unique_ptr<cvk_command_group> &&group) {
        m_lock.lock();
        m_groups.push_back(std::move(group));
        m_cv.notify_one();
        m_lock.unlock();
    }

    void shutdown() {

        // Tell the executor to shutdown
        m_shutdown = true;
        m_lock.lock();
        m_cv.notify_one();
        m_lock.unlock();

        // Wait for executor to shutdown
        if (m_thread != nullptr) {
            m_thread->join();
        }
    }

private:
    void executor();

    std::mutex m_lock;
    std::condition_variable m_cv;
    std::unique_ptr<std::thread> m_thread;
    bool m_shutdown;
    std::deque<std::unique_ptr<cvk_command_group>> m_groups;
    bool m_profiling;
};

typedef struct _cl_command_queue : public api_object {

    _cl_command_queue(cvk_context *ctx, cvk_device *dev,
                      cl_command_queue_properties props);

    cl_int init();

    virtual ~_cl_command_queue();

    bool has_property(cl_command_queue_properties prop) const {
        return (m_properties & prop) == prop;
    }

    void enqueue_command_with_deps(cvk_command *cmd, cl_uint num_dep_events,
                                   cvk_event *const* dep_events, cvk_event **event);
    CHECK_RETURN cl_int enqueue_command_with_deps(cvk_command *cmd, bool blocking,
                                                  cl_uint num_dep_events,
                                                  cvk_event *const* dep_events,
                                                  cvk_event **event);

    CHECK_RETURN static cl_int wait_for_events(cl_uint num_events,
                                               const cl_event *event_list);
    CHECK_RETURN cl_int flush(cvk_event** event);

    CHECK_RETURN cl_int flush() {
        return flush(nullptr);
    }

    CHECK_RETURN bool allocate_command_buffer(VkCommandBuffer *buf);

    void free_command_buffer(VkCommandBuffer buf);

    VkQueue vulkan_queue() const {
        return m_vulkan_queue;
    }

    cvk_device *device() const { return m_device; }
    cl_command_queue_properties properties() const { return m_properties; }

private:
    void enqueue_command(cvk_command *cmd, cvk_event **event);
    void executor();

    cvk_device *m_device;
    cl_command_queue_properties m_properties;

    cvk_executor_thread *m_executor;

    std::mutex m_lock;
    std::deque<std::unique_ptr<cvk_command_group>> m_groups;
    VkCommandPool m_command_pool;

    VkQueue m_vulkan_queue;
    uint32_t m_vulkan_queue_family;

} cvk_command_queue;

struct cvk_executor_thread_pool {

    ~cvk_executor_thread_pool() {
        // Shutdown all executors
        for (auto &exec_state : m_executors) {
            auto exec = exec_state.first;
            exec->shutdown();
            delete exec;
        }
    }

    cvk_executor_thread* get_executor(cvk_command_queue *queue) {

        std::unique_lock<std::mutex> lock(m_lock);

        bool profiling = queue->has_property(CL_QUEUE_PROFILING_ENABLE);

        // Try to find a free executor
        for (auto &exec_state : m_executors) {
            if (exec_state.second == executor_state::free) {
                exec_state.second = executor_state::bound;
                auto exec = exec_state.first;
                exec->set_profiling(profiling);
                return exec;
            }
        }

        // No free executor found in the pool, create a new one
        cvk_executor_thread *exec = new cvk_executor_thread();
        exec->set_profiling(profiling);
        m_executors[exec] = executor_state::bound;
        return exec;
    }

    void return_executor(cvk_executor_thread *exec) {
        // FIXME Drain all commands before returning to the pool
        std::unique_lock<std::mutex> lock(m_lock);

        m_executors[exec] = executor_state::free;
    }

private:

    enum class executor_state {
        free,
        bound,
    };

    std::mutex m_lock;
    std::unordered_map<cvk_executor_thread*, executor_state> m_executors;
};

struct cvk_command {

    cvk_command(cl_command_type type, cvk_command_queue *queue) :
        m_type(type),
        m_queue(queue),
        m_event(new cvk_event(m_queue->context(), CL_QUEUED, type, queue)) {}

    virtual ~cvk_command() {
        m_event->release();
    }

    void set_dependencies(cl_uint num_event_deps, cvk_event *const* event_deps) {
        CVK_ASSERT(m_event_deps.size() == 0);

        for (cl_uint i = 0; i < num_event_deps; i++) {
                event_deps[i]->retain();
                cvk_debug_fn("adding dep on event %p", event_deps[i]);
                m_event_deps.push_back(event_deps[i]);
        }
    }

    CHECK_RETURN cl_int execute() {

        // First wait for dependencies
        cl_int status = CL_COMPLETE;
        for (auto &ev : m_event_deps) {
            if (ev->wait() != CL_COMPLETE) {
                status = CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST;
            }
        }

        // Then execute the action if no dependencies failed
        if (status == CL_COMPLETE) {
            status = do_action();
        }

        return status;
    }

    CHECK_RETURN virtual cl_int do_action() = 0;

    cvk_event* event() const { return m_event; }

    cl_command_type type() const { return m_type; }

    cvk_command_queue *queue() const { return m_queue; }

protected:
    cl_command_type m_type;
    cvk_command_queue_holder m_queue;
    cvk_event *m_event;
private:
    std::vector<cvk_event*> m_event_deps;
};

struct cvk_command_memobj : public cvk_command {

    cvk_command_memobj(cvk_command_queue *queue, cl_command_type type, cvk_mem *memobj)
                      : cvk_command(type, queue), m_mem(memobj) {}

protected:
    cvk_mem_holder m_mem;
};

struct cvk_command_memobj_region : public cvk_command_memobj {

    cvk_command_memobj_region(cvk_command_queue *queue, cl_command_type type,
                              cvk_mem *memobj, size_t offset, size_t size)
                             : cvk_command_memobj(queue, type, memobj), m_offset(offset), m_size(size) {}

protected:
    size_t m_offset;
    size_t m_size;
};

struct cvk_command_copy : public cvk_command_memobj_region {

    cvk_command_copy(cvk_command_queue *q, cl_command_type type, cvk_mem *memobj,
                     const void *ptr, size_t offset, size_t size)
                    : cvk_command_memobj_region(q, type, memobj, offset, size),
                      m_ptr(const_cast<void*>(ptr)) {}

    virtual cl_int do_action() override;

private:
    void *m_ptr;
};

struct cvk_command_copy_rect : public cvk_command_memobj {

    cvk_command_copy_rect(cvk_command_queue *q, cl_command_type type, cvk_mem *memobj,
                          const void *ptr, const size_t *buffer_origin, const size_t *host_origin,
                          const size_t *region,
                          size_t buffer_row_pitch, size_t buffer_slice_pitch,
                          size_t host_row_pitch, size_t host_slice_pitch)
                         :
                         cvk_command_memobj(q, type, memobj),
                         m_ptr(const_cast<void*>(ptr)),
                         m_buffer_row_pitch(buffer_row_pitch),
                         m_buffer_slice_pitch(buffer_slice_pitch),
                         m_host_row_pitch(host_row_pitch),
                         m_host_slice_pitch(host_slice_pitch) {
        m_buffer_origin[0] = buffer_origin[0];
        m_buffer_origin[1] = buffer_origin[1];
        m_buffer_origin[2] = buffer_origin[2];

        m_host_origin[0] = host_origin[0];
        m_host_origin[1] = host_origin[1];
        m_host_origin[2] = host_origin[2];

        m_region[0] = region[0];
        m_region[1] = region[1];
        m_region[2] = region[2];
    }

    virtual cl_int do_action() override;

private:
    void *m_ptr;
    size_t m_buffer_row_pitch;
    size_t m_buffer_slice_pitch;
    size_t m_host_row_pitch;
    size_t m_host_slice_pitch;
    size_t m_buffer_origin[3];
    size_t m_host_origin[3];
    size_t m_region[3];
};

struct cvk_command_copy_buffer : public cvk_command {

    cvk_command_copy_buffer(cvk_command_queue *q, cl_command_type type, cvk_mem *src, cvk_mem *dst,
                            size_t src_offset, size_t dst_offset, size_t size)
                            : cvk_command(type, q), m_src_buffer(src), m_dst_buffer(dst),
                              m_src_offset(src_offset), m_dst_offset(dst_offset), m_size(size) {}

    virtual cl_int do_action() override;

private:
    cvk_mem_holder m_src_buffer;
    cvk_mem_holder m_dst_buffer;
    size_t m_src_offset;
    size_t m_dst_offset;
    size_t m_size;
};

struct cvk_command_fill : public cvk_command_memobj_region {

    cvk_command_fill(cvk_command_queue *q, cvk_mem *memobj, size_t offset, size_t size,
                     const void *pattern, size_t pattern_size)
                    : cvk_command_memobj_region(q, CL_COMMAND_FILL_BUFFER, memobj, offset, size),
                      m_pattern(std::make_unique<char[]>(pattern_size)),
                      m_pattern_size(pattern_size)
                    {
        memcpy(m_pattern.get(), pattern, pattern_size);
    }

    virtual cl_int do_action() override;

private:
    std::unique_ptr<char[]> m_pattern;
    size_t m_pattern_size;
};

struct cvk_command_kernel : public cvk_command {

    cvk_command_kernel(cvk_command_queue *q, cvk_kernel *kernel, uint32_t *num_wg, uint32_t *wg_size) :
        cvk_command(CL_COMMAND_NDRANGE_KERNEL, q),
        m_kernel(kernel),
        m_descriptor_set(VK_NULL_HANDLE),
        m_argument_values(nullptr)
    {
        m_num_wg[0] = num_wg[0];
        m_num_wg[1] = num_wg[1];
        m_num_wg[2] = num_wg[2];

        m_wg_size[0] = wg_size[0];
        m_wg_size[1] = wg_size[1];
        m_wg_size[2] = wg_size[2];
    }

    ~cvk_command_kernel() {
        if (m_descriptor_set != VK_NULL_HANDLE) {
            m_kernel->free_descriptor_set(m_descriptor_set);
        }
    }


    CHECK_RETURN cl_int build();
    virtual cl_int do_action() override;
private:
    uint32_t m_num_wg[3];
    uint32_t m_wg_size[3];
    VkCommandBuffer m_command_buffer;
    cvk_kernel_holder m_kernel;
    VkDescriptorSet m_descriptor_set;
    std::unique_ptr<cvk_kernel_argument_values> m_argument_values;
};

struct cvk_command_map : public cvk_command_memobj_region {

    cvk_command_map(cvk_command_queue *q, cl_command_type type, cvk_mem *memobj, size_t offset, size_t size)
                   : cvk_command_memobj_region(q, type, memobj, offset, size) {}
    virtual cl_int do_action() override;
};

struct cvk_command_unmap : public cvk_command_memobj {

    cvk_command_unmap(cvk_command_queue *q, cvk_mem *memobj)
                     : cvk_command_memobj(q, CL_COMMAND_UNMAP_MEM_OBJECT, memobj) {}
    virtual cl_int do_action() override;
};

struct cvk_command_dep : public cvk_command {
    cvk_command_dep(cvk_command_queue *q, cl_command_type type) : cvk_command(type, q) {}

    virtual cl_int do_action() override {
        return CL_COMPLETE;
    }
};

