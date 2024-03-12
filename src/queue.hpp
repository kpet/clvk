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

#include <array>
#include <memory>

#include "config.hpp"
#include "event.hpp"
#include "init.hpp"
#include "kernel.hpp"
#include "objects.hpp"
#include "printf.hpp"
#include "tracing.hpp"

struct cvk_command;
struct cvk_command_queue;
struct cvk_command_batch;
using cvk_command_queue_holder = refcounted_holder<cvk_command_queue>;

struct cvk_command_group {
    std::deque<cvk_command*> commands;
    cl_int execute_cmds();
};

struct cvk_executor_thread {

    cvk_executor_thread()
        : m_thread(nullptr), m_shutdown(false), m_running(false) {
        m_thread =
            std::make_unique<std::thread>(&cvk_executor_thread::executor, this);
    }

    void send_group(std::unique_ptr<cvk_command_group>&& group) {
        m_lock.lock();
        m_groups.push_back(std::move(group));
        m_cv.notify_one();
        m_running = true;
        m_lock.unlock();
    }

    bool is_idle() {
        std::unique_lock<std::mutex> lock(m_lock);
        return !m_running;
    }

    void shutdown() {

        // Tell the executor to shutdown
        m_lock.lock();
        m_shutdown = true;
        m_cv.notify_one();
        m_lock.unlock();

        // Wait for executor to shutdown
        if (m_thread != nullptr) {
            m_thread->join();
        }
    }

    cvk_command_group extract_cmds_required_by(bool only_non_batch_cmds,
                                               cl_uint num_events,
                                               _cl_event* const* event_list);

private:
    void executor();

    std::mutex m_lock;
    std::condition_variable m_cv;
    std::unique_ptr<std::thread> m_thread;
    bool m_shutdown;
    std::deque<std::unique_ptr<cvk_command_group>> m_groups;

    bool m_running;
};

struct cvk_command_pool {

    cvk_command_pool(cvk_device* device, uint32_t queue_family)
        : m_device(device), m_queue_family(queue_family),
          m_command_pool(VK_NULL_HANDLE) {}

    ~cvk_command_pool() {
        if (m_command_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(m_device->vulkan_device(), m_command_pool,
                                 nullptr);
        }
    }

    CHECK_RETURN VkResult init() {
        VkCommandPoolCreateFlags flags = 0;
        if (m_device->is_driver_behavior_enabled(
                cvk_device::use_reset_command_buffer_bit)) {
            flags |= VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        }

        // Create command pool
        VkCommandPoolCreateInfo createInfo = {
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, flags,
            m_queue_family};

        return vkCreateCommandPool(m_device->vulkan_device(), &createInfo,
                                   nullptr, &m_command_pool);
    }

    VkResult allocate_command_buffer(VkCommandBuffer* buf);
    void free_command_buffer(VkCommandBuffer buf);

    void lock() { m_lock.lock(); }

    void unlock() { m_lock.unlock(); }

private:
    cvk_device* m_device;
    uint32_t m_queue_family;
    VkCommandPool m_command_pool;
    std::mutex m_lock;
};

struct cvk_command_queue : public _cl_command_queue,
                           api_object<object_magic::command_queue> {

    cvk_command_queue(cvk_context* ctx, cvk_device* dev,
                      cl_command_queue_properties props,
                      std::vector<cl_queue_properties>&& properties_array);

    CHECK_RETURN cl_int init();

    virtual ~cvk_command_queue();

    bool has_property(cl_command_queue_properties prop) const {
        return (m_properties & prop) == prop;
    }

    CHECK_RETURN cl_int enqueue_command_with_deps(cvk_command* cmd,
                                                  cl_uint num_dep_events,
                                                  _cl_event* const* dep_events,
                                                  _cl_event** event);
    CHECK_RETURN cl_int enqueue_command_with_deps(cvk_command* cmd,
                                                  bool blocking,
                                                  cl_uint num_dep_events,
                                                  _cl_event* const* dep_events,
                                                  _cl_event** event);

    CHECK_RETURN static cl_int wait_for_events(cl_uint num_events,
                                               const cl_event* event_list);
    CHECK_RETURN cl_int flush_no_lock();
    CHECK_RETURN cl_int flush();
    CHECK_RETURN cl_int finish();
    bool profiling_on_device() const {
        return m_device->has_timer_support() ||
               config.queue_profiling_use_timestamp_queries;
    }

    CHECK_RETURN bool allocate_command_buffer(VkCommandBuffer* cmdbuf) {
        return m_command_pool.allocate_command_buffer(cmdbuf) == VK_SUCCESS;
    }

    void free_command_buffer(VkCommandBuffer cmdbuf) {
        return m_command_pool.free_command_buffer(cmdbuf);
    }

    cvk_buffer* get_or_create_printf_buffer() {
        if (!m_printf_buffer) {
            cl_int status;
            m_printf_buffer = cvk_buffer::create(
                context(), 0, config.printf_buffer_size, nullptr, &status);
            CVK_ASSERT(status == CL_SUCCESS);
        }
        return m_printf_buffer.get();
    }

    cvk_buffer* get_printf_buffer() {
        if (!m_printf_buffer) {
            return nullptr;
        }
        return m_printf_buffer.get();
    }

    cl_int reset_printf_buffer() {
        if (m_printf_buffer && m_printf_buffer->map()) {
            memset(m_printf_buffer->host_va(), 0, 4);
            m_printf_buffer->unmap();
            return CL_SUCCESS;
        }
        cvk_error_fn("Could not reset printf buffer");
        return CL_OUT_OF_RESOURCES;
    }

    void command_pool_lock() { m_command_pool.lock(); }

    void command_pool_unlock() { m_command_pool.unlock(); }

    cvk_vulkan_queue_wrapper& vulkan_queue() { return m_vulkan_queue; }

    cvk_device* device() const { return m_device; }
    cl_command_queue_properties properties() const { return m_properties; }
    const std::vector<cl_queue_properties>& properties_array() const {
        return m_properties_array;
    }

    void batch_enqueued() {
        uint64_t batches = m_nb_batch_in_flight.fetch_add(1);
        TRACE_CNT(batch_in_flight_counter, batches + 1);
    }
    void batch_completed() {
        uint64_t batches = m_nb_batch_in_flight.fetch_sub(1);
        TRACE_CNT(batch_in_flight_counter, batches - 1);
    }

    void group_sent() {
        uint64_t group = m_nb_group_in_flight.fetch_add(1);
        TRACE_CNT(group_in_flight_counter, group + 1);
    }
    void group_completed() {
        uint64_t group = m_nb_group_in_flight.fetch_sub(1);
        TRACE_CNT(group_in_flight_counter, group - 1);
    }

    cl_int execute_cmds_required_by(cl_uint num_events,
                                    _cl_event* const* event_list);
    cl_int execute_cmds_required_by_no_lock(cl_uint num_events,
                                            _cl_event* const* event_list);

private:
    CHECK_RETURN cl_int satisfy_data_dependencies(cvk_command* cmd);
    void enqueue_command(cvk_command* cmd);
    CHECK_RETURN cl_int enqueue_command_with_retry(cvk_command*,
                                                   _cl_event** event);
    CHECK_RETURN cl_int enqueue_command(cvk_command* cmd, _cl_event** event);
    CHECK_RETURN cl_int end_current_command_batch();
    void executor();

    cvk_device* m_device;
    cl_command_queue_properties m_properties;
    std::vector<cl_queue_properties> m_properties_array;

    cvk_executor_thread* m_executor;
    cvk_event_holder m_finish_event;

    std::mutex m_lock;
    std::deque<std::unique_ptr<cvk_command_group>> m_groups;

    cvk_command_batch* m_command_batch;

    cvk_vulkan_queue_wrapper& m_vulkan_queue;
    cvk_command_pool m_command_pool;

    cl_uint m_max_cmd_batch_size;
    cl_uint m_max_first_cmd_batch_size;
    cl_uint m_max_cmd_group_size;
    cl_uint m_max_first_cmd_group_size;

    std::atomic<uint64_t> m_nb_batch_in_flight;
    std::atomic<uint64_t> m_nb_group_in_flight;

    TRACE_CNT_VAR(batch_in_flight_counter);
    TRACE_CNT_VAR(group_in_flight_counter);

    std::unique_ptr<cvk_buffer> m_printf_buffer;
};

static inline cvk_command_queue* icd_downcast(cl_command_queue queue) {
    return static_cast<cvk_command_queue*>(queue);
}

struct cvk_command_pool_lock_holder {
    cvk_command_pool_lock_holder(cvk_command_queue* queue) : m_queue(queue) {
        m_queue->command_pool_lock();
    }
    ~cvk_command_pool_lock_holder() { m_queue->command_pool_unlock(); }

private:
    cvk_command_queue* m_queue;
};

struct cvk_executor_thread_pool {

    ~cvk_executor_thread_pool() {
        // Shutdown all executors
        for (auto& exec_state : m_executors) {
            auto exec = exec_state.first;
            exec->shutdown();
            delete exec;
        }
    }

    cvk_executor_thread* get_executor() {

        std::unique_lock<std::mutex> lock(m_lock);

        // Try to find a free executor
        for (auto& exec_state : m_executors) {
            if (exec_state.second == executor_state::free) {
                exec_state.second = executor_state::bound;
                auto exec = exec_state.first;
                if (!exec->is_idle()) {
                    continue;
                }
                return exec;
            }
        }

        // No free executor found in the pool, create a new one
        cvk_executor_thread* exec = new cvk_executor_thread();
        m_executors[exec] = executor_state::bound;
        return exec;
    }

    void return_executor(cvk_executor_thread* exec) {
        // No need to drain all commands here.
        // We will make sure the thread is idle before giving it to another
        // queue.
        //
        // Also trying to wait_idle here can produce a deadlock.
        // When a queue is released before the execution of all commands, the
        // queue will be destroyed in `cvk_executor_thread::executor` when
        // releasing the holder on the queue. But at that point, the executor
        // will have the lock. When calling wait_idle here, we will never be
        // able to take the lock, generating the deadlock.
        std::unique_lock<std::mutex> lock(m_lock);

        m_executors[exec] = executor_state::free;
    }

private:
    enum class executor_state
    {
        free,
        bound,
    };

    std::mutex m_lock;
    std::unordered_map<cvk_executor_thread*, executor_state> m_executors;
};

struct cvk_command_buffer {
    cvk_command_buffer(cvk_command_queue* queue)
        : m_queue(queue), m_command_buffer(VK_NULL_HANDLE) {}

    ~cvk_command_buffer() {
        if (m_command_buffer != VK_NULL_HANDLE) {
            m_queue->free_command_buffer(m_command_buffer);
        }
    }

    CHECK_RETURN bool begin();

    CHECK_RETURN bool end() {
        auto res = vkEndCommandBuffer(m_command_buffer);
        return res == VK_SUCCESS;
    }

    CHECK_RETURN bool submit_and_wait();

    operator VkCommandBuffer() { return m_command_buffer; }

protected:
    cvk_command_queue_holder m_queue;
    VkCommandBuffer m_command_buffer;
};

#define CLVK_COMMAND_BATCH 0x5000
#define CLVK_COMMAND_IMAGE_INIT 0x5001

struct cvk_command {

    cvk_command(cl_command_type type, cvk_command_queue* queue)
        : m_type(type), m_queue(queue),
          m_event(new cvk_event(m_queue->context(), this, queue)) {}

    virtual ~cvk_command() { m_event->release(); }

    void set_dependencies(cl_uint num_event_deps,
                          _cl_event* const* event_deps) {
        CVK_ASSERT(m_event_deps.size() == 0);

        for (cl_uint i = 0; i < num_event_deps; i++) {
            auto evt = icd_downcast(event_deps[i]);
            evt->retain();
            cvk_debug_fn("adding dep on event %p", evt);
            m_event_deps.push_back(evt);
        }
    }

    void set_dependencies(const std::vector<cvk_event*>& deps) {
        CVK_ASSERT(m_event_deps.size() == 0);
        m_event_deps = deps;
    }

    virtual bool can_be_batched() const { return false; }

    virtual bool is_built_before_enqueue() const { return true; }

    // Data movement commands are only created as part of the centralised
    // memory object asynchronous data consistency management code when
    // another non-data movement command has reported the memory object. They
    // never have data movement requirements of their own.
    virtual bool is_data_movement() const { return false; }

    void add_dependency(cvk_event* dep) {
        dep->retain();
        m_event_deps.push_back(dep);
    }

    CHECK_RETURN cl_int execute() {

        // First wait for dependencies
        cl_int status = CL_COMPLETE;
        for (auto& ev : m_event_deps) {
            if (ev->wait() != CL_COMPLETE) {
                status = CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST;
            }
            ev->release();
        }

        // Then execute the action if no dependencies failed
        if (status != CL_COMPLETE) {
            cvk_error_fn("one or more dependencies have failed for cmd %p (%s)",
                         this, cl_command_type_to_string(m_type));
        } else {
            set_event_status(CL_RUNNING);
            TRACE_BEGIN_CMD(m_type, "queue", (uintptr_t) & (*m_queue),
                            "command", (uintptr_t)this);
            status = do_action();
            TRACE_END();
        }

        // When executing batch with many commands, "set_event_status" can take
        // a while. Trace it to be able to understand it easily.
        TRACE_BEGIN("set_event_status");
        set_event_status(status);
        TRACE_END();
        return status;
    }

    CHECK_RETURN virtual cl_int do_action() = 0;

    cvk_event* event() const { return m_event; }

    cl_command_type type() const { return m_type; }

    cvk_command_queue* queue() const { return m_queue; }

    const std::vector<cvk_event*>& dependencies() const { return m_event_deps; }

    virtual const std::vector<cvk_mem*> memory_objects() const {
        CVK_ASSERT(false && "Should never be called");
        return {};
    }

    virtual void set_event_status(cl_int status) {
        m_event->set_status(status);
    }

    CHECK_RETURN virtual cl_int set_profiling_info(cl_profiling_info pinfo) {
        m_event->set_profiling_info_from_monotonic_clock(pinfo);
        return CL_SUCCESS;
    }

protected:
    cl_command_type m_type;
    cvk_command_queue_holder m_queue;
    cvk_event* m_event;

private:
    std::vector<cvk_event*> m_event_deps;
};

struct cvk_command_buffer_base : public cvk_command {

    cvk_command_buffer_base(cvk_command_queue* queue, cl_command_type type,
                            cvk_buffer* buffer)
        : cvk_command(type, queue), m_buffer(buffer) {}

protected:
    cvk_buffer_holder m_buffer;
};

struct cvk_command_buffer_base_region : public cvk_command_buffer_base {

    cvk_command_buffer_base_region(cvk_command_queue* queue,
                                   cl_command_type type, cvk_buffer* buffer,
                                   size_t offset, size_t size)
        : cvk_command_buffer_base(queue, type, buffer), m_offset(offset),
          m_size(size) {}

protected:
    size_t m_offset;
    size_t m_size;
};

struct cvk_command_buffer_host_copy final
    : public cvk_command_buffer_base_region {

    cvk_command_buffer_host_copy(cvk_command_queue* q, cl_command_type type,
                                 cvk_buffer* buffer, const void* ptr,
                                 size_t offset, size_t size)
        : cvk_command_buffer_base_region(q, type, buffer, offset, size),
          m_ptr(const_cast<void*>(ptr)) {}

    CHECK_RETURN cl_int do_action() override final;

    const std::vector<cvk_mem*> memory_objects() const override {
        return {m_buffer};
    }

private:
    void* m_ptr;
};

struct cvk_rectangle_copier {

    cvk_rectangle_copier(const size_t* a_origin, const size_t* b_origin,
                         const size_t* region, size_t a_row_pitch,
                         size_t a_slice_pitch, size_t b_row_pitch,
                         size_t b_slice_pitch, size_t elem_size)
        : m_a_row_pitch(a_row_pitch), m_a_slice_pitch(a_slice_pitch),
          m_b_row_pitch(b_row_pitch), m_b_slice_pitch(b_slice_pitch),
          m_elem_size(elem_size) {

        m_a_origin[0] = a_origin[0];
        m_a_origin[1] = a_origin[1];
        m_a_origin[2] = a_origin[2];

        m_b_origin[0] = b_origin[0];
        m_b_origin[1] = b_origin[1];
        m_b_origin[2] = b_origin[2];

        m_region[0] = region[0];
        m_region[1] = region[1];
        m_region[2] = region[2];

        cvk_debug_fn("region = {%zu,%zu,%zu}", m_region[0], m_region[1],
                     m_region[2]);
    }

    enum class direction
    {
        A_TO_B,
        B_TO_A,
    };

    void do_copy(direction dir, void* src_base, void* dst_base);

private:
    std::array<size_t, 3> m_a_origin;
    size_t m_a_row_pitch;
    size_t m_a_slice_pitch;
    std::array<size_t, 3> m_b_origin;
    size_t m_b_row_pitch;
    size_t m_b_slice_pitch;
    std::array<size_t, 3> m_region;
    size_t m_elem_size;
};

struct cvk_command_copy_host_buffer_rect final : public cvk_command {

    cvk_command_copy_host_buffer_rect(
        cvk_command_queue* queue, cl_command_type type, cvk_buffer* buffer,
        void* hostptr, const size_t* host_origin, const size_t* buffer_origin,
        const size_t* region, size_t host_row_pitch, size_t host_slice_pitch,
        size_t buffer_row_pitch, size_t buffer_slice_pitch,
        size_t elem_size = 1)
        : cvk_command(type, queue),
          m_copier(buffer_origin, host_origin, region, buffer_row_pitch,
                   buffer_slice_pitch, host_row_pitch, host_slice_pitch,
                   elem_size),
          m_buffer(buffer), m_hostptr(hostptr) {}

    CHECK_RETURN cl_int do_action() override final;

    const std::vector<cvk_mem*> memory_objects() const override final {
        return {m_buffer};
    }

private:
    cvk_rectangle_copier m_copier;
    cvk_buffer_holder m_buffer;
    void* m_hostptr;
};

struct cvk_command_copy_buffer final : public cvk_command {

    cvk_command_copy_buffer(cvk_command_queue* q, cl_command_type type,
                            cvk_buffer* src, cvk_buffer* dst, size_t src_offset,
                            size_t dst_offset, size_t size)
        : cvk_command(type, q), m_src_buffer(src), m_dst_buffer(dst),
          m_src_offset(src_offset), m_dst_offset(dst_offset), m_size(size) {}

    CHECK_RETURN cl_int do_action() override final;

    const std::vector<cvk_mem*> memory_objects() const override {
        return {m_src_buffer, m_dst_buffer};
    }

private:
    cvk_buffer_holder m_src_buffer;
    cvk_buffer_holder m_dst_buffer;
    size_t m_src_offset;
    size_t m_dst_offset;
    size_t m_size;
};

struct cvk_command_copy_buffer_rect final : public cvk_command {
    cvk_command_copy_buffer_rect(cvk_command_queue* queue,
                                 cvk_buffer* src_buffer, cvk_buffer* dst_buffer,
                                 const size_t* src_origin,
                                 const size_t* dst_origin, const size_t* region,
                                 size_t src_row_pitch, size_t src_slice_pitch,
                                 size_t dst_row_pitch, size_t dst_slice_pitch)
        : cvk_command(CL_COMMAND_COPY_BUFFER_RECT, queue),
          m_copier(src_origin, dst_origin, region, src_row_pitch,
                   src_slice_pitch, dst_row_pitch, dst_slice_pitch, 1),
          m_src_buffer(src_buffer), m_dst_buffer(dst_buffer) {}

    CHECK_RETURN cl_int do_action() override final;

    const std::vector<cvk_mem*> memory_objects() const override {
        return {m_src_buffer, m_dst_buffer};
    }

private:
    cvk_rectangle_copier m_copier;
    cvk_buffer_holder m_src_buffer;
    cvk_buffer_holder m_dst_buffer;
};

struct cvk_command_fill_buffer final : public cvk_command_buffer_base_region {

    cvk_command_fill_buffer(cvk_command_queue* q, cvk_buffer* buffer,
                            size_t offset, size_t size, const void* pattern,
                            size_t pattern_size, cl_command_type type)
        : cvk_command_buffer_base_region(q, type, buffer, offset, size),
          m_pattern_size(pattern_size) {
        memcpy(m_pattern.data(), pattern, pattern_size);
    }

    CHECK_RETURN cl_int do_action() override final;

    const std::vector<cvk_mem*> memory_objects() const override {
        return {m_buffer};
    }

private:
    static constexpr int MAX_PATTERN_SIZE = 128;
    std::array<char, MAX_PATTERN_SIZE> m_pattern;
    size_t m_pattern_size;
};

struct cvk_command_batchable : public cvk_command {
    cvk_command_batchable(cl_command_type type, cvk_command_queue* queue)
        : cvk_command(type, queue), m_query_pool(VK_NULL_HANDLE) {}

    virtual ~cvk_command_batchable() {
        if (m_query_pool != VK_NULL_HANDLE) {
            auto vkdev = m_queue->device()->vulkan_device();
            vkDestroyQueryPool(vkdev, m_query_pool, nullptr);
        }
    }

    bool can_be_batched() const override;
    bool is_built_before_enqueue() const override final { return false; }

    CHECK_RETURN cl_int get_timestamp_query_results(cl_ulong* start,
                                                    cl_ulong* end);

    CHECK_RETURN cl_int build();
    CHECK_RETURN cl_int build(cvk_command_buffer& cmdbuf);
    CHECK_RETURN virtual cl_int
    build_batchable_inner(cvk_command_buffer& cmdbuf) = 0;
    CHECK_RETURN cl_int do_action() override;
    CHECK_RETURN virtual cl_int do_post_action() { return CL_SUCCESS; }

    CHECK_RETURN cl_int set_profiling_info_end(cl_ulong sync_dev,
                                               cl_ulong sync_host) {
        cl_ulong start, end;
        auto perr = get_timestamp_query_results(&start, &end);
        if (perr != CL_COMPLETE) {
            return perr;
        }
        start =
            m_queue->device()->device_timer_to_host(start, sync_dev, sync_host);
        end = m_queue->device()->device_timer_to_host(end, sync_dev, sync_host);
        m_event->set_profiling_info(CL_PROFILING_COMMAND_START, start);
        m_event->set_profiling_info(CL_PROFILING_COMMAND_END, end);
        return CL_SUCCESS;
    }

    CHECK_RETURN cl_int
    set_profiling_info(cl_profiling_info pinfo) override final {
        if (!m_queue->profiling_on_device()) {
            return cvk_command::set_profiling_info(pinfo);
        }

        if (pinfo == CL_PROFILING_COMMAND_QUEUED ||
            pinfo == CL_PROFILING_COMMAND_SUBMIT) {
            return cvk_command::set_profiling_info(pinfo);
        } else if (pinfo == CL_PROFILING_COMMAND_START) {
            return m_queue->device()->get_device_host_timer(&m_sync_dev,
                                                            &m_sync_host);
        } else {
            CVK_ASSERT(pinfo == CL_PROFILING_COMMAND_END);
            CVK_ASSERT(m_sync_dev != 0 && m_sync_host != 0);
            return set_profiling_info_end(m_sync_dev, m_sync_host);
        }
    }

private:
    std::unique_ptr<cvk_command_buffer> m_command_buffer;
    VkQueryPool m_query_pool;

    static const int NUM_POOL_QUERIES_PER_COMMAND = 2;
    static const int POOL_QUERY_CMD_START = 0;
    static const int POOL_QUERY_CMD_END = 1;

    cl_ulong m_sync_dev{}, m_sync_host{};
};

struct cvk_ndrange {

    cvk_ndrange() : offset({0}), gws({1, 1, 1}), lws({1, 1, 1}) {}

    cvk_ndrange(cl_uint work_dim, const size_t* global_work_offset,
                const size_t* global_work_size, const size_t* local_work_size)
        : cvk_ndrange() {
        for (cl_uint dim = 0; dim < work_dim; dim++) {
            if (global_work_offset != nullptr) {
                offset[dim] = global_work_offset[dim];
            }
            gws[dim] = global_work_size[dim];
            if (local_work_size != nullptr) {
                lws[dim] = local_work_size[dim];
            }
        }
    }

    bool is_uniform() const {
        for (cl_uint i = 0; i < 3; i++) {
            if (gws[i] % lws[i] != 0) {
                return false;
            }
        }
        return true;
    }

    std::array<uint32_t, 3> offset;
    std::array<uint32_t, 3> gws;
    std::array<uint32_t, 3> lws;
};

struct cvk_command_kernel final : public cvk_command_batchable {

    cvk_command_kernel(cvk_command_queue* q, cvk_kernel* kernel, uint32_t dims,
                       const cvk_ndrange& ndrange)
        : cvk_command_batchable(CL_COMMAND_NDRANGE_KERNEL, q), m_kernel(kernel),
          m_dimensions(dims), m_ndrange(ndrange), m_pipeline(VK_NULL_HANDLE),
          m_argument_values(nullptr) {}

    ~cvk_command_kernel() {
        if (m_argument_values) {
            m_argument_values->release_resources();
        }
    }

    CHECK_RETURN cl_int
    build_batchable_inner(cvk_command_buffer& cmdbuf) override final;

    CHECK_RETURN cl_int do_post_action() override final;

    bool can_be_batched() const override final {
        return !m_kernel->uses_printf() &&
               cvk_command_batchable::can_be_batched();
    }

    const std::vector<cvk_mem*> memory_objects() const override {
        std::vector<cvk_mem*> ret;
        std::shared_ptr<cvk_kernel_argument_values> argvals = m_argument_values;
        if (argvals == nullptr) {
            argvals = m_kernel->argument_values();
        }
        return argvals->memory_objects();
    }

private:
    CHECK_RETURN cl_int
    build_and_dispatch_regions(cvk_command_buffer& command_buffer);
    CHECK_RETURN cl_int
    update_global_push_constants(cvk_command_buffer& command_buffer);
    CHECK_RETURN cl_int dispatch_uniform_region_within_vklimits(
        const cvk_ndrange& region, cvk_command_buffer& command_buffer);
    CHECK_RETURN cl_int dispatch_uniform_region_iterate(
        unsigned int dim, const cvk_ndrange& region, const size_t* region_lws,
        size_t* region_gws, size_t* region_offset,
        cvk_command_buffer& command_buffer, uint32_t* num_workgroups);
    CHECK_RETURN cl_int dispatch_uniform_region(
        const cvk_ndrange& region, cvk_command_buffer& command_buffer);

    cvk_kernel_holder m_kernel;
    uint32_t m_dimensions;
    cvk_ndrange m_ndrange;
    VkPipeline m_pipeline;
    std::shared_ptr<cvk_kernel_argument_values> m_argument_values;
};

struct cvk_command_batch : public cvk_command {
    cvk_command_batch(cvk_command_queue* queue)
        : cvk_command(CLVK_COMMAND_BATCH, queue) {}

    cl_int do_action() override final;
    cl_int add_command(cvk_command_batchable* cmd) {
        if (!m_command_buffer) {
            // Create command buffer and start recording on first call
            m_command_buffer = std::make_unique<cvk_command_buffer>(m_queue);
            if (!m_command_buffer->begin()) {
                return CL_OUT_OF_RESOURCES;
            }
        }
        cvk_command_pool_lock_holder lock(m_queue);

        cl_int ret = cmd->build(*m_command_buffer);
        if (ret != CL_SUCCESS) {
            return ret;
        }

        cvk_debug_fn("add command %p (%s) to batch %p", cmd,
                     cl_command_type_to_string(cmd->type()), this);
        m_commands.emplace_back(cmd);

        return ret;
    }

    CHECK_RETURN bool end() {
        cvk_command_pool_lock_holder lock(m_queue);
        return m_command_buffer->end();
    }

    cl_uint batch_size() { return m_commands.size(); }

    CHECK_RETURN cl_int
    set_profiling_info(cl_profiling_info pinfo) override final {
        cl_int status = cvk_command::set_profiling_info(pinfo);
        if (m_queue->profiling_on_device()) {
            if (pinfo == CL_PROFILING_COMMAND_START) {
                return m_queue->device()->get_device_host_timer(&m_sync_dev,
                                                                &m_sync_host);
            } else {
                for (auto& cmd : m_commands) {
                    cl_int err;
                    if (pinfo == CL_PROFILING_COMMAND_END) {
                        err = cmd->set_profiling_info_end(m_sync_dev,
                                                          m_sync_host);
                    } else {
                        err = cmd->set_profiling_info(pinfo);
                    }
                    // do not stop at first error, but record only the first one
                    if (err != CL_SUCCESS && status == CL_SUCCESS) {
                        status = err;
                    }
                }
            }
        } else {
            for (auto& cmd : m_commands) {
                cmd->event()->copy_profiling_info(pinfo, m_event);
            }
        }
        return status;
    }

    void set_event_status(cl_int status) override final {
        m_event->set_status(status);
        for (auto& cmd : m_commands) {
            cmd->set_event_status(status);
        }
    }

private:
    std::vector<std::unique_ptr<cvk_command_batchable>> m_commands;
    std::unique_ptr<cvk_command_buffer> m_command_buffer;
    cl_ulong m_sync_dev, m_sync_host;
};

struct cvk_command_map_buffer final : public cvk_command_buffer_base_region {

    cvk_command_map_buffer(cvk_command_queue* queue, cvk_buffer* buffer,
                           size_t offset, size_t size, cl_map_flags flags,
                           cl_command_type type)
        : cvk_command_buffer_base_region(queue, type, buffer, offset, size),
          m_flags(flags), m_mapping_needs_releasing_on_destruction(false) {}
    ~cvk_command_map_buffer() {
        if (m_mapping_needs_releasing_on_destruction) {
            m_buffer->cleanup_mapping(m_mapping);
        }
    }
    CHECK_RETURN cl_int build(void** map_ptr);
    CHECK_RETURN cl_int do_action() override final;

    const std::vector<cvk_mem*> memory_objects() const override {
        return {m_buffer};
    }

private:
    cl_map_flags m_flags;
    cvk_buffer_mapping m_mapping;
    bool m_mapping_needs_releasing_on_destruction;
};

struct cvk_command_unmap_buffer final : public cvk_command_buffer_base {

    cvk_command_unmap_buffer(cvk_command_queue* queue, cvk_buffer* buffer,
                             void* map_ptr)
        : cvk_command_buffer_base(queue, CL_COMMAND_UNMAP_MEM_OBJECT, buffer),
          m_mapped_ptr(map_ptr) {}
    CHECK_RETURN cl_int do_action() override final;

    const std::vector<cvk_mem*> memory_objects() const override {
        return {m_buffer};
    }

private:
    void* m_mapped_ptr;
};

struct cvk_command_dep : public cvk_command {
    cvk_command_dep(cvk_command_queue* q, cl_command_type type)
        : cvk_command(type, q) {}

    CHECK_RETURN cl_int do_action() override final { return CL_COMPLETE; }

    const std::vector<cvk_mem*> memory_objects() const override { return {}; }
};

struct cvk_command_buffer_image_copy final : public cvk_command_batchable {
    cvk_command_buffer_image_copy(cl_command_type type,
                                  cvk_command_queue* queue, cvk_buffer* buffer,
                                  cvk_image* image, size_t offset,
                                  const std::array<size_t, 3>& origin,
                                  const std::array<size_t, 3>& region)
        : cvk_command_batchable(type, queue), m_buffer(buffer), m_image(image),
          m_offset(offset), m_origin(origin), m_region(region),
          m_copy_type(type) {}

    cvk_command_buffer_image_copy(cl_command_type type,
                                  cl_command_type copy_type,
                                  cvk_command_queue* queue, cvk_buffer* buffer,
                                  cvk_image* image, size_t offset,
                                  const std::array<size_t, 3>& origin,
                                  const std::array<size_t, 3>& region)
        : cvk_command_batchable(type, queue), m_buffer(buffer), m_image(image),
          m_offset(offset), m_origin(origin), m_region(region),
          m_copy_type(copy_type) {}

    CHECK_RETURN cl_int
    build_batchable_inner(cvk_command_buffer& cmdbuf) override final;

    const std::vector<cvk_mem*> memory_objects() const override final {
        return {m_buffer, m_image};
    }

private:
    void build_inner_image_to_buffer(cvk_command_buffer& cmdbuf,
                                     const VkBufferImageCopy& region);
    void build_inner_buffer_to_image(cvk_command_buffer& cmdbuf,
                                     const VkBufferImageCopy& region);

    cvk_buffer_holder m_buffer;
    cvk_image_holder m_image;
    size_t m_offset;
    std::array<size_t, 3> m_origin;
    std::array<size_t, 3> m_region;
    cl_command_type m_copy_type;
};

struct cvk_command_combine final : public cvk_command {
    cvk_command_combine(cvk_command_queue* queue, cl_command_type type,
                        std::vector<std::unique_ptr<cvk_command>>&& commands)
        : cvk_command(type, queue), m_commands(std::move(commands)) {}

    CHECK_RETURN cl_int do_action() override final {
        for (auto& cmd : m_commands) {
            cl_int ret = cmd->do_action();
            if (ret != CL_COMPLETE) {
                return ret;
            }
        }

        return CL_COMPLETE;
    }
    const std::vector<cvk_mem*> memory_objects() const override {
        std::vector<cvk_mem*> ret;
        // Reduce the number of reallocations
        ret.reserve(m_commands.size() * 2);
        for (auto& cmd : m_commands) {
            auto const mems = cmd->memory_objects();
            ret.insert(std::end(ret), std::begin(mems), std::end(mems));
        }
        return ret;
    }

private:
    std::vector<std::unique_ptr<cvk_command>> m_commands;
};

struct cvk_command_map_image final : public cvk_command {
    cvk_command_map_image(cvk_command_queue* q, cvk_image* img,
                          const std::array<size_t, 3>& origin,
                          const std::array<size_t, 3>& region,
                          cl_map_flags flags, bool update_host_ptr = false)
        : cvk_command(CL_COMMAND_MAP_IMAGE, q), m_image(img), m_origin(origin),
          m_region(region), m_flags(flags),
          m_update_host_ptr(update_host_ptr &&
                            m_image->has_flags(CL_MEM_USE_HOST_PTR)) {}

    CHECK_RETURN cl_int build(void** map_ptr);
    CHECK_RETURN cl_int do_action() override final;
    cvk_buffer* map_buffer() { return m_mapping.buffer; }
    size_t row_pitch() const {
        if (m_update_host_ptr) {
            return m_image->row_pitch();
        } else {
            return m_image->map_buffer_row_pitch(m_mapping);
        }
    }
    size_t slice_pitch() const {
        if (m_update_host_ptr) {
            return m_image->slice_pitch();
        } else {
            return m_image->map_buffer_slice_pitch(m_mapping);
        }
    }

    const std::vector<cvk_mem*> memory_objects() const override {
        return {m_image};
    }

private:
    bool needs_copy() const {
        return (m_flags & CL_MAP_WRITE_INVALIDATE_REGION) == 0;
    }

    cvk_image_holder m_image;
    cvk_image_mapping m_mapping;
    std::array<size_t, 3> m_origin;
    std::array<size_t, 3> m_region;
    cl_map_flags m_flags;
    std::unique_ptr<cvk_command_buffer_image_copy> m_cmd_copy;
    std::unique_ptr<cvk_command_copy_host_buffer_rect> m_cmd_host_ptr_update;
    bool m_update_host_ptr;
};

struct cvk_command_unmap_image final : public cvk_command {

    cvk_command_unmap_image(cvk_command_queue* q, cvk_image* image,
                            void* mapptr, bool update_host_ptr = false)
        : cvk_command_unmap_image(q, image, mapptr, image->mapping_for(mapptr),
                                  update_host_ptr) {
    } // FIXME crashes when the mapping doesn't exist

    cvk_command_unmap_image(cvk_command_queue* queue, cvk_image* image,
                            void* mapped_ptr, const cvk_image_mapping& mapping,
                            bool update_host_ptr)
        : cvk_command(CL_COMMAND_UNMAP_MEM_OBJECT, queue),
          m_needs_copy((mapping.flags &
                        (CL_MAP_WRITE | CL_MAP_WRITE_INVALIDATE_REGION)) != 0),
          m_mapped_ptr(mapped_ptr), m_image(image),
          m_cmd_copy(CL_COMMAND_UNMAP_MEM_OBJECT, queue, mapping.buffer, image,
                     0, mapping.origin, mapping.region),
          m_update_host_ptr(update_host_ptr &&
                            m_image->has_flags(CL_MEM_USE_HOST_PTR)) {}
    cl_int build() {
        if (m_needs_copy) {
            auto err = m_cmd_copy.build();
            if (err != CL_SUCCESS) {
                return err;
            }
            if (m_update_host_ptr) {
                size_t zero_origin[3] = {0, 0, 0};
                auto const& mapping = m_image->mapping_for(m_mapped_ptr);
                m_cmd_host_ptr_update =
                    std::make_unique<cvk_command_copy_host_buffer_rect>(
                        m_queue, CL_COMMAND_WRITE_BUFFER_RECT, mapping.buffer,
                        m_image->host_ptr(), mapping.origin.data(), zero_origin,
                        mapping.region.data(), m_image->row_pitch(),
                        m_image->slice_pitch(),
                        m_image->map_buffer_row_pitch(mapping),
                        m_image->map_buffer_slice_pitch(mapping),
                        m_image->element_size());
            }
        }
        return CL_SUCCESS;
    }
    CHECK_RETURN cl_int do_action() override final;

    const std::vector<cvk_mem*> memory_objects() const override final {
        return {m_image};
    }

private:
    bool m_needs_copy;
    void* m_mapped_ptr;
    cvk_image_holder m_image;
    cvk_command_buffer_image_copy m_cmd_copy;
    std::unique_ptr<cvk_command_copy_host_buffer_rect> m_cmd_host_ptr_update;
    bool m_update_host_ptr;
};

struct cvk_command_image_image_copy final : public cvk_command_batchable {

    cvk_command_image_image_copy(cvk_command_queue* queue, cvk_image* src_image,
                                 cvk_image* dst_image,
                                 const std::array<size_t, 3>& src_origin,
                                 const std::array<size_t, 3>& dst_origin,
                                 const std::array<size_t, 3>& region)
        : cvk_command_batchable(CL_COMMAND_COPY_IMAGE, queue),
          m_src_image(src_image), m_dst_image(dst_image),
          m_src_origin(src_origin), m_dst_origin(dst_origin), m_region(region) {
    }
    CHECK_RETURN cl_int
    build_batchable_inner(cvk_command_buffer& cmdbuf) override final;

    const std::vector<cvk_mem*> memory_objects() const override final {
        return {m_src_image, m_dst_image};
    }

private:
    cvk_image_holder m_src_image;
    cvk_image_holder m_dst_image;
    std::array<size_t, 3> m_src_origin;
    std::array<size_t, 3> m_dst_origin;
    std::array<size_t, 3> m_region;
};

struct cvk_command_fill_image final : public cvk_command {

    cvk_command_fill_image(cvk_command_queue* queue, void* ptr,
                           const cvk_image::fill_pattern_array& pattern,
                           size_t pattern_size,
                           const std::array<size_t, 3>& region)
        : cvk_command(CL_COMMAND_FILL_IMAGE, queue), m_ptr(ptr),
          m_pattern(pattern), m_pattern_size(pattern_size), m_region(region) {}
    CHECK_RETURN cl_int do_action() override final;

    const std::vector<cvk_mem*> memory_objects() const override { return {}; }

private:
    void* m_ptr;
    cvk_image::fill_pattern_array m_pattern;
    size_t m_pattern_size;
    std::array<size_t, 3> m_region;
};

struct cvk_command_image_init final : public cvk_command_batchable {

    cvk_command_image_init(cvk_command_queue* queue, cvk_image* image)
        : cvk_command_batchable(CLVK_COMMAND_IMAGE_INIT, queue),
          m_image(image) {
        CVK_ASSERT(!m_image->is_backed_by_buffer_view());
    }
    bool is_data_movement() const override { return true; }
    CHECK_RETURN cl_int
    build_batchable_inner(cvk_command_buffer& cmdbuf) override final;
    ~cvk_command_image_init() { m_image->discard_init_data(); }

private:
    cvk_image_holder m_image;
};
