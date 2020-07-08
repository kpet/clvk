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

#include "init.hpp"
#include "kernel.hpp"
#include "objects.hpp"

struct cvk_command;
struct cvk_command_queue;
struct cvk_command_kernel_group;
using cvk_command_queue_holder = refcounted_holder<cvk_command_queue>;

using cvk_event_callback_pointer_type = void(CL_CALLBACK*)(
    cl_event event, cl_int event_command_exec_status, void* user_data);

struct cvk_event_callback {
    cvk_event_callback_pointer_type pointer;
    void* data;
};

struct cvk_event : public _cl_event, api_object {

    cvk_event(cvk_context* ctx, cl_int status, cl_command_type type,
              cvk_command_queue* queue)
        : api_object(ctx), m_status(status), m_command_type(type),
          m_queue(queue) {}

    bool completed() { return m_status == CL_COMPLETE; }

    bool terminated() { return m_status < 0; }

    void set_status(cl_int status) {
        cvk_debug("cvk_event::set_status: event = %p, status = %d", this,
                  status);
        std::lock_guard<std::mutex> lock(m_lock);

        CVK_ASSERT(status < m_status);
        m_status = status;

        if (completed() || terminated()) {

            for (auto& type_cb : m_callbacks) {
                for (auto& cb : type_cb.second) {
                    execute_callback(cb);
                }
            }

            m_cv.notify_all();
        }
    }

    void register_callback(cl_int callback_type,
                           cvk_event_callback_pointer_type ptr,
                           void* user_data) {
        std::lock_guard<std::mutex> lock(m_lock);

        cvk_event_callback cb = {ptr, user_data};

        if (m_status <= callback_type) {
            execute_callback(cb);
        } else {
            m_callbacks[callback_type].push_back(cb);
        }
    }

    cl_int get_status() const { return m_status; }
    cl_command_type command_type() const { return m_command_type; }

    bool is_user_event() const { return m_command_type == CL_COMMAND_USER; }

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
    cl_ulong m_profiling_data[4];
    cl_command_type m_command_type;
    cvk_command_queue* m_queue;
    std::unordered_map<cl_int, std::vector<cvk_event_callback>> m_callbacks;
};

static inline cvk_event* icd_downcast(cl_event event) {
    return static_cast<cvk_event*>(event);
}

struct cvk_command_group {
    std::deque<cvk_command*> commands;
};

struct cvk_executor_thread {

    cvk_executor_thread()
        : m_thread(nullptr), m_shutdown(false), m_profiling(false) {
        m_thread =
            std::make_unique<std::thread>(&cvk_executor_thread::executor, this);
    }

    void set_profiling(bool profiling) { m_profiling = profiling; }

    void send_group(std::unique_ptr<cvk_command_group>&& group) {
        m_lock.lock();
        m_groups.push_back(std::move(group));
        m_cv.notify_one();
        m_lock.unlock();
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

private:
    void executor();

    std::mutex m_lock;
    std::condition_variable m_cv;
    std::unique_ptr<std::thread> m_thread;
    bool m_shutdown;
    std::deque<std::unique_ptr<cvk_command_group>> m_groups;
    bool m_profiling;
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

struct cvk_command_queue : public _cl_command_queue, api_object {

    cvk_command_queue(cvk_context* ctx, cvk_device* dev,
                      cl_command_queue_properties props);

    cl_int init();

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
    CHECK_RETURN cl_int flush(cvk_event** event);

    CHECK_RETURN cl_int flush() { return flush(nullptr); }

    CHECK_RETURN bool allocate_command_buffer(VkCommandBuffer* cmdbuf) {
        return m_command_pool.allocate_command_buffer(cmdbuf) == VK_SUCCESS;
    }

    void free_command_buffer(VkCommandBuffer cmdbuf) {
        return m_command_pool.free_command_buffer(cmdbuf);
    }

    void command_pool_lock() { m_command_pool.lock(); }

    void command_pool_unlock() { m_command_pool.unlock(); }

    cvk_vulkan_queue_wrapper& vulkan_queue() { return m_vulkan_queue; }

    cvk_device* device() const { return m_device; }
    cl_command_queue_properties properties() const { return m_properties; }

private:
    CHECK_RETURN cl_int enqueue_command(cvk_command* cmd, _cl_event** event);
    CHECK_RETURN cl_int end_current_kernel_group();
    void executor();

    cvk_device* m_device;
    cl_command_queue_properties m_properties;

    cvk_executor_thread* m_executor;

    std::mutex m_lock;
    std::deque<std::unique_ptr<cvk_command_group>> m_groups;

    cl_uint m_max_batch_size;
    cvk_command_kernel_group* m_kernel_group;

    cvk_vulkan_queue_wrapper& m_vulkan_queue;
    cvk_command_pool m_command_pool;
};

static inline cvk_command_queue* icd_downcast(cl_command_queue queue) {
    return static_cast<cvk_command_queue*>(queue);
}

struct cvk_executor_thread_pool {

    ~cvk_executor_thread_pool() {
        // Shutdown all executors
        for (auto& exec_state : m_executors) {
            auto exec = exec_state.first;
            exec->shutdown();
            delete exec;
        }
    }

    cvk_executor_thread* get_executor(cvk_command_queue* queue) {

        std::unique_lock<std::mutex> lock(m_lock);

        bool profiling = queue->has_property(CL_QUEUE_PROFILING_ENABLE);

        // Try to find a free executor
        for (auto& exec_state : m_executors) {
            if (exec_state.second == executor_state::free) {
                exec_state.second = executor_state::bound;
                auto exec = exec_state.first;
                exec->set_profiling(profiling);
                return exec;
            }
        }

        // No free executor found in the pool, create a new one
        cvk_executor_thread* exec = new cvk_executor_thread();
        exec->set_profiling(profiling);
        m_executors[exec] = executor_state::bound;
        return exec;
    }

    void return_executor(cvk_executor_thread* exec) {
        // FIXME Drain all commands before returning to the pool
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

struct cvk_command {

    cvk_command(cl_command_type type, cvk_command_queue* queue)
        : m_type(type), m_queue(queue),
          m_event(new cvk_event(m_queue->context(), CL_QUEUED, type, queue)) {}

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
        if (status == CL_COMPLETE) {
            status = do_action();
        }

        return status;
    }

    CHECK_RETURN virtual cl_int do_action() = 0;

    cvk_event* event() const { return m_event; }

    cl_command_type type() const { return m_type; }

    cvk_command_queue* queue() const { return m_queue; }

    virtual bool is_profiled_by_executor() const { return true; }

    const std::vector<cvk_event*>& dependencies() const { return m_event_deps; }

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

struct cvk_command_buffer_host_copy : public cvk_command_buffer_base_region {

    cvk_command_buffer_host_copy(cvk_command_queue* q, cl_command_type type,
                                 cvk_buffer* buffer, const void* ptr,
                                 size_t offset, size_t size)
        : cvk_command_buffer_base_region(q, type, buffer, offset, size),
          m_ptr(const_cast<void*>(ptr)) {}

    virtual cl_int do_action() override;

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

struct cvk_command_copy_host_buffer_rect : public cvk_command {

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

    virtual cl_int do_action() override;

private:
    cvk_rectangle_copier m_copier;
    cvk_buffer_holder m_buffer;
    void* m_hostptr;
};

struct cvk_command_copy_buffer : public cvk_command {

    cvk_command_copy_buffer(cvk_command_queue* q, cl_command_type type,
                            cvk_buffer* src, cvk_buffer* dst, size_t src_offset,
                            size_t dst_offset, size_t size)
        : cvk_command(type, q), m_src_buffer(src), m_dst_buffer(dst),
          m_src_offset(src_offset), m_dst_offset(dst_offset), m_size(size) {}

    virtual cl_int do_action() override;

private:
    cvk_buffer_holder m_src_buffer;
    cvk_buffer_holder m_dst_buffer;
    size_t m_src_offset;
    size_t m_dst_offset;
    size_t m_size;
};

struct cvk_command_copy_buffer_rect : public cvk_command {
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

    cl_int do_action() override;

private:
    cvk_rectangle_copier m_copier;
    cvk_buffer_holder m_src_buffer;
    cvk_buffer_holder m_dst_buffer;
};

struct cvk_command_fill_buffer : public cvk_command_buffer_base_region {

    cvk_command_fill_buffer(cvk_command_queue* q, cvk_buffer* buffer,
                            size_t offset, size_t size, const void* pattern,
                            size_t pattern_size)
        : cvk_command_buffer_base_region(q, CL_COMMAND_FILL_BUFFER, buffer,
                                         offset, size),
          m_pattern_size(pattern_size) {
        memcpy(m_pattern.data(), pattern, pattern_size);
    }

    virtual cl_int do_action() override;

private:
    static constexpr int MAX_PATTERN_SIZE = 128;
    std::array<char, MAX_PATTERN_SIZE> m_pattern;
    size_t m_pattern_size;
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
        m_queue->command_pool_unlock();
        return res == VK_SUCCESS;
    }

    CHECK_RETURN bool submit_and_wait();

    operator VkCommandBuffer() { return m_command_buffer; }

protected:
    cvk_command_queue_holder m_queue;
    VkCommandBuffer m_command_buffer;
};

struct cvk_command_kernel : public cvk_command {

    cvk_command_kernel(cvk_command_queue* q, cvk_kernel* kernel, uint32_t dims,
                       const std::array<uint32_t, 3>& global_offsets,
                       const std::array<uint32_t, 3>& gws,
                       const std::array<uint32_t, 3>& lws)
        : cvk_command(CL_COMMAND_NDRANGE_KERNEL, q), m_kernel(kernel),
          m_dimensions(dims), m_global_offsets(global_offsets), m_gws(gws),
          m_lws(lws), m_descriptor_sets{VK_NULL_HANDLE},
          m_pipeline(VK_NULL_HANDLE), m_query_pool(VK_NULL_HANDLE),
          m_argument_values(nullptr) {}

    ~cvk_command_kernel() {
        for (auto ds : m_descriptor_sets) {
            if (ds != VK_NULL_HANDLE) {
                m_kernel->free_descriptor_set(ds);
            }
        }

        if (m_query_pool != VK_NULL_HANDLE) {
            auto vkdev = m_queue->device()->vulkan_device();
            vkDestroyQueryPool(vkdev, m_query_pool, nullptr);
        }
    }

    bool is_profiled_by_executor() const override {
        return !gQueueProfilingUsesTimestampQueries;
    }

    CHECK_RETURN cl_int set_profiling_info_from_query_results();

    CHECK_RETURN cl_int build(cvk_command_buffer& command_buffer);
    virtual cl_int do_action() override;

private:
    struct cvk_ndrange {
        std::array<uint32_t, 3> offset;
        std::array<uint32_t, 3> gws;
        std::array<uint32_t, 3> lws;
    };
    CHECK_RETURN cl_int
    build_and_dispatch_regions(cvk_command_buffer& command_buffer);
    void update_global_push_constants(cvk_command_buffer& command_buffer);
    CHECK_RETURN cl_int dispatch_uniform_region(
        const cvk_ndrange& region, cvk_command_buffer& command_buffer);

    cvk_kernel_holder m_kernel;
    uint32_t m_dimensions;
    std::array<uint32_t, 3> m_global_offsets;
    std::array<uint32_t, 3> m_gws;
    std::array<uint32_t, 3> m_lws;
    std::array<VkDescriptorSet, spir_binary::MAX_DESCRIPTOR_SETS>
        m_descriptor_sets;
    VkPipeline m_pipeline;
    VkQueryPool m_query_pool;
    std::unique_ptr<cvk_kernel_argument_values> m_argument_values;

    static const int NUM_POOL_QUERIES_PER_KERNEL = 2;
    static const int POOL_QUERY_KERNEL_START = 0;
    static const int POOL_QUERY_KERNEL_END = 1;
};

struct cvk_command_kernel_group : public cvk_command {
    cvk_command_kernel_group(cvk_command_queue* queue)
        : cvk_command(CL_COMMAND_NDRANGE_KERNEL, queue) {}
    ~cvk_command_kernel_group() {}

    cl_int do_action() override;
    cl_int add_kernel(cvk_command_kernel* cmd) {
        // Create command buffer and start recording on first call
        if (!m_command_buffer) {
            m_command_buffer = std::make_unique<cvk_command_buffer>(m_queue);
            if (!m_command_buffer->begin()) {
                return CL_OUT_OF_RESOURCES;
            }
        }

        m_kernel_commands.emplace_back(cmd);
        return cmd->build(*m_command_buffer);
    }

    CHECK_RETURN bool end() { return m_command_buffer->end(); }

    cl_uint batch_size() { return m_kernel_commands.size(); }

    bool is_profiled_by_executor() const override { return false; }

private:
    CHECK_RETURN cl_int submit_and_wait();

    std::vector<std::unique_ptr<cvk_command_kernel>> m_kernel_commands;
    std::unique_ptr<cvk_command_buffer> m_command_buffer;
};

struct cvk_command_map_buffer : public cvk_command_buffer_base_region {

    cvk_command_map_buffer(cvk_command_queue* queue, cvk_buffer* buffer,
                           size_t offset, size_t size, cl_map_flags flags)
        : cvk_command_buffer_base_region(queue, CL_COMMAND_MAP_BUFFER, buffer,
                                         offset, size),
          m_flags(flags) {}
    CHECK_RETURN cl_int build(void** map_ptr);
    virtual cl_int do_action() override;

private:
    cl_map_flags m_flags;
    cvk_buffer_mapping m_mapping;
};

struct cvk_command_unmap_buffer : public cvk_command_buffer_base {

    cvk_command_unmap_buffer(cvk_command_queue* queue, cvk_buffer* buffer,
                             void* map_ptr)
        : cvk_command_buffer_base(queue, CL_COMMAND_UNMAP_MEM_OBJECT, buffer),
          m_mapped_ptr(map_ptr) {}
    virtual cl_int do_action() override;

private:
    void* m_mapped_ptr;
};

struct cvk_command_dep : public cvk_command {
    cvk_command_dep(cvk_command_queue* q, cl_command_type type)
        : cvk_command(type, q) {}

    virtual cl_int do_action() override { return CL_COMPLETE; }
};

struct cvk_command_buffer_image_copy : public cvk_command {
    cvk_command_buffer_image_copy(cl_command_type type,
                                  cvk_command_queue* queue, cvk_buffer* buffer,
                                  cvk_image* image, size_t offset,
                                  const std::array<size_t, 3>& origin,
                                  const std::array<size_t, 3>& region)
        : cvk_command(type, queue), m_command_buffer(queue), m_buffer(buffer),
          m_image(image), m_offset(offset), m_origin(origin), m_region(region) {
    }

    void build_inner_image_to_buffer(const VkBufferImageCopy& region);
    void build_inner_buffer_to_image(const VkBufferImageCopy& region);
    CHECK_RETURN cl_int build();
    virtual cl_int do_action() override;

private:
    cvk_command_buffer m_command_buffer;
    cvk_buffer_holder m_buffer;
    cvk_image_holder m_image;
    size_t m_offset;
    std::array<size_t, 3> m_origin;
    std::array<size_t, 3> m_region;
};

struct cvk_command_combine : public cvk_command {
    cvk_command_combine(cvk_command_queue* queue, cl_command_type type,
                        std::vector<std::unique_ptr<cvk_command>>&& commands)
        : cvk_command(type, queue), m_commands(std::move(commands)) {}

    virtual cl_int do_action() override {
        for (auto& cmd : m_commands) {
            cl_int ret = cmd->do_action();
            if (ret != CL_COMPLETE) {
                return ret;
            }
        }

        return CL_COMPLETE;
    }

private:
    std::vector<std::unique_ptr<cvk_command>> m_commands;
};

struct cvk_command_map_image : public cvk_command {
    cvk_command_map_image(cvk_command_queue* q, cvk_image* img,
                          const std::array<size_t, 3>& origin,
                          const std::array<size_t, 3>& region,
                          cl_map_flags flags)
        : cvk_command(CL_COMMAND_MAP_IMAGE, q), m_image(img), m_origin(origin),
          m_region(region), m_flags(flags) {}

    CHECK_RETURN cl_int build(void** map_ptr);
    virtual cl_int do_action() override;
    cvk_buffer* map_buffer() { return m_mapping.buffer; }
    size_t map_buffer_row_pitch() const {
        return m_region[0] * m_image->element_size();
    }
    size_t map_buffer_slice_pitch() const {
        switch (m_image->type()) {
        case CL_MEM_OBJECT_IMAGE1D_ARRAY:
            return map_buffer_row_pitch();
            break;
        default:
            return map_buffer_row_pitch() * m_region[1];
        }
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
};

struct cvk_command_unmap_image : public cvk_command {

    cvk_command_unmap_image(cvk_command_queue* q, cvk_image* image,
                            void* mapptr)
        : cvk_command_unmap_image(q, image, mapptr,
                                  image->mapping_for(mapptr)) {}

    cvk_command_unmap_image(cvk_command_queue* queue, cvk_image* image,
                            void* mapped_ptr, const cvk_image_mapping& mapping)
        : cvk_command(CL_COMMAND_UNMAP_MEM_OBJECT, queue),
          m_needs_copy((mapping.flags &
                        (CL_MAP_WRITE | CL_MAP_WRITE_INVALIDATE_REGION)) != 0),
          m_mapped_ptr(mapped_ptr), m_image(image),
          m_cmd_copy(CL_COMMAND_UNMAP_MEM_OBJECT, queue, mapping.buffer, image,
                     0, mapping.origin, mapping.region) {}
    cl_int build() {
        if (m_needs_copy) {
            return m_cmd_copy.build();
        } else {
            return CL_SUCCESS;
        }
    }
    virtual cl_int do_action() override;

private:
    bool m_needs_copy;
    void* m_mapped_ptr;
    cvk_image_holder m_image;
    cvk_command_buffer_image_copy m_cmd_copy;
};

struct cvk_command_image_image_copy : public cvk_command {

    cvk_command_image_image_copy(cvk_command_queue* queue, cvk_image* src_image,
                                 cvk_image* dst_image,
                                 const std::array<size_t, 3>& src_origin,
                                 const std::array<size_t, 3>& dst_origin,
                                 const std::array<size_t, 3>& region)
        : cvk_command(CL_COMMAND_COPY_IMAGE, queue), m_src_image(src_image),
          m_dst_image(dst_image), m_src_origin(src_origin),
          m_dst_origin(dst_origin), m_region(region),
          m_command_buffer(m_queue) {}
    cl_int build();
    virtual cl_int do_action() override;

private:
    cvk_image_holder m_src_image;
    cvk_image_holder m_dst_image;
    std::array<size_t, 3> m_src_origin;
    std::array<size_t, 3> m_dst_origin;
    std::array<size_t, 3> m_region;
    cvk_command_buffer m_command_buffer;
};

struct cvk_command_fill_image : public cvk_command {

    cvk_command_fill_image(cvk_command_queue* queue, void* ptr,
                           const cvk_image::fill_pattern_array& pattern,
                           size_t pattern_size,
                           const std::array<size_t, 3>& region)
        : cvk_command(CL_COMMAND_FILL_IMAGE, queue), m_ptr(ptr),
          m_pattern(pattern), m_pattern_size(pattern_size), m_region(region) {}
    cl_int do_action() override;

private:
    void* m_ptr;
    cvk_image::fill_pattern_array m_pattern;
    size_t m_pattern_size;
    std::array<size_t, 3> m_region;
};
