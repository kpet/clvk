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

#include "memory.hpp"
#include "queue.hpp"

cvk_executor_thread_pool gThreadPool;

cvk_command_queue* cvk_event::queue() const
{
    CVK_ASSERT(!is_user_event());
    return m_command->queue();
}

cl_command_type cvk_event::command_type() const
{
    CVK_ASSERT(!is_user_event());
    return m_command->type();
}

_cl_command_queue::_cl_command_queue(cvk_context *ctx, cvk_device *device,
                                     cl_command_queue_properties properties) :
    api_object(ctx),
    m_device(device),
    m_properties(properties),
    m_executor(nullptr),
    m_command_pool(VK_NULL_HANDLE)
{
    m_groups.push_back(new cvk_command_group());

    if (properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
        cvk_warn_fn("out-of-order execution enabled, will be ignored");
    }

    m_vulkan_queue = device->vulkan_queue_allocate();
    m_vulkan_queue_family = device->vulkan_queue_family();
}

cl_int cvk_command_queue::init() {

    // Create command pool
    VkCommandPoolCreateInfo createInfo = {
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        nullptr,
        0, // flags
        m_vulkan_queue_family
    };

    VkResult res = vkCreateCommandPool(m_device->vulkan_device(), &createInfo, nullptr, &m_command_pool);
    if (res != VK_SUCCESS) {
        return CL_OUT_OF_RESOURCES;
    }

    return CL_SUCCESS;
}

_cl_command_queue::~_cl_command_queue() {
    if (m_executor != nullptr) {
        gThreadPool.return_executor(m_executor);
    }
    if (m_command_pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(m_device->vulkan_device(), m_command_pool, nullptr);
    }
}

void cvk_command_queue::enqueue_command(cvk_command *cmd, cvk_event **event) {

    std::lock_guard<std::mutex> lock(m_lock);

    m_groups.back()->commands.push_back(cmd);

    cvk_debug_fn("enqueued command %p, event %p", cmd, cmd->event());

    cmd->event()->set_profiling_info_from_monotonic_clock(CL_PROFILING_COMMAND_QUEUED);

    if (event != nullptr) {
        // The event will be returned to the app, retain it for the user
        cmd->event()->retain();
        *event = cmd->event();
        cvk_debug_fn("returning event %p", *event);
    }
}

void cvk_command_queue::enqueue_command_with_deps(cvk_command *cmd, cl_uint num_dep_events,
                                                  cvk_event *const* dep_events, cvk_event **event) {
    cmd->set_dependencies(num_dep_events, dep_events);
    enqueue_command(cmd, event);
}

void cvk_executor_thread::executor() {

    std::unique_lock<std::mutex> lock(m_lock);

    while (!m_shutdown) {

        if (m_groups.size() == 0) {
            m_cv.wait(lock);
        }

        if (m_shutdown) {
            continue;
        }

        auto group = m_groups.front();
        m_groups.pop_front();

        cvk_debug_fn("received group %p", group);

        lock.unlock();

        while (group->commands.size() > 0) {

            cvk_command *cmd = group->commands.front();
            cvk_debug_fn("executing command %p, event %p", cmd, cmd->event());

            if (m_profiling) {
                cmd->event()->set_profiling_info_from_monotonic_clock(CL_PROFILING_COMMAND_START);
            }

            cl_int status = cmd->execute();
            cvk_debug_fn("command returned %d", status);

            if (m_profiling) {
                cmd->event()->set_profiling_info_from_monotonic_clock(CL_PROFILING_COMMAND_END);
            }

            cmd->event()->set_status(status);

            group->commands.pop_front();

            delete cmd;
        }
        lock.lock();

        delete group;
    }
}

cl_int cvk_command_queue::flush(cvk_event** event) {

    cvk_info_fn("queue = %p, event = %p", this, event);

    // Get command group to the executor's queue
    cvk_command_group *group;
    {
        std::lock_guard<std::mutex> lock(m_lock);
        group = m_groups.front();
        if (group->commands.size() == 0) {
            return CL_SUCCESS;
        }
        m_groups.pop_front();
        m_groups.push_back(new cvk_command_group());
    }

    cvk_debug_fn("groups.size() = %zu", m_groups.size());

    CVK_ASSERT(group->commands.size() > 0);

    // Set event state and profiling info
    for (auto cmd : group->commands) {
        cmd->event()->set_status(CL_SUBMITTED);
        if (has_property(CL_QUEUE_PROFILING_ENABLE)) {
            cmd->event()->set_profiling_info_from_monotonic_clock(CL_PROFILING_COMMAND_SUBMIT);
        }
    }

    // Create execution thread if it doesn't exist
    {
        std::lock_guard<std::mutex> lock(m_lock);
        if (m_executor == nullptr) {
            m_executor = gThreadPool.get_executor(this);
        }
    }

    if (event != nullptr) {
        *event = group->commands.back()->event();
        cvk_debug_fn("returned event %p", *event);
    }

    // Submit command group to executor
    m_executor->send_group(group);

    return CL_SUCCESS;
}

bool cvk_command_queue::allocate_command_buffer(VkCommandBuffer *buf) {

    std::lock_guard<std::mutex> lock(m_lock);

    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      0,
      m_command_pool,
      VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      1 // commandBufferCount
    };

    VkResult res = vkAllocateCommandBuffers(m_device->vulkan_device(), &commandBufferAllocateInfo, buf);

    if (res != VK_SUCCESS) {
        return false;
    }

    return true;
}

void cvk_command_queue::free_command_buffer(VkCommandBuffer buf) {
    vkFreeCommandBuffers(m_device->vulkan_device(), m_command_pool, 1, &buf);
}

cl_int cvk_command_kernel::build() {

    auto vklimits = m_queue->device()->vulkan_limits();

    // Check we have a valid dispatch size
    for (cl_uint i = 0; i < 3; ++i) {
        if (m_num_wg[i] > vklimits.maxComputeWorkGroupCount[i]) {
            cvk_error_fn("global work size exceeds device limits");
            // There is no suitable error code to report this
            // use something
            return CL_INVALID_WORK_ITEM_SIZE;
        }
    }

    if (m_wg_size[0] * m_wg_size[1] * m_wg_size[2] > vklimits.maxComputeWorkGroupInvocations) {
        return CL_INVALID_WORK_GROUP_SIZE;
    }

    for (int i = 0; i < 3; i++) {
        if (m_wg_size[i] > vklimits.maxComputeWorkGroupSize[i]) {
            return CL_INVALID_WORK_ITEM_SIZE;
        }
    }

    // TODO check against the size specified at compile time, if any

    // Setup descriptors
    if (!m_kernel->setup_descriptor_set(&m_descriptor_set, m_pod_buffer)) {
        return CL_OUT_OF_RESOURCES;
    }

    // Create and populate the command buffer
    if (!m_queue->allocate_command_buffer(&m_command_buffer)) {
        return CL_OUT_OF_RESOURCES;
    }

    VkCommandBufferBeginInfo beginInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        nullptr,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        nullptr // pInheritanceInfo
    };

    VkResult res = vkBeginCommandBuffer(m_command_buffer, &beginInfo);

    if (res != VK_SUCCESS) {
        return CL_OUT_OF_RESOURCES;
    }

    VkPipeline pipeline = m_kernel->get_pipeline(
        m_wg_size[0],
        m_wg_size[1],
        m_wg_size[2]
    );

    if (pipeline == VK_NULL_HANDLE) {
        return CL_OUT_OF_RESOURCES;
    }

    vkCmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    vkCmdBindDescriptorSets(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_kernel->pipeline_layout(), 0, 1, &m_descriptor_set, 0, 0);

    vkCmdDispatch(m_command_buffer, m_num_wg[0], m_num_wg[1], m_num_wg[2]);

    res = vkEndCommandBuffer(m_command_buffer);

    if (res != VK_SUCCESS) {
        return CL_OUT_OF_RESOURCES;
    }

    return CL_SUCCESS;
}

cl_int cvk_command_kernel::do_action()
{
    VkSubmitInfo submitInfo = {
      VK_STRUCTURE_TYPE_SUBMIT_INFO,
      nullptr,
      0, // waitSemaphoreCOunt
      nullptr, // pWaitSemaphores
      nullptr, // pWaitDstStageMask
      1, // commandBufferCount
      &m_command_buffer,
      0, // signalSemaphoreCount
      nullptr, // pSignalSemaphores
    };

    auto queue = m_queue->vulkan_queue();

    VkResult res = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);

    if (res != VK_SUCCESS) {
        cvk_error_fn("could not submit work to queue");
        return CL_OUT_OF_RESOURCES;
    }

    res = vkQueueWaitIdle(queue);

    if (res != VK_SUCCESS) {
        cvk_error_fn("could not wait for queue to become idle: %s", vulkan_error_string(res));
        return CL_OUT_OF_RESOURCES;
    }

    m_queue->free_command_buffer(m_command_buffer);

    return CL_COMPLETE;
}

cl_int cvk_command_copy::do_action()
{
    bool success = false;

    switch (m_type) {
    case CL_COMMAND_WRITE_BUFFER:
        success = m_mem->copy_from(m_ptr, m_offset, m_size);
        break;
    case CL_COMMAND_READ_BUFFER:
        success = m_mem->copy_to(m_ptr, m_offset, m_size);
        break;
    default:
        CVK_ASSERT(false);
        break;
    }

    return success ? CL_COMPLETE : CL_OUT_OF_RESOURCES;
}

struct rectangle {
public:
    void set_params(size_t *origin, size_t slicep, size_t rowp) {
        m_origin[0] = origin[0];
        m_origin[1] = origin[1];
        m_origin[2] = origin[2];
        m_slice_pitch = slicep;
        m_row_pitch = rowp;
    }

    size_t get_row_offset(size_t slice, size_t row) {
        return m_slice_pitch * (m_origin[2] + slice) + m_row_pitch * (m_origin[1] + row) + m_origin[0];
    }
private:
    size_t m_origin[3];
    size_t m_slice_pitch;
    size_t m_row_pitch;
};

struct memobj_map_holder {
    memobj_map_holder(cvk_mem *memobj) : m_mem(memobj), m_mapped(false) {}
    ~memobj_map_holder() {
        if (m_mapped) {
            m_mem->unmap();
        }
    }

    bool CHECK_RETURN map() {
        m_mapped = m_mem->map();
        return m_mapped;
    }

private:
    cvk_mem *m_mem;
    bool m_mapped;
};

cl_int cvk_command_copy_rect::do_action()
{
    uintptr_t dst_base, src_base;
    rectangle rsrc, rdst;

    memobj_map_holder map_holder{m_mem};

    if (!map_holder.map()) {
        return CL_OUT_OF_RESOURCES;
    }

    switch (m_type) {
    case CL_COMMAND_READ_BUFFER_RECT:
        dst_base = reinterpret_cast<uintptr_t>(m_ptr);
        src_base = reinterpret_cast<uintptr_t>(m_mem->host_va());
        rsrc.set_params(m_buffer_origin, m_buffer_slice_pitch, m_buffer_row_pitch);
        rdst.set_params(m_host_origin, m_host_slice_pitch, m_host_row_pitch);
        break;
    case CL_COMMAND_WRITE_BUFFER_RECT:
        dst_base = reinterpret_cast<uintptr_t>(m_mem->host_va());
        src_base = reinterpret_cast<uintptr_t>(m_ptr);
        rsrc.set_params(m_host_origin, m_host_slice_pitch, m_host_row_pitch);
        rdst.set_params(m_buffer_origin, m_buffer_slice_pitch, m_buffer_row_pitch);
        break;
    default:
        return CL_INVALID_OPERATION;
    }

    for (size_t slice = 0; slice < m_region[2]; slice++) {
        cvk_debug_fn("slice = %zu", slice);
        for (size_t row = 0; row < m_region[1]; row++) {
            cvk_debug_fn("row = %zu", row);
            uintptr_t dst = dst_base + rdst.get_row_offset(slice, row);
            uintptr_t src = src_base + rsrc.get_row_offset(slice, row);
            memcpy(reinterpret_cast<void*>(dst), reinterpret_cast<void*>(src), m_region[0]);
        }
    }

    return CL_COMPLETE;
}

cl_int cvk_command_copy_buffer::do_action()
{
    bool success = m_src_buffer->copy_to(m_dst_buffer, m_src_offset, m_dst_offset, m_size);

    return success ? CL_COMPLETE : CL_OUT_OF_RESOURCES;
}

cl_int cvk_command_fill::do_action()
{
    memobj_map_holder map_holder{m_mem};

    if (!map_holder.map()) {
        return CL_OUT_OF_RESOURCES;
    }

    uintptr_t begin = reinterpret_cast<uintptr_t>(m_mem->host_va()) + m_offset;
    uintptr_t end = begin + m_size;

    uintptr_t address = begin;
    while (address < end) {
        memcpy(reinterpret_cast<void*>(address), m_pattern.get(), m_pattern_size);
        address += m_pattern_size;
    }

    return CL_COMPLETE;
}

cl_int cvk_command_map::do_action()
{
    bool success = true;
    if (m_mem->has_flags(CL_MEM_USE_HOST_PTR)) {
        success = m_mem->copy_to(m_mem->host_ptr(), m_offset, m_size);
    }

    return success ? CL_COMPLETE : CL_OUT_OF_RESOURCES;
}

cl_int cvk_command_unmap::do_action()
{
    m_mem->unmap();

    return CL_COMPLETE;
}
