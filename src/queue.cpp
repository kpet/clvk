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

#include <unordered_set>

#include "init.hpp"
#include "memory.hpp"
#include "queue.hpp"

cvk_executor_thread_pool gThreadPool;

cvk_command_queue::cvk_command_queue(cvk_context* ctx, cvk_device* device,
                                     cl_command_queue_properties properties)
    : api_object(ctx), m_device(device), m_properties(properties),
      m_executor(nullptr), m_vulkan_queue(device->vulkan_queue_allocate()),
      m_command_pool(device->vulkan_device(), m_vulkan_queue.queue_family()) {

    m_groups.push_back(std::make_unique<cvk_command_group>());

    if (properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
        cvk_warn_fn("out-of-order execution enabled, will be ignored");
    }
}

cl_int cvk_command_queue::init() {

    if (m_command_pool.init() != VK_SUCCESS) {
        return CL_OUT_OF_RESOURCES;
    }

    return CL_SUCCESS;
}

cvk_command_queue::~cvk_command_queue() {
    if (m_executor != nullptr) {
        gThreadPool.return_executor(m_executor);
    }
}

void cvk_command_queue::enqueue_command(cvk_command* cmd, _cl_event** event) {

    std::lock_guard<std::mutex> lock(m_lock);

    m_groups.back()->commands.push_back(cmd);

    cvk_debug_fn("enqueued command %p, event %p", cmd, cmd->event());

    cmd->event()->set_profiling_info_from_monotonic_clock(
        CL_PROFILING_COMMAND_QUEUED);

    if (event != nullptr) {
        // The event will be returned to the app, retain it for the user
        cmd->event()->retain();
        *event = cmd->event();
        cvk_debug_fn("returning event %p", *event);
    }
}

void cvk_command_queue::enqueue_command_with_deps(cvk_command* cmd,
                                                  cl_uint num_dep_events,
                                                  _cl_event* const* dep_events,
                                                  _cl_event** event) {
    cmd->set_dependencies(num_dep_events, dep_events);
    enqueue_command(cmd, event);
}

cl_int cvk_command_queue::enqueue_command_with_deps(
    cvk_command* cmd, bool blocking, cl_uint num_dep_events,
    _cl_event* const* dep_events, _cl_event** event) {
    cmd->set_dependencies(num_dep_events, dep_events);

    _cl_event* evt;
    enqueue_command(cmd, &evt);

    cl_int err = CL_SUCCESS;

    if (blocking) {
        err = wait_for_events(1, &evt);
    }

    if (event != nullptr) {
        *event = evt;
    } else {
        icd_downcast(evt)->release();
    }

    return err;
}

cl_int cvk_command_queue::wait_for_events(cl_uint num_events,
                                          const cl_event* event_list) {
    cl_int ret = CL_SUCCESS;

    // Create set of queues to flush
    std::unordered_set<cvk_command_queue*> queues_to_flush;
    for (cl_uint i = 0; i < num_events; i++) {
        cvk_event* event = icd_downcast(event_list[i]);

        if (!event->is_user_event()) {
            queues_to_flush.insert(event->queue());
        }
    }

    // Flush queues
    for (auto q : queues_to_flush) {
        cl_int qerr = q->flush();
        if (qerr != CL_SUCCESS) {
            return qerr;
        }
    }

    // Now wait for all the events
    for (cl_uint i = 0; i < num_events; i++) {
        cvk_event* event = icd_downcast(event_list[i]);
        if (event->wait() != CL_COMPLETE) {
            ret = CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST;
        }
    }

    return ret;
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

        auto group = std::move(m_groups.front());
        m_groups.pop_front();

        cvk_debug_fn("received group %p", group.get());

        lock.unlock();

        while (group->commands.size() > 0) {

            cvk_command* cmd = group->commands.front();
            cvk_debug_fn("executing command %p, event %p", cmd, cmd->event());

            if (m_profiling && cmd->is_profiled_by_executor()) {
                cmd->event()->set_profiling_info_from_monotonic_clock(
                    CL_PROFILING_COMMAND_START);
            }

            cl_int status = cmd->execute();
            cvk_debug_fn("command returned %d", status);

            if (m_profiling && cmd->is_profiled_by_executor()) {
                cmd->event()->set_profiling_info_from_monotonic_clock(
                    CL_PROFILING_COMMAND_END);
            }

            cmd->event()->set_status(status);

            group->commands.pop_front();

            delete cmd;
        }
        lock.lock();
    }
}

cl_int cvk_command_queue::flush(cvk_event** event) {

    cvk_info_fn("queue = %p, event = %p", this, event);

    // Get command group to the executor's queue
    std::unique_ptr<cvk_command_group> group;
    {
        std::lock_guard<std::mutex> lock(m_lock);
        if (m_groups.front()->commands.size() == 0) {
            return CL_SUCCESS;
        }
        group = std::move(m_groups.front());
        m_groups.pop_front();
        m_groups.push_back(std::make_unique<cvk_command_group>());
    }

    cvk_debug_fn("groups.size() = %zu", m_groups.size());

    CVK_ASSERT(group->commands.size() > 0);

    // Set event state and profiling info
    for (auto cmd : group->commands) {
        cmd->event()->set_status(CL_SUBMITTED);
        if (has_property(CL_QUEUE_PROFILING_ENABLE)) {
            cmd->event()->set_profiling_info_from_monotonic_clock(
                CL_PROFILING_COMMAND_SUBMIT);
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
        auto ev = group->commands.back()->event();
        ev->retain();
        *event = ev;
        cvk_debug_fn("returned event %p", *event);
    }

    // Submit command group to executor
    m_executor->send_group(std::move(group));

    return CL_SUCCESS;
}

VkResult cvk_command_pool::allocate_command_buffer(VkCommandBuffer* cmdbuf) {

    std::lock_guard<std::mutex> lock(m_lock);

    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, 0, m_command_pool,
        VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        1 // commandBufferCount
    };

    return vkAllocateCommandBuffers(m_device, &commandBufferAllocateInfo,
                                    cmdbuf);
}

void cvk_command_pool::free_command_buffer(VkCommandBuffer buf) {
    std::lock_guard<std::mutex> lock(m_lock);
    vkFreeCommandBuffers(m_device, m_command_pool, 1, &buf);
}

bool cvk_command_buffer::begin() {

    if (!m_queue->allocate_command_buffer(&m_command_buffer)) {
        return false;
    }

    m_queue->command_pool_lock();

    VkCommandBufferBeginInfo beginInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        nullptr // pInheritanceInfo
    };

    VkResult res = vkBeginCommandBuffer(m_command_buffer, &beginInfo);
    if (res != VK_SUCCESS) {
        return false;
    }

    return true;
}

bool cvk_command_buffer::submit_and_wait() {
    auto& queue = m_queue->vulkan_queue();

    VkResult res = queue.submit(m_command_buffer);

    if (res != VK_SUCCESS) {
        return false;
    }

    res = queue.wait_idle();

    if (res != VK_SUCCESS) {
        return false;
    }

    return true;
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

    if (m_wg_size[0] * m_wg_size[1] * m_wg_size[2] >
        vklimits.maxComputeWorkGroupInvocations) {
        return CL_INVALID_WORK_GROUP_SIZE;
    }

    for (int i = 0; i < 3; i++) {
        if (m_wg_size[i] > vklimits.maxComputeWorkGroupSize[i]) {
            return CL_INVALID_WORK_ITEM_SIZE;
        }
    }

    // TODO check against the size specified at compile time, if any
    // TODO CL_INVALID_KERNEL_ARGS if the kernel argument values have not been
    // specified.

    // Setup descriptors
    if (!m_kernel->setup_descriptor_sets(m_descriptor_sets.data(),
                                         m_argument_values)) {
        return CL_OUT_OF_RESOURCES;
    }

    // Create query pool
    VkQueryPoolCreateInfo query_pool_create_info = {
        VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        nullptr,
        0,                           // flags
        VK_QUERY_TYPE_TIMESTAMP,     // queryType
        NUM_POOL_QUERIES_PER_KERNEL, // queryCount
        0,                           // pipelineStatistics
    };

    bool profiling = m_queue->has_property(CL_QUEUE_PROFILING_ENABLE);

    if (profiling && !is_profiled_by_executor()) {
        auto vkdev = m_queue->device()->vulkan_device();
        auto res = vkCreateQueryPool(vkdev, &query_pool_create_info, nullptr,
                                     &m_query_pool);
        if (res != VK_SUCCESS) {
            return CL_OUT_OF_RESOURCES;
        }
    }

    if (!m_command_buffer.begin()) {
        return CL_OUT_OF_RESOURCES;
    }

    cvk_spec_constant_map specConstants = {
        {0, m_wg_size[0]},
        {1, m_wg_size[1]},
        {2, m_wg_size[2]},
    };
    for (auto const& spec_value :
         m_argument_values->specialization_constants()) {
        specConstants[spec_value.first] = spec_value.second;
    }

    m_pipeline = m_kernel->create_pipeline(specConstants);

    if (m_pipeline == VK_NULL_HANDLE) {
        return CL_OUT_OF_RESOURCES;
    }

    vkCmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      m_pipeline);

    if (profiling && !is_profiled_by_executor()) {
        vkCmdResetQueryPool(m_command_buffer, m_query_pool, 0,
                            NUM_POOL_QUERIES_PER_KERNEL);
        vkCmdWriteTimestamp(m_command_buffer,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, m_query_pool,
                            POOL_QUERY_KERNEL_START);
    }

    if (m_kernel->num_set_layouts() > 0) {
        vkCmdBindDescriptorSets(
            m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_kernel->pipeline_layout(), 0, m_kernel->num_set_layouts(),
            m_descriptor_sets.data(), 0, 0);
    }

    auto program = m_kernel->program();

    if (auto pc = program->push_constant(pushconstant::dimensions)) {
        CVK_ASSERT(pc->size == 4);
        vkCmdPushConstants(m_command_buffer, m_kernel->pipeline_layout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, pc->offset, pc->size,
                           &m_dimensions);
    }

    if (auto pc = program->push_constant(pushconstant::global_offset)) {
        CVK_ASSERT(pc->size == 12);
        vkCmdPushConstants(m_command_buffer, m_kernel->pipeline_layout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, pc->offset, pc->size,
                           &m_global_offsets);
    }

    if (m_kernel->has_pod_arguments() && !m_kernel->has_pod_buffer_arguments()) {
        for (auto& arg : m_kernel->arguments()) {
            if (arg.kind == kernel_argument_kind::pod_pushconstant) {
                CVK_ASSERT(arg.offset + arg.size <=
                           m_argument_values->pod_pushconstant_buffer().size());
                vkCmdPushConstants(
                    m_command_buffer, m_kernel->pipeline_layout(),
                    VK_SHADER_STAGE_COMPUTE_BIT, arg.offset, arg.size,
                    &m_argument_values->pod_pushconstant_buffer()[arg.offset]);
            }
        }
    }

    vkCmdDispatch(m_command_buffer, m_num_wg[0], m_num_wg[1], m_num_wg[2]);

    VkMemoryBarrier memoryBarrier = {
        VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT,
        VK_ACCESS_HOST_READ_BIT | VK_ACCESS_HOST_WRITE_BIT};

    vkCmdPipelineBarrier(m_command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT,
                         0, // dependencyFlags
                         1, // memoryBarrierCount
                         &memoryBarrier,
                         0,        // bufferMemoryBarrierCount
                         nullptr,  // pBufferMemoryBarriers
                         0,        // imageMemoryBarrierCount
                         nullptr); // pImageMemoryBarriers

    if (profiling && !is_profiled_by_executor()) {
        vkCmdWriteTimestamp(m_command_buffer,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, m_query_pool,
                            POOL_QUERY_KERNEL_END);
    }

    if (!m_command_buffer.end()) {
        return CL_OUT_OF_RESOURCES;
    }

    return CL_SUCCESS;
}

cl_int cvk_command_kernel::do_action() {
    if (!m_command_buffer.submit_and_wait()) {
        return CL_OUT_OF_RESOURCES;
    }

    bool profiling = m_queue->has_property(CL_QUEUE_PROFILING_ENABLE);

    if (profiling && !is_profiled_by_executor()) {
        uint64_t timestamps[NUM_POOL_QUERIES_PER_KERNEL];
        auto dev = m_queue->device();
        vkGetQueryPoolResults(
            dev->vulkan_device(), m_query_pool, 0, NUM_POOL_QUERIES_PER_KERNEL,
            sizeof(timestamps), timestamps, sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

        double nsPerTick = dev->vulkan_limits().timestampPeriod;

        auto ts_start_raw = timestamps[POOL_QUERY_KERNEL_START];
        auto ts_end_raw = timestamps[POOL_QUERY_KERNEL_END];
        uint64_t ts_start, ts_end;

        // Most implementations seem to use 1 ns = 1 tick, handle this as a
        // special case to not lose precision.
        if (nsPerTick == 1.0) {
            ts_start = ts_start_raw;
            ts_end = ts_end_raw;
        } else {
            ts_start = ts_start_raw * nsPerTick;
            ts_end = ts_end_raw * nsPerTick;
        }
        m_event->set_profiling_info(CL_PROFILING_COMMAND_START, ts_start);
        m_event->set_profiling_info(CL_PROFILING_COMMAND_END, ts_end);
    }

    return CL_COMPLETE;
}

cl_int cvk_command_buffer_host_copy::do_action() {
    bool success = false;

    switch (m_type) {
    case CL_COMMAND_WRITE_BUFFER:
        success = m_buffer->copy_from(m_ptr, m_offset, m_size);
        break;
    case CL_COMMAND_READ_BUFFER:
        success = m_buffer->copy_to(m_ptr, m_offset, m_size);
        break;
    default:
        CVK_ASSERT(false);
        break;
    }

    return success ? CL_COMPLETE : CL_OUT_OF_RESOURCES;
}

struct rectangle {
public:
    void set_params(const std::array<size_t, 3>& origin, size_t slicep,
                    size_t rowp, size_t elem_size) {
        m_origin = origin;
        m_slice_pitch = slicep;
        m_row_pitch = rowp;
        m_elem_size = elem_size;
    }

    size_t get_row_offset(size_t slice, size_t row) {
        return m_slice_pitch * (m_origin[2] + slice) +
               m_row_pitch * (m_origin[1] + row) + m_origin[0] * m_elem_size;
    }

private:
    std::array<size_t, 3> m_origin;
    size_t m_slice_pitch;
    size_t m_row_pitch;
    size_t m_elem_size;
};

struct memobj_map_holder {
    memobj_map_holder(cvk_mem* memobj) : m_mem(memobj), m_mapped(false) {
        CVK_ASSERT(memobj != nullptr);
    }
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
    cvk_mem* m_mem;
    bool m_mapped;
};

void cvk_rectangle_copier::do_copy(direction dir, void* src_base,
                                   void* dst_base) {
    rectangle ra, rb;

    ra.set_params(m_a_origin, m_a_slice_pitch, m_a_row_pitch, m_elem_size);
    rb.set_params(m_b_origin, m_b_slice_pitch, m_b_row_pitch, m_elem_size);

    rectangle *rsrc, *rdst;
    if (dir == direction::A_TO_B) {
        rsrc = &ra;
        rdst = &rb;
    } else {
        CVK_ASSERT(dir == direction::B_TO_A);
        rsrc = &rb;
        rdst = &ra;
    }

    for (size_t slice = 0; slice < m_region[2]; slice++) {
        // cvk_debug_fn("slice = %zu", slice);
        for (size_t row = 0; row < m_region[1]; row++) {
            // cvk_debug_fn("row = %zu (size = %zu)", row, m_region[0]);
            auto dst =
                pointer_offset(dst_base, rdst->get_row_offset(slice, row));
            auto src =
                pointer_offset(src_base, rsrc->get_row_offset(slice, row));
            memcpy(dst, src, m_region[0] * m_elem_size);
        }
    }
}

cl_int cvk_command_copy_host_buffer_rect::do_action() {
    memobj_map_holder map_holder{m_buffer};

    if (!map_holder.map()) {
        return CL_OUT_OF_RESOURCES;
    }

    cvk_rectangle_copier::direction dir;
    void *src_base, *dst_base;

    switch (m_type) {
    case CL_COMMAND_READ_BUFFER_RECT:
    case CL_COMMAND_READ_IMAGE:
        dst_base = m_hostptr;
        src_base = m_buffer->host_va();
        dir = cvk_rectangle_copier::direction::A_TO_B;
        break;
    case CL_COMMAND_WRITE_BUFFER_RECT:
    case CL_COMMAND_WRITE_IMAGE:
        dst_base = m_buffer->host_va();
        src_base = m_hostptr;
        dir = cvk_rectangle_copier::direction::B_TO_A;
        break;
    default:
        return CL_INVALID_OPERATION;
    }

    m_copier.do_copy(dir, src_base, dst_base);

    return CL_COMPLETE;
}

cl_int cvk_command_copy_buffer_rect::do_action() {
    memobj_map_holder src_map_holder{m_src_buffer};
    memobj_map_holder dst_map_holder{m_dst_buffer};

    if (!src_map_holder.map()) {
        return CL_OUT_OF_RESOURCES;
    }

    if (!dst_map_holder.map()) {
        return CL_OUT_OF_RESOURCES;
    }

    auto dir = cvk_rectangle_copier::direction::A_TO_B;
    auto src_base = m_src_buffer->host_va();
    auto dst_base = m_dst_buffer->host_va();

    m_copier.do_copy(dir, src_base, dst_base);

    return CL_COMPLETE;
}

cl_int cvk_command_copy_buffer::do_action() {
    bool success =
        m_src_buffer->copy_to(m_dst_buffer, m_src_offset, m_dst_offset, m_size);

    return success ? CL_COMPLETE : CL_OUT_OF_RESOURCES;
}

namespace {
template<typename T>
void memset_multi(void *dst, void *pattern_ptr, size_t size) {
    T pattern = *static_cast<T*>(pattern_ptr);
    auto end = pointer_offset(dst, size);
    while (dst < end) {
        *static_cast<T*>(dst) = pattern;
        dst = pointer_offset(dst, sizeof(pattern));
    }
}
}

cl_int cvk_command_fill_buffer::do_action() {
    memobj_map_holder map_holder{m_buffer};

    if (!map_holder.map()) {
        return CL_OUT_OF_RESOURCES;
    }

    auto begin = pointer_offset(m_buffer->host_va(), m_offset);
    auto end = pointer_offset(begin, m_size);

    if (m_pattern_size == 1) {
        int pattern = *reinterpret_cast<uint8_t*>(m_pattern.data());
        memset(begin, pattern, m_size);
    } else if (m_pattern_size == 2) {
        memset_multi<uint16_t>(begin, m_pattern.data(), m_size);
    } else if (m_pattern_size == 4) {
        memset_multi<uint32_t>(begin, m_pattern.data(), m_size);
    } else if (m_pattern_size == 8) {
        memset_multi<uint64_t>(begin, m_pattern.data(), m_size);
    } else {
        auto address = begin;
        while (address < end) {
            memcpy(address, m_pattern.data(), m_pattern_size);
            address = pointer_offset(address, m_pattern_size);
        }
    }

    return CL_COMPLETE;
}

cl_int cvk_command_map_buffer::build(void** map_ptr) {
    if (!m_buffer->map()) {
        return CL_OUT_OF_RESOURCES;
    }

    *map_ptr = m_buffer->map_ptr(m_offset);

    return CL_SUCCESS;
}

cl_int cvk_command_map_buffer::do_action() {
    bool success = true;
    if (m_buffer->has_flags(CL_MEM_USE_HOST_PTR)) {
        success = m_buffer->copy_to(m_buffer->host_ptr(), m_offset, m_size);
    }

    return success ? CL_COMPLETE : CL_OUT_OF_RESOURCES;
}

cl_int cvk_command_unmap_buffer::do_action() {
    m_buffer->unmap();
    return CL_COMPLETE;
}

cl_int cvk_command_unmap_image::do_action() {
    // TODO flush caches on non-coherent memory
    m_image->remove_mapping(m_mapped_ptr);

    if (m_needs_copy) {
        auto err = m_cmd_copy.do_action();
        if (err != CL_COMPLETE) {
            return err;
        }
    }

    return CL_COMPLETE;
}

VkImageSubresourceLayers prepare_subresource(cvk_image* image,
                                             std::array<size_t, 3> origin,
                                             std::array<size_t, 3> region) {
    uint32_t baseArrayLayer = 0;
    uint32_t layerCount = 1;

    switch (image->type()) {
    case CL_MEM_OBJECT_IMAGE1D_ARRAY:
        baseArrayLayer = origin[1];
        layerCount = region[1];
        break;
    case CL_MEM_OBJECT_IMAGE2D_ARRAY:
        baseArrayLayer = origin[2];
        layerCount = region[2];
        break;
    }

    VkImageSubresourceLayers ret = {VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
                                    0,                         // mipLevel
                                    baseArrayLayer, layerCount};

    return ret;
}

VkOffset3D prepare_offset(cvk_image* image, std::array<size_t, 3> origin) {

    auto x = static_cast<int32_t>(origin[0]);
    auto y = static_cast<int32_t>(origin[1]);
    auto z = static_cast<int32_t>(origin[2]);

    switch (image->type()) {
    case CL_MEM_OBJECT_IMAGE1D_ARRAY:
        y = 0;
        z = 0;
        break;
    case CL_MEM_OBJECT_IMAGE2D_ARRAY:
        z = 0;
        break;
    }

    VkOffset3D offset = {x, y, z};

    return offset;
}

VkExtent3D prepare_extent(cvk_image* image, std::array<size_t, 3> region) {
    uint32_t extentHeight = region[1];
    uint32_t extentDepth = region[2];

    switch (image->type()) {
    case CL_MEM_OBJECT_IMAGE1D_ARRAY:
        extentHeight = 1;
        extentDepth = 1;
        break;
    case CL_MEM_OBJECT_IMAGE2D_ARRAY:
        extentDepth = 1;
        break;
    }

    VkExtent3D extent = {static_cast<uint32_t>(region[0]), extentHeight,
                         extentDepth};

    return extent;
}

VkBufferImageCopy prepare_buffer_image_copy(cvk_image* image,
                                            size_t bufferOffset,
                                            std::array<size_t, 3> origin,
                                            std::array<size_t, 3> region) {
    uint32_t extentHeight = region[1];
    uint32_t extentDepth = region[2];
    uint32_t baseArrayLayer = 0;
    uint32_t layerCount = 1;
    switch (image->type()) {
    case CL_MEM_OBJECT_IMAGE1D_ARRAY:
        baseArrayLayer = origin[1];
        layerCount = region[1];
        extentHeight = 1;
        extentDepth = 1;
        break;
    case CL_MEM_OBJECT_IMAGE2D_ARRAY:
        baseArrayLayer = origin[2];
        layerCount = region[2];
        extentDepth = 1;
        break;
    }
    VkImageSubresourceLayers subResource = {
        VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
        0,                         // mipLevel
        baseArrayLayer, layerCount};

    VkOffset3D offset = prepare_offset(image, origin);
    cvk_debug_fn("offset: %d, %d, %d", offset.x, offset.y, offset.z);

    VkExtent3D extent = {static_cast<uint32_t>(region[0]), extentHeight,
                         extentDepth};
    cvk_debug_fn("extent: %u, %u, %u", extent.width, extent.height,
                 extent.depth);

    // Tightly pack the data in the destination buffer
    VkBufferImageCopy ret = {
        bufferOffset, // bufferOffset
        0,            // bufferRowLength
        0,            // bufferImageHeight
        subResource,  // imageSubresource
        offset,       // imageOffset
        extent,       // imageExtent
    };
    return ret;
}

cl_int cvk_command_map_image::build(void** map_ptr) {
    // Get a mapping
    if (!m_image->find_or_create_mapping(m_mapping, m_origin, m_region,
                                         m_flags)) {
        return CL_OUT_OF_RESOURCES;
    }

    *map_ptr = m_mapping.ptr;

    // TODO deal with CL_MEM_USE_HOST_PTR

    if (needs_copy()) {
        m_cmd_copy = std::make_unique<cvk_command_buffer_image_copy>(
            CL_COMMAND_MAP_IMAGE, m_queue, m_mapping.buffer, m_image, 0,
            m_origin, m_region);

        cl_int err = m_cmd_copy->build();
        if (err != CL_SUCCESS) {
            return err;
        }
    }

    return CL_SUCCESS;
}

cl_int cvk_command_map_image::do_action() {
    if (needs_copy()) {
        auto err = m_cmd_copy->do_action();
        if (err != CL_COMPLETE) {
            return CL_OUT_OF_RESOURCES;
        }
    }

    // TODO invalidate buffer if the memory isn't coherent

    return CL_COMPLETE;
}

void cvk_command_buffer_image_copy::build_inner_image_to_buffer(
    const VkBufferImageCopy& region) {
    VkImageSubresourceRange subresourceRange = {
        VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
        0,                         // baseMipLevel
        VK_REMAINING_MIP_LEVELS,   // levelCount
        0,                         // baseArrayLayer
        VK_REMAINING_ARRAY_LAYERS, // layerCount
    };

    VkImageMemoryBarrier imageBarrier = {
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        nullptr,
        VK_ACCESS_MEMORY_WRITE_BIT,  // srcAccessMask
        VK_ACCESS_TRANSFER_READ_BIT, // dstAccessMask
        VK_IMAGE_LAYOUT_GENERAL,     // oldLayout
        VK_IMAGE_LAYOUT_GENERAL,     // newLayout
        0,                           // srcQueueFamilyIndex
        0,                           // dstQueueFamilyIndex
        m_image->vulkan_image(),     // image
        subresourceRange,
    };

    vkCmdPipelineBarrier(m_command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0,              // dependencyFlags
                         0,              // memoryBarrierCount
                         nullptr,        // pMemoryBarriers
                         0,              // bufferMemoryBarrierCount
                         nullptr,        // pBufferMemoryBarriers
                         1,              // imageMemoryBarrierCount
                         &imageBarrier); // pImageMemoryBarriers

    vkCmdCopyImageToBuffer(m_command_buffer, m_image->vulkan_image(),
                           VK_IMAGE_LAYOUT_GENERAL, m_buffer->vulkan_buffer(),
                           1, &region);
}

void cvk_command_buffer_image_copy::build_inner_buffer_to_image(
    const VkBufferImageCopy& region) {
    VkBufferMemoryBarrier bufferBarrier = {
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        nullptr,
        VK_ACCESS_MEMORY_WRITE_BIT,
        VK_ACCESS_TRANSFER_READ_BIT,
        0, // srcQueueFamilyIndex
        0, // dstQueueFamilyIndex
        m_buffer->vulkan_buffer(),
        0, // offset
        VK_WHOLE_SIZE};

    vkCmdPipelineBarrier(m_command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0,              // dependencyFlags
                         0,              // memoryBarrierCount
                         nullptr,        // pMemoryBarriers
                         1,              // bufferMemoryBarrierCount
                         &bufferBarrier, // pBufferMemoryBarriers
                         0,              // imageMemoryBarrierCount
                         nullptr);       // pImageMemoryBarriers

    vkCmdCopyBufferToImage(m_command_buffer, m_buffer->vulkan_buffer(),
                           m_image->vulkan_image(), VK_IMAGE_LAYOUT_GENERAL, 1,
                           &region);
}

cl_int cvk_command_buffer_image_copy::build() {
    if (!m_command_buffer.begin()) {
        return CL_OUT_OF_RESOURCES;
    }

    VkBufferImageCopy region =
        prepare_buffer_image_copy(m_image, m_offset, m_origin, m_region);

    switch (type()) {
    case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
    case CL_COMMAND_MAP_IMAGE:
        build_inner_image_to_buffer(region);
        break;
    case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
    case CL_COMMAND_UNMAP_MEM_OBJECT:
        build_inner_buffer_to_image(region);
        break;
    default:
        CVK_ASSERT(false);
        break;
    }

    VkMemoryBarrier memoryBarrier = {
        VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT};

    vkCmdPipelineBarrier(
        m_command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
        // TODO HOST only when the dest buffer is an image mapping buffer
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        0, // dependencyFlags
        1, // memoryBarrierCount
        &memoryBarrier,
        0,        // bufferMemoryBarrierCount
        nullptr,  // pBufferMemoryBarriers
        0,        // imageMemoryBarrierCount
        nullptr); // pImageMemoryBarriers

    if (!m_command_buffer.end()) {
        return CL_OUT_OF_RESOURCES;
    }

    return CL_SUCCESS;
}

cl_int cvk_command_buffer_image_copy::do_action() {
    if (!m_command_buffer.submit_and_wait()) {
        return CL_OUT_OF_RESOURCES;
    }

    return CL_COMPLETE;
}

cl_int cvk_command_image_image_copy::build() {
    if (!m_command_buffer.begin()) {
        return CL_OUT_OF_RESOURCES;
    }

    VkImageSubresourceLayers srcSubresource =
        prepare_subresource(m_src_image, m_src_origin, m_region);

    VkOffset3D srcOffset = prepare_offset(m_src_image, m_src_origin);

    VkImageSubresourceLayers dstSubresource =
        prepare_subresource(m_dst_image, m_dst_origin, m_region);

    VkOffset3D dstOffset = prepare_offset(m_dst_image, m_dst_origin);

    VkExtent3D extent = prepare_extent(m_src_image, m_region);

    VkImageCopy region = {srcSubresource, srcOffset, dstSubresource, dstOffset,
                          extent};

    vkCmdCopyImage(m_command_buffer, m_src_image->vulkan_image(),
                   VK_IMAGE_LAYOUT_GENERAL, m_dst_image->vulkan_image(),
                   VK_IMAGE_LAYOUT_GENERAL, 1, &region);

    if (!m_command_buffer.end()) {
        return CL_OUT_OF_RESOURCES;
    }

    return CL_SUCCESS;
}

cl_int cvk_command_image_image_copy::do_action() {
    if (!m_command_buffer.submit_and_wait()) {
        return CL_OUT_OF_RESOURCES;
    }

    return CL_COMPLETE;
}

cl_int cvk_command_fill_image::do_action() {
    // TODO use bigger memcpy's when possible
    size_t num_elems = m_region[2] * m_region[1] * m_region[0];
    for (size_t elem = 0; elem < num_elems; elem++) {
        auto dst = pointer_offset(m_ptr, elem * m_pattern_size);
        memcpy(dst, m_pattern.data(), m_pattern_size);
    }

    return CL_COMPLETE;
}
