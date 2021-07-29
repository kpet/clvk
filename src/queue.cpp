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

static cvk_executor_thread_pool* get_thread_pool() {
    auto state = get_or_init_global_state();
    return state->thread_pool();
}

cvk_command_queue::cvk_command_queue(
    cvk_context* ctx, cvk_device* device,
    cl_command_queue_properties properties,
    std::vector<cl_queue_properties>&& properties_array)
    : api_object(ctx), m_device(device), m_properties(properties),
      m_properties_array(std::move(properties_array)), m_executor(nullptr),
      m_command_batch(nullptr), m_vulkan_queue(device->vulkan_queue_allocate()),
      m_command_pool(device, m_vulkan_queue.queue_family()) {

    m_groups.push_back(std::make_unique<cvk_command_group>());

    if (properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
        cvk_warn_fn("out-of-order execution enabled, will be ignored");
    }

    char* max_batch_size_env = getenv("CLVK_MAX_BATCH_SIZE");
    m_max_batch_size = 10000;
    if (max_batch_size_env) {
        m_max_batch_size = atoi(max_batch_size_env);
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
        get_thread_pool()->return_executor(m_executor);
    }
}

cl_int cvk_command_queue::enqueue_command(cvk_command* cmd, _cl_event** event) {

    std::lock_guard<std::mutex> lock(m_lock);

    cl_int err;

    if (cmd->can_be_batched()) {
        if (!m_command_batch) {
            // Create a new command batch
            m_command_batch = new cvk_command_batch(this);
        }

        // Add command to current batch
        err = m_command_batch->add_command(
            static_cast<cvk_command_batchable*>(cmd));
        if (err != CL_SUCCESS) {
            return err;
        }

        // End command batch when size limit reached
        if (m_command_batch->batch_size() >= m_max_batch_size) {
            if ((err = end_current_command_batch()) != CL_SUCCESS) {
                return err;
            }
        }
    } else {
        // End the current command batch
        if ((err = end_current_command_batch()) != CL_SUCCESS) {
            return err;
        }

        if (!cmd->is_built_before_enqueue()) {
            // Build batchable command as non-batched (in its own command
            // buffer)
            err = static_cast<cvk_command_batchable*>(cmd)->build();
            if (err != CL_SUCCESS) {
                return err;
            }
        }

        m_groups.back()->commands.push_back(cmd);
    }

    cvk_debug_fn("enqueued command %p, event %p", cmd, cmd->event());

    cmd->event()->set_profiling_info_from_monotonic_clock(
        CL_PROFILING_COMMAND_QUEUED);

    if (event != nullptr) {
        // The event will be returned to the app, retain it for the user
        cmd->event()->retain();
        *event = cmd->event();
        cvk_debug_fn("returning event %p", *event);
    }

    return CL_SUCCESS;
}

cl_int cvk_command_queue::enqueue_command_with_deps(
    cvk_command* cmd, cl_uint num_dep_events, _cl_event* const* dep_events,
    _cl_event** event) {
    cmd->set_dependencies(num_dep_events, dep_events);
    return enqueue_command(cmd, event);
}

cl_int cvk_command_queue::enqueue_command_with_deps(
    cvk_command* cmd, bool blocking, cl_uint num_dep_events,
    _cl_event* const* dep_events, _cl_event** event) {
    cmd->set_dependencies(num_dep_events, dep_events);

    _cl_event* evt;
    cl_int err = enqueue_command(cmd, &evt);
    if (err != CL_SUCCESS) {
        return err;
    }

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

cl_int cvk_command_queue::end_current_command_batch() {
    if (m_command_batch) {
        if (!m_command_batch->end()) {
            return CL_OUT_OF_RESOURCES;
        }
        m_groups.back()->commands.push_back(m_command_batch);
        m_command_batch = nullptr;
    }
    return CL_SUCCESS;
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

cl_int cvk_command_queue::flush_no_lock() {

    cvk_debug_fn("queue = %p", this);

    std::unique_ptr<cvk_command_group> group;

    // End current command batch
    cl_int err = end_current_command_batch();
    if (err != CL_SUCCESS) {
        return err;
    }

    // Early exit if there are no commands in the queue
    if (m_groups.front()->commands.size() == 0) {
        return CL_SUCCESS;
    }

    // Get the commands from the queue and prepare the queue to receive
    // further commands
    group = std::move(m_groups.front());
    m_groups.pop_front();
    m_groups.push_back(std::make_unique<cvk_command_group>());

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
    if (m_executor == nullptr) {
        m_executor = get_thread_pool()->get_executor(this);
    }

    auto ev = group->commands.back()->event();
    m_finish_event.reset(ev);
    cvk_debug_fn("stored event %p", ev);

    // Submit command group to executor
    m_executor->send_group(std::move(group));

    return CL_SUCCESS;
}

cl_int cvk_command_queue::flush() {
    std::lock_guard<std::mutex> lock(m_lock);
    return flush_no_lock();
}

cl_int cvk_command_queue::finish() {
    std::lock_guard<std::mutex> lock(m_lock);

    auto status = flush_no_lock();

    if ((status == CL_SUCCESS) && (m_finish_event != nullptr)) {
        m_finish_event->wait();
    }

    return status;
}

VkResult cvk_command_pool::allocate_command_buffer(VkCommandBuffer* cmdbuf) {

    std::lock_guard<std::mutex> lock(m_lock);

    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, 0, m_command_pool,
        VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        1 // commandBufferCount
    };

    return vkAllocateCommandBuffers(m_device->vulkan_device(),
                                    &commandBufferAllocateInfo, cmdbuf);
}

void cvk_command_pool::free_command_buffer(VkCommandBuffer buf) {
    std::lock_guard<std::mutex> lock(m_lock);
    vkFreeCommandBuffers(m_device->vulkan_device(), m_command_pool, 1, &buf);
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

void cvk_command_kernel::update_global_push_constants(
    cvk_command_buffer& command_buffer) {
    auto program = m_kernel->program();

    if (auto pc = program->push_constant(pushconstant::global_offset)) {
        CVK_ASSERT(pc->size == 12);
        vkCmdPushConstants(command_buffer, m_kernel->pipeline_layout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, pc->offset, pc->size,
                           &m_ndrange.offset);
    }

    if (auto pc = program->push_constant(pushconstant::enqueued_local_size)) {
        CVK_ASSERT(pc->size == 12);
        vkCmdPushConstants(command_buffer, m_kernel->pipeline_layout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, pc->offset, pc->size,
                           &m_ndrange.lws);
    }

    if (auto pc = program->push_constant(pushconstant::global_size)) {
        CVK_ASSERT(pc->size == 12);
        vkCmdPushConstants(command_buffer, m_kernel->pipeline_layout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, pc->offset, pc->size,
                           &m_ndrange.gws);
    }

    if (auto pc = program->push_constant(pushconstant::num_workgroups)) {
        CVK_ASSERT(pc->size == 12);
        uint32_t num_workgroups[3] = {m_ndrange.gws[0] / m_ndrange.lws[0],
                                      m_ndrange.gws[1] / m_ndrange.lws[1],
                                      m_ndrange.gws[2] / m_ndrange.lws[2]};

        for (int i = 0; i < 3; i++) {
            if (m_ndrange.gws[i] % m_ndrange.lws[i] != 0) {
                num_workgroups[i]++;
            }
        }
        vkCmdPushConstants(command_buffer, m_kernel->pipeline_layout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, pc->offset, pc->size,
                           &num_workgroups);
    }

    if (m_kernel->has_pod_arguments() &&
        !m_kernel->has_pod_buffer_arguments()) {
        for (auto& arg : m_kernel->arguments()) {
            if (arg.kind == kernel_argument_kind::pod_pushconstant) {
                CVK_ASSERT(arg.offset + arg.size <=
                           m_argument_values->pod_data().size());

                // Vulkan valid usage states push constants can only be updated
                // in chunks whose offset and size are a multiple of 4.
                uint32_t size = round_up(arg.size, 4);
                uint32_t offset = arg.offset & ~0x3U;
                vkCmdPushConstants(command_buffer, m_kernel->pipeline_layout(),
                                   VK_SHADER_STAGE_COMPUTE_BIT, offset, size,
                                   &m_argument_values->pod_data()[offset]);
            }
        }
    }
}

cl_int cvk_command_kernel::dispatch_uniform_region(
    const cvk_ndrange& region, cvk_command_buffer& command_buffer) {

    // Calculate number of workgroups for region
    std::array<uint32_t, 3> num_workgroups;
    for (cl_uint i = 0; i < 3; i++) {
        CVK_ASSERT(region.gws[i] % region.lws[i] == 0);
        num_workgroups[i] = region.gws[i] / region.lws[i];
    };

    // Checks region satisfies the vulkan limits
    auto& vklimits = m_queue->device()->vulkan_limits();
    for (cl_uint i = 0; i < 3; ++i) {
        if (num_workgroups[i] > vklimits.maxComputeWorkGroupCount[i]) {
            cvk_error_fn("global work size (%d, %d, %d) exceeds device limits"
                         " of (%d, %d, %d)",
                         num_workgroups[0], num_workgroups[1],
                         num_workgroups[2],
                         vklimits.maxComputeWorkGroupCount[0],
                         vklimits.maxComputeWorkGroupCount[1],
                         vklimits.maxComputeWorkGroupCount[2]);

            // TODO split further
            return CL_INVALID_WORK_ITEM_SIZE;
        }
    }

    if (region.lws[0] * region.lws[1] * region.lws[2] >
        vklimits.maxComputeWorkGroupInvocations) {
        return CL_INVALID_WORK_GROUP_SIZE;
    }

    for (int i = 0; i < 3; i++) {
        if (region.lws[i] > vklimits.maxComputeWorkGroupSize[i]) {
            return CL_INVALID_WORK_ITEM_SIZE;
        }
    }

    auto program = m_kernel->program();
    auto constants = program->spec_constants();
    // TODO: if all kernels in the module use the same reqd_workgroup_size ,
    // clspv will not generate specialization constants for workgroup size, but
    // these values should be error checked.
    uint32_t wgsize_x_id = 0;
    auto where = constants.find(spec_constant::workgroup_size_x);
    if (where != constants.end()) {
        wgsize_x_id = where->second;
    }
    uint32_t wgsize_y_id = 1;
    where = constants.find(spec_constant::workgroup_size_y);
    if (where != constants.end()) {
        wgsize_y_id = where->second;
    }
    uint32_t wgsize_z_id = 2;
    where = constants.find(spec_constant::workgroup_size_z);
    if (where != constants.end()) {
        wgsize_z_id = where->second;
    }

    cvk_spec_constant_map specConstants = {
        {wgsize_x_id, region.lws[0]},
        {wgsize_y_id, region.lws[1]},
        {wgsize_z_id, region.lws[2]},
    };
    for (auto const& spec_value :
         m_argument_values->specialization_constants()) {
        specConstants[spec_value.first] = spec_value.second;
    }
    // Clspv allocates a spec constant for work dimensions if get_work_dim() is
    // used.
    where = constants.find(spec_constant::work_dim);
    if (where != constants.end()) {
        uint32_t dim_id = where->second;
        specConstants[dim_id] = m_dimensions;
    }

    // Clspv can allocate spec constants for global offset.
    where = constants.find(spec_constant::global_offset_x);
    if (where != constants.end()) {
        uint32_t offset_id = where->second;
        specConstants[offset_id] = m_ndrange.offset[0];
    }
    where = constants.find(spec_constant::global_offset_y);
    if (where != constants.end()) {
        uint32_t offset_id = where->second;
        specConstants[offset_id] = m_ndrange.offset[1];
    }
    where = constants.find(spec_constant::global_offset_z);
    if (where != constants.end()) {
        uint32_t offset_id = where->second;
        specConstants[offset_id] = m_ndrange.offset[2];
    }

    m_pipeline = m_kernel->create_pipeline(specConstants);

    if (m_pipeline == VK_NULL_HANDLE) {
        return CL_OUT_OF_RESOURCES;
    }

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      m_pipeline);

    if (auto pc = program->push_constant(pushconstant::region_offset)) {
        CVK_ASSERT(pc->size == 12);
        uint32_t region_offsets[3] = {
            m_ndrange.offset[0] + region.offset[0],
            m_ndrange.offset[1] + region.offset[1],
            m_ndrange.offset[2] + region.offset[2],
        };
        vkCmdPushConstants(command_buffer, m_kernel->pipeline_layout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, pc->offset, pc->size,
                           &region_offsets);
    }

    if (auto pc = program->push_constant(pushconstant::region_group_offset)) {
        CVK_ASSERT(pc->size == 12);
        uint32_t region_group_offsets[3] = {
            region.offset[0] / m_ndrange.lws[0],
            region.offset[1] / m_ndrange.lws[1],
            region.offset[2] / m_ndrange.lws[2],
        };
        vkCmdPushConstants(command_buffer, m_kernel->pipeline_layout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, pc->offset, pc->size,
                           &region_group_offsets);
    }

    vkCmdDispatch(command_buffer, num_workgroups[0], num_workgroups[1],
                  num_workgroups[2]);

    return CL_SUCCESS;
}

cl_int cvk_command_kernel::build_and_dispatch_regions(
    cvk_command_buffer& command_buffer) {

    // Split non-uniform NDRange into uniform regions
    cvk_ndrange regions[8];
    regions[0].offset = {0};
    regions[0].gws = m_ndrange.gws;
    regions[0].lws = m_ndrange.lws;
    uint32_t stackpos = 0;
    uint32_t num_regions = 1;
    do {
        cvk_ndrange* region = &regions[stackpos];
        for (uint32_t dim = 0; dim < m_dimensions; dim++) {
            auto mod = region->gws[dim] % region->lws[dim];

            cvk_ndrange* splitout_region = &regions[num_regions];

            if (mod != 0) {
                *splitout_region = *region;

                auto quo = region->gws[dim] - mod;

                region->gws[dim] = quo;

                splitout_region->gws[dim] = mod;
                splitout_region->lws[dim] = mod;
                splitout_region->offset[dim] = quo;

                num_regions++;
            }
        }
    } while (++stackpos < num_regions);

    // Dispatch regions
    for (uint32_t i = 0; i < num_regions; i++) {
        cvk_debug("region %u: gws = {%u,%u,%u}, lws = {%u,%u,%u}, offset = "
                  "{%u,%u,%u}",
                  i, regions[i].gws[0], regions[i].gws[1], regions[i].gws[2],
                  regions[i].lws[0], regions[i].lws[1], regions[i].lws[2],
                  regions[i].offset[0], regions[i].offset[1],
                  regions[i].offset[2]);
        auto err = dispatch_uniform_region(regions[i], command_buffer);
        if (err != CL_SUCCESS) {
            return err;
        }
    }

    return CL_SUCCESS;
}

cl_int
cvk_command_kernel::build_batchable_inner(cvk_command_buffer& command_buffer) {

    // TODO check against the size specified at compile time, if any
    // TODO CL_INVALID_KERNEL_ARGS if the kernel argument values have not been
    // specified.

    m_argument_values = m_kernel->argument_values();
    m_argument_values->retain_resources();

    // Setup descriptors
    if (!m_argument_values->setup_descriptor_sets()) {
        return CL_OUT_OF_RESOURCES;
    }

    // Bind descriptors and update push constants
    if (m_kernel->num_set_layouts() > 0) {
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                m_kernel->pipeline_layout(), 0,
                                m_kernel->num_set_layouts(),
                                m_argument_values->descriptor_sets(), 0, 0);
    }

    update_global_push_constants(command_buffer);

    // Dispatch work
    auto err = build_and_dispatch_regions(command_buffer);
    if (err != CL_SUCCESS) {
        return err;
    }

    // Synchronise wrt to memory and other commands
    VkMemoryBarrier memoryBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, // sType
                                     nullptr,                          // pNext
                                     VK_ACCESS_SHADER_WRITE_BIT,
                                     VK_ACCESS_MEMORY_READ_BIT |
                                         VK_ACCESS_MEMORY_WRITE_BIT};

    // Workaround for a bug on some NVIDIA devices.
    // This should already be covered by VK_ACCESS_MEMORY_READ_BIT.
    memoryBarrier.dstAccessMask |= VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(
        command_buffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, // srcStageMask
        VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, // dstStageMask
        0,                                        // dependencyFlags
        1,                                        // memoryBarrierCount
        &memoryBarrier,
        0,        // bufferMemoryBarrierCount
        nullptr,  // pBufferMemoryBarriers
        0,        // imageMemoryBarrierCount
        nullptr); // pImageMemoryBarriers

    return CL_SUCCESS;
}

bool cvk_command_batchable::can_be_batched() const {
    bool unresolved_user_event_dependencies = false;
    bool unresolved_other_queue_dependencies = false;

    for (auto ev : dependencies()) {
        if (ev->is_user_event()) {
            if (!ev->completed()) {
                unresolved_user_event_dependencies = true;
                break;
            } else {
                continue;
            }
        }

        if ((ev->queue() != queue()) && !ev->completed()) {
            unresolved_other_queue_dependencies = true;
            break;
        }
    }

    return !unresolved_user_event_dependencies &&
           !unresolved_other_queue_dependencies;
}

cl_int cvk_command_batchable::build() {
    m_command_buffer = std::make_unique<cvk_command_buffer>(m_queue);
    if (!m_command_buffer->begin()) {
        return CL_OUT_OF_RESOURCES;
    }

    cl_int err = build(*m_command_buffer);
    if (err != CL_SUCCESS) {
        return err;
    }

    if (!m_command_buffer->end()) {
        return CL_OUT_OF_RESOURCES;
    }

    return CL_SUCCESS;
}

cl_int cvk_command_batchable::build(cvk_command_buffer& command_buffer) {
    // Create query pool
    VkQueryPoolCreateInfo query_pool_create_info = {
        VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        nullptr,
        0,                            // flags
        VK_QUERY_TYPE_TIMESTAMP,      // queryType
        NUM_POOL_QUERIES_PER_COMMAND, // queryCount
        0,                            // pipelineStatistics
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

    // Sample timestamp if profiling
    if (profiling && !is_profiled_by_executor()) {
        vkCmdResetQueryPool(command_buffer, m_query_pool, 0,
                            NUM_POOL_QUERIES_PER_COMMAND);
        vkCmdWriteTimestamp(command_buffer,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, m_query_pool,
                            POOL_QUERY_CMD_START);
    }

    auto err = build_batchable_inner(command_buffer);
    if (err != CL_SUCCESS) {
        return CL_OUT_OF_RESOURCES;
    }

    // Sample timestamp if profiling
    if (profiling && !is_profiled_by_executor()) {
        vkCmdWriteTimestamp(command_buffer,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, m_query_pool,
                            POOL_QUERY_CMD_END);
    }

    return CL_SUCCESS;
}

cl_int cvk_command_batchable::get_timestamp_query_results(cl_ulong* start,
                                                          cl_ulong* end) {
    uint64_t timestamps[NUM_POOL_QUERIES_PER_COMMAND];
    auto dev = m_queue->device();
    auto res = vkGetQueryPoolResults(
        dev->vulkan_device(), m_query_pool, 0, NUM_POOL_QUERIES_PER_COMMAND,
        sizeof(timestamps), timestamps, sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (res != VK_SUCCESS) {
        return CL_OUT_OF_RESOURCES;
    }

    auto ts_start_raw = timestamps[POOL_QUERY_CMD_START];
    auto ts_end_raw = timestamps[POOL_QUERY_CMD_END];

    *start = dev->timestamp_to_ns(ts_start_raw);
    *end = dev->timestamp_to_ns(ts_end_raw);

    return CL_COMPLETE;
}

cl_int cvk_command_batchable::do_action() {
    CVK_ASSERT(m_command_buffer);

    bool profiling = m_queue->has_property(CL_QUEUE_PROFILING_ENABLE);
    auto dev = m_queue->device();

    cl_ulong sync_host, sync_dev;
    if (profiling && dev->has_timer_support()) {
        if (dev->get_device_host_timer(&sync_dev, &sync_host) != CL_SUCCESS) {
            return CL_OUT_OF_RESOURCES;
        }
    }

    if (!m_command_buffer->submit_and_wait()) {
        return CL_OUT_OF_RESOURCES;
    }

    cl_int err = CL_COMPLETE;

    if (profiling && m_queue->profiling_on_device()) {
        cl_ulong start, end;
        err = get_timestamp_query_results(&start, &end);
        if (dev->has_timer_support()) {
            start = dev->device_timer_to_host(start, sync_dev, sync_host);
            end = dev->device_timer_to_host(end, sync_dev, sync_host);
        }
        m_event->set_profiling_info(CL_PROFILING_COMMAND_START, start);
        m_event->set_profiling_info(CL_PROFILING_COMMAND_END, end);
    }

    return err;
}

cl_int cvk_command_batch::submit_and_wait() {
    bool profiling = m_queue->has_property(CL_QUEUE_PROFILING_ENABLE);

    if (profiling && !m_queue->profiling_on_device()) {
        m_event->set_profiling_info_from_monotonic_clock(
            CL_PROFILING_COMMAND_START);
    }

    if (!m_command_buffer->submit_and_wait()) {
        return CL_OUT_OF_RESOURCES;
    }

    if (profiling && !m_queue->profiling_on_device()) {
        m_event->set_profiling_info_from_monotonic_clock(
            CL_PROFILING_COMMAND_END);
    }

    return CL_COMPLETE;
}

cl_int cvk_command_batch::do_action() {

    cvk_info("executing batch of %lu commands", m_commands.size());

    cl_ulong sync_host, sync_dev;
    auto dev = m_queue->device();
    if (dev->has_timer_support()) {
        if (dev->get_device_host_timer(&sync_dev, &sync_host) != CL_SUCCESS) {
            return CL_OUT_OF_RESOURCES;
        }
    }
    cl_int status = submit_and_wait();

    bool profiling = m_queue->has_property(CL_QUEUE_PROFILING_ENABLE);

    for (auto& cmd : m_commands) {

        auto ev = cmd->event();

        if (profiling) {
            ev->copy_profiling_info(CL_PROFILING_COMMAND_SUBMIT, m_event);
            if (m_queue->profiling_on_device()) {
                cl_ulong start, end;
                auto perr = cmd->get_timestamp_query_results(&start, &end);
                // Report the first error if no errors were present
                // Keep going through the events
                if (status == CL_COMPLETE) {
                    status = perr;
                }
                if (dev->has_timer_support()) {
                    start =
                        dev->device_timer_to_host(start, sync_dev, sync_host);
                    end = dev->device_timer_to_host(end, sync_dev, sync_host);
                }
                ev->set_profiling_info(CL_PROFILING_COMMAND_START, start);
                ev->set_profiling_info(CL_PROFILING_COMMAND_END, end);
            } else {
                ev->copy_profiling_info(CL_PROFILING_COMMAND_START, m_event);
                ev->copy_profiling_info(CL_PROFILING_COMMAND_END, m_event);
            }
        }

        ev->set_status(status);
    }

    return status;
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
template <typename T>
void memset_multi(void* dst, void* pattern_ptr, size_t size) {
    T pattern = *static_cast<T*>(pattern_ptr);
    auto end = pointer_offset(dst, size);
    while (dst < end) {
        *static_cast<T*>(dst) = pattern;
        dst = pointer_offset(dst, sizeof(pattern));
    }
}
} // namespace

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

    if (!m_buffer->find_or_create_mapping(m_mapping, m_offset, m_size,
                                          m_flags)) {
        return CL_OUT_OF_RESOURCES;
    }

    *map_ptr = m_mapping.buffer->map_ptr(m_offset);

    return CL_SUCCESS;
}

cl_int cvk_command_map_buffer::do_action() {
    bool success = true;

    if (!m_buffer->insert_mapping(m_mapping)) {
        return false;
    }

    if (m_buffer->has_flags(CL_MEM_USE_HOST_PTR)) {
        auto dst = m_mapping.buffer->host_ptr();
        dst = pointer_offset(dst, m_offset);
        success = m_buffer->copy_to(dst, m_offset, m_size);
    }

    return success ? CL_COMPLETE : CL_OUT_OF_RESOURCES;
}

cl_int cvk_command_unmap_buffer::do_action() {
    bool success = true;

    auto mapping = m_buffer->remove_mapping(m_mapped_ptr);

    if (m_buffer->has_flags(CL_MEM_USE_HOST_PTR)) {
        auto src = m_buffer->host_ptr();
        src = pointer_offset(src, mapping.offset);
        success = mapping.buffer->copy_from(src, mapping.offset, mapping.size);
    }

    return success ? CL_COMPLETE : CL_OUT_OF_RESOURCES;
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
    cvk_command_buffer& cmdbuf, const VkBufferImageCopy& region) {
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

    vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0,              // dependencyFlags
                         0,              // memoryBarrierCount
                         nullptr,        // pMemoryBarriers
                         0,              // bufferMemoryBarrierCount
                         nullptr,        // pBufferMemoryBarriers
                         1,              // imageMemoryBarrierCount
                         &imageBarrier); // pImageMemoryBarriers

    vkCmdCopyImageToBuffer(cmdbuf, m_image->vulkan_image(),
                           VK_IMAGE_LAYOUT_GENERAL, m_buffer->vulkan_buffer(),
                           1, &region);
}

void cvk_command_buffer_image_copy::build_inner_buffer_to_image(
    cvk_command_buffer& cmdbuf, const VkBufferImageCopy& region) {
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

    vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0,              // dependencyFlags
                         0,              // memoryBarrierCount
                         nullptr,        // pMemoryBarriers
                         1,              // bufferMemoryBarrierCount
                         &bufferBarrier, // pBufferMemoryBarriers
                         0,              // imageMemoryBarrierCount
                         nullptr);       // pImageMemoryBarriers

    vkCmdCopyBufferToImage(cmdbuf, m_buffer->vulkan_buffer(),
                           m_image->vulkan_image(), VK_IMAGE_LAYOUT_GENERAL, 1,
                           &region);
}

cl_int cvk_command_buffer_image_copy::build_batchable_inner(
    cvk_command_buffer& cmdbuf) {

    VkBufferImageCopy region =
        prepare_buffer_image_copy(m_image, m_offset, m_origin, m_region);

    switch (type()) {
    case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
    case CL_COMMAND_MAP_IMAGE:
        build_inner_image_to_buffer(cmdbuf, region);
        break;
    case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
    case CL_COMMAND_UNMAP_MEM_OBJECT:
        build_inner_buffer_to_image(cmdbuf, region);
        break;
    default:
        CVK_ASSERT(false);
        break;
    }

    VkMemoryBarrier memoryBarrier = {
        VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT};

    vkCmdPipelineBarrier(
        cmdbuf, VK_PIPELINE_STAGE_TRANSFER_BIT,
        // TODO HOST only when the dest buffer is an image mapping buffer
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        0, // dependencyFlags
        1, // memoryBarrierCount
        &memoryBarrier,
        0,        // bufferMemoryBarrierCount
        nullptr,  // pBufferMemoryBarriers
        0,        // imageMemoryBarrierCount
        nullptr); // pImageMemoryBarriers

    return CL_SUCCESS;
}

cl_int cvk_command_image_image_copy::build_batchable_inner(
    cvk_command_buffer& cmdbuf) {

    VkImageSubresourceLayers srcSubresource =
        prepare_subresource(m_src_image, m_src_origin, m_region);

    VkOffset3D srcOffset = prepare_offset(m_src_image, m_src_origin);

    VkImageSubresourceLayers dstSubresource =
        prepare_subresource(m_dst_image, m_dst_origin, m_region);

    VkOffset3D dstOffset = prepare_offset(m_dst_image, m_dst_origin);

    VkExtent3D extent = prepare_extent(m_src_image, m_region);

    VkImageCopy region = {srcSubresource, srcOffset, dstSubresource, dstOffset,
                          extent};

    vkCmdCopyImage(cmdbuf, m_src_image->vulkan_image(), VK_IMAGE_LAYOUT_GENERAL,
                   m_dst_image->vulkan_image(), VK_IMAGE_LAYOUT_GENERAL, 1,
                   &region);

    return CL_SUCCESS;
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
