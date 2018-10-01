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


std::unique_ptr<cvk_mem> cvk_mem::create(cvk_context *context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret
){
    auto mem = std::make_unique<cvk_mem>(context, flags, size, host_ptr, nullptr, 0);

    if (!mem->init()) {
        *errcode_ret = CL_OUT_OF_RESOURCES;
        return nullptr;
    }

    *errcode_ret = CL_SUCCESS;
    return mem;
}

bool cvk_mem::init()
{
    auto device = m_context->device();
    auto vkdev = device->vulkan_device();

    // Create the buffer
    const VkBufferCreateInfo createInfo = {
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, // sType
        nullptr, // pNext
        0, // flags
        m_size,
        cvk_mem::USAGE_FLAGS, // usage
        VK_SHARING_MODE_EXCLUSIVE,
        0, // queueFamilyIndexCount
        nullptr, // pQueueFamilyIndices
    };

    VkResult res = vkCreateBuffer(vkdev, &createInfo, nullptr, &m_buffer);

    if (res != VK_SUCCESS) {
        return false;
    }

    cvk_debug_fn("created Vk buffer handle = %p", m_buffer);

    // Get memory requirements
    VkMemoryRequirements memreqs;
    vkGetBufferMemoryRequirements(vkdev, m_buffer, &memreqs);

    // Select memory type
    uint32_t memoryTypeIndex = device->memory_type_index();

    if (memoryTypeIndex == VK_MAX_MEMORY_TYPES) {
        return false;
    }

    // Check against the memory requirements
    // TODO get the type index from the requirements
    if (!((1 << memoryTypeIndex) & memreqs.memoryTypeBits)) {
        return false;
    }

    // Allocate memory
    const VkMemoryAllocateInfo memoryAllocateInfo = {
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        nullptr,
        memreqs.size,
        memoryTypeIndex,
    };

    res = vkAllocateMemory(vkdev, &memoryAllocateInfo, 0, &m_memory);

    if (res != VK_SUCCESS) {
        return false;
    }

    // Bind the buffer to memory
    res = vkBindBufferMemory(vkdev, m_buffer, m_memory, 0);
    
    if (res != VK_SUCCESS) {
        return false;
    }

    if (has_any_flag(CL_MEM_COPY_HOST_PTR | CL_MEM_USE_HOST_PTR)) {
        if (!copy_from(m_host_ptr, 0, m_size)) {
            return false;
        }
    }

    return true;
}



cvk_mem* cvk_mem::create_subbuffer(cl_mem_flags flags, size_t origin, size_t size)
{
    std::unique_ptr<cvk_mem> mem(new cvk_mem(m_context, flags, size, nullptr, this, origin));

    if (!mem->init_subbuffer()) {
        return nullptr;
    }

    return mem.release();
}

bool cvk_mem::init_subbuffer() {

    // Create the buffer
    const VkBufferCreateInfo createInfo = {
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, // sType
        nullptr, // pNext
        0, // flags
        m_size,
        cvk_mem::USAGE_FLAGS, // usage
        VK_SHARING_MODE_EXCLUSIVE,
        0, // queueFamilyIndexCount
        nullptr, // pQueueFamilyIndices
    };

    auto vkdev = m_context->device()->vulkan_device();
    VkResult res = vkCreateBuffer(vkdev, &createInfo, nullptr, &m_buffer);

    if (res != VK_SUCCESS) {
        return false;
    }

    cvk_debug_fn("created Vk buffer handle = %p", m_buffer);

    // Get memory requirements
    VkMemoryRequirements memreqs;
    vkGetBufferMemoryRequirements(vkdev, m_buffer, &memreqs);

    if (m_size != memreqs.size) {
        cvk_warn_fn("Sub-buffer %p requires more memory than its size, you're on your own!", this);
    }

    if (m_parent_offset % memreqs.alignment != 0) {
        cvk_warn_fn("Sub-buffer %p offset (%zu) does not satisfy the alignment "
                    "requirements (%lu) of the Vulkan implementation, "
                    "you're on your own!", this, m_parent_offset, memreqs.alignment);
    }

    // Bind the buffer to memory
    res = vkBindBufferMemory(vkdev, m_buffer, m_parent->m_memory, m_parent_offset);

    if(res != VK_SUCCESS) {
        return false;
    }

    return true;
}
