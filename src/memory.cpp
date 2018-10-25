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

bool cvk_mem::map()
{
    std::lock_guard<std::mutex> lock(m_map_lock);
    cvk_debug("%p::map", this);

    if (m_parent != nullptr) {
        if (!m_parent->map()) {
            return false;
        }
        auto parent_host_va = reinterpret_cast<uintptr_t>(m_parent->host_va());
        m_map_ptr = reinterpret_cast<void*>(parent_host_va + m_parent_offset);
        cvk_debug("%p::map, sub-buffer, map_ptr = %p", this, m_map_ptr);
    } else {
        if (m_map_count == 0) {
            auto vkdev = m_context->device()->vulkan_device();
            VkResult res = vkMapMemory(vkdev, m_memory, 0, m_size, 0, &m_map_ptr);
            if (res != VK_SUCCESS) {
                return false;
            }
            cvk_debug("%p::map, map_ptr = %p", this, m_map_ptr);
        }
    }

    m_map_count++;
    retain();
    cvk_debug("%p::map, new map_count = %u", this, m_map_count);

    return true;
}

void cvk_mem::unmap()
{
    std::lock_guard<std::mutex> lock(m_map_lock);
    cvk_debug("%p::unmap", this);

    CVK_ASSERT(m_map_count > 0);
    m_map_count--;
    release();
    if (m_parent != nullptr) {
        m_parent->unmap();
        cvk_debug("%p::unmap, sub-buffer", this);
    } else {
        if (m_map_count == 0) {
            auto vkdev = m_context->device()->vulkan_device();
            vkUnmapMemory(vkdev, m_memory);
        }
    }
    cvk_debug("%p::unmap, new map_count = %u", this, m_map_count);
}

std::unique_ptr<cvk_buffer> cvk_buffer::create(cvk_context *context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret
){
    auto buffer = std::make_unique<cvk_buffer>(context, flags, size, host_ptr, nullptr, 0);

    if (!buffer->init()) {
        *errcode_ret = CL_OUT_OF_RESOURCES;
        return nullptr;
    }

    *errcode_ret = CL_SUCCESS;
    return buffer;
}

bool cvk_buffer::init()
{
    auto device = m_context->device();
    auto vkdev = device->vulkan_device();

    // Create the buffer
    const VkBufferCreateInfo createInfo = {
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, // sType
        nullptr, // pNext
        0, // flags
        m_size,
        cvk_buffer::USAGE_FLAGS, // usage
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



cvk_mem* cvk_buffer::create_subbuffer(cl_mem_flags flags, size_t origin, size_t size)
{
    auto buffer = std::make_unique<cvk_buffer>(m_context, flags, size, nullptr, this, origin);

    if (!buffer->init_subbuffer()) {
        return nullptr;
    }

    return buffer.release();
}

bool cvk_buffer::init_subbuffer() {

    // Create the buffer
    const VkBufferCreateInfo createInfo = {
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, // sType
        nullptr, // pNext
        0, // flags
        m_size,
        cvk_buffer::USAGE_FLAGS, // usage
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
    auto parent_buffer = static_cast<cvk_buffer*>(m_parent);
    res = vkBindBufferMemory(vkdev, m_buffer, parent_buffer->m_memory, m_parent_offset);

    if(res != VK_SUCCESS) {
        return false;
    }

    return true;
}


cvk_sampler*
cvk_sampler::create(cvk_context *context, bool normalized_coords,
                    cl_addressing_mode addressing_mode, cl_filter_mode filter_mode)
{
    auto sampler = std::make_unique<cvk_sampler>(context, normalized_coords,
                                                 addressing_mode, filter_mode);

    if (!sampler->init()) {
        return nullptr;
    }

    return sampler.release();
}

bool cvk_sampler::init()
{
    auto vkdev = context()->device()->vulkan_device();

    // Translate addressing mode
    VkSamplerAddressMode address_mode;
    switch (m_addressing_mode) {
    default:
    case CL_ADDRESS_NONE:
        address_mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        break;
    case CL_ADDRESS_CLAMP_TO_EDGE:
        address_mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        break;
    case CL_ADDRESS_CLAMP:
        address_mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        break;
    case CL_ADDRESS_REPEAT:
        address_mode = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        break;
    case CL_ADDRESS_MIRRORED_REPEAT:
        address_mode = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        break;
    }

    // Translate filtering
    VkFilter filter;
    VkSamplerMipmapMode mipmap_mode;
    switch (m_filter_mode) {
    default:
    case CL_FILTER_NEAREST:
        filter = VK_FILTER_NEAREST;
        mipmap_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        break;
    case CL_FILTER_LINEAR:
        filter = VK_FILTER_LINEAR;
        mipmap_mode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        break;
    }

    // Translate coordinate type
    VkBool32 unnormalized_coordinates;
    if (m_normalized_coords) {
        unnormalized_coordinates = VK_FALSE;
    } else {
        unnormalized_coordinates = VK_TRUE;
    }

    // TODO this is a rough first pass, dig into the details
    const VkSamplerCreateInfo create_info = {
        VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        nullptr, // pNext
        0, // flags
        filter, // magFilter
        filter, // minFilter
        mipmap_mode, // mipmapMode
        address_mode, // addressModeU
        address_mode, // addressModeV
        address_mode, // addressModeW
        0.0f, // mipLodBias
        VK_FALSE, // anisotropyEnable
        0.0f, // maxAnisotropy
        VK_FALSE, // compareEnable
        VK_COMPARE_OP_NEVER, // compareOp
        0.0f, // minLod
        0.0f, // maxLod
        VK_BORDER_COLOR_INT_TRANSPARENT_BLACK, // borderColor
        unnormalized_coordinates, // unnormalizedCoordinates
    };

    auto res = vkCreateSampler(vkdev, &create_info, nullptr, &m_sampler);

    return (res == VK_SUCCESS);
}
