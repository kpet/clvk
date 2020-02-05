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

bool cvk_mem::map() {
    std::lock_guard<std::mutex> lock(m_map_lock);
    cvk_debug("%p::map", this);

    if (m_parent != nullptr) {
        if (!m_parent->map()) {
            return false;
        }
        m_map_ptr = pointer_offset(m_parent->host_va(), m_parent_offset);
        cvk_debug("%p::map, sub-buffer, map_ptr = %p", this, m_map_ptr);
    } else {
        if (m_map_count == 0) {
            auto vkdev = m_context->device()->vulkan_device();
            VkResult res =
                vkMapMemory(vkdev, m_memory, 0, m_size, 0, &m_map_ptr);
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

void cvk_mem::unmap() {
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

std::unique_ptr<cvk_buffer> cvk_buffer::create(cvk_context* context,
                                               cl_mem_flags flags, size_t size,
                                               void* host_ptr,
                                               cl_int* errcode_ret) {
    auto buffer = std::make_unique<cvk_buffer>(context, flags, size, host_ptr,
                                               nullptr, 0);

    if (!buffer->init()) {
        *errcode_ret = CL_OUT_OF_RESOURCES;
        return nullptr;
    }

    *errcode_ret = CL_SUCCESS;
    return buffer;
}

bool cvk_buffer::init() {
    auto device = m_context->device();
    auto vkdev = device->vulkan_device();

    // Create the buffer
    const VkBufferCreateInfo createInfo = {
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, // sType
        nullptr,                              // pNext
        0,                                    // flags
        m_size,
        cvk_buffer::USAGE_FLAGS, // usage
        VK_SHARING_MODE_EXCLUSIVE,
        0,       // queueFamilyIndexCount
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
    uint32_t memoryTypeIndex = device->memory_type_index_for_buffer(memreqs.memoryTypeBits);

    if (memoryTypeIndex == VK_MAX_MEMORY_TYPES) {
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

cvk_mem* cvk_buffer::create_subbuffer(cl_mem_flags flags, size_t origin,
                                      size_t size) {
    auto buffer = std::make_unique<cvk_buffer>(m_context, flags, size, nullptr,
                                               this, origin);

    if (!buffer->init_subbuffer()) {
        return nullptr;
    }

    return buffer.release();
}

bool cvk_buffer::init_subbuffer() {

    // Create the buffer
    const VkBufferCreateInfo createInfo = {
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, // sType
        nullptr,                              // pNext
        0,                                    // flags
        m_size,
        cvk_buffer::USAGE_FLAGS, // usage
        VK_SHARING_MODE_EXCLUSIVE,
        0,       // queueFamilyIndexCount
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
        cvk_warn_fn(
            "Sub-buffer %p requires more memory (%lu) than its size (%zu), "
            "you're on your own!",
            this, memreqs.size, m_size);
    }

    if (m_parent_offset % memreqs.alignment != 0) {
        cvk_warn_fn("Sub-buffer %p offset (%zu) does not satisfy the alignment "
                    "requirements (%lu) of the Vulkan implementation, "
                    "you're on your own!",
                    this, m_parent_offset, memreqs.alignment);
    }

    // Bind the buffer to memory
    cvk_mem* parent = m_parent;
    auto parent_buffer = static_cast<cvk_buffer*>(parent);
    res = vkBindBufferMemory(vkdev, m_buffer, parent_buffer->m_memory,
                             m_parent_offset);

    if (res != VK_SUCCESS) {
        return false;
    }

    return true;
}

cvk_sampler* cvk_sampler::create(cvk_context* context, bool normalized_coords,
                                 cl_addressing_mode addressing_mode,
                                 cl_filter_mode filter_mode) {
    auto sampler = std::make_unique<cvk_sampler>(context, normalized_coords,
                                                 addressing_mode, filter_mode);

    if (!sampler->init()) {
        return nullptr;
    }

    return sampler.release();
}

bool cvk_sampler::init() {
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
        nullptr,                               // pNext
        0,                                     // flags
        filter,                                // magFilter
        filter,                                // minFilter
        mipmap_mode,                           // mipmapMode
        address_mode,                          // addressModeU
        address_mode,                          // addressModeV
        address_mode,                          // addressModeW
        0.0f,                                  // mipLodBias
        VK_FALSE,                              // anisotropyEnable
        0.0f,                                  // maxAnisotropy
        VK_FALSE,                              // compareEnable
        VK_COMPARE_OP_NEVER,                   // compareOp
        0.0f,                                  // minLod
        0.0f,                                  // maxLod
        VK_BORDER_COLOR_INT_TRANSPARENT_BLACK, // borderColor
        unnormalized_coordinates,              // unnormalizedCoordinates
    };

    auto res = vkCreateSampler(vkdev, &create_info, nullptr, &m_sampler);

    return (res == VK_SUCCESS);
}

cvk_image* cvk_image::create(cvk_context* ctx, cl_mem_flags flags,
                             const cl_image_desc* desc,
                             const cl_image_format* format, void* host_ptr) {
    auto image =
        std::make_unique<cvk_image>(ctx, flags, desc, format, host_ptr);

    if (!image->init()) {
        return nullptr;
    }

    return image.release();
}

extern bool cl_image_format_to_vulkan_format(const cl_image_format& clfmt,
                                             VkFormat& format);

bool cvk_image::init() {
    // Translate image type and size
    VkImageType image_type;
    VkImageViewType view_type;
    VkExtent3D extent;

    extent.width = m_desc.image_width;
    extent.height = m_desc.image_height;
    extent.depth = m_desc.image_depth;

    switch (m_desc.image_type) {
    case CL_MEM_OBJECT_IMAGE1D:
        image_type = VK_IMAGE_TYPE_1D;
        view_type = VK_IMAGE_VIEW_TYPE_1D;
        extent.height = 1;
        extent.depth = 1;
        break;
    case CL_MEM_OBJECT_IMAGE1D_ARRAY:
        image_type = VK_IMAGE_TYPE_1D;
        view_type = VK_IMAGE_VIEW_TYPE_1D_ARRAY;
        extent.height = 1;
        extent.depth = 1;
        break;
    case CL_MEM_OBJECT_IMAGE2D:
        image_type = VK_IMAGE_TYPE_2D;
        view_type = VK_IMAGE_VIEW_TYPE_2D;
        extent.depth = 1;
        break;
    case CL_MEM_OBJECT_IMAGE2D_ARRAY:
        image_type = VK_IMAGE_TYPE_2D;
        view_type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        extent.depth = 1;
        break;
    case CL_MEM_OBJECT_IMAGE3D:
        image_type = VK_IMAGE_TYPE_3D;
        view_type = VK_IMAGE_VIEW_TYPE_3D;
        break;
    case CL_MEM_OBJECT_IMAGE1D_BUFFER: // TODO support that
    default:
        CVK_ASSERT(false);
        image_type = VK_IMAGE_TYPE_MAX_ENUM;
        view_type = VK_IMAGE_VIEW_TYPE_MAX_ENUM;
        break;
    }

    uint32_t array_layers = 1;
    if ((m_desc.image_type == CL_MEM_OBJECT_IMAGE1D_ARRAY) ||
        (m_desc.image_type == CL_MEM_OBJECT_IMAGE2D_ARRAY)) {
        array_layers = m_desc.image_array_size;
    }

    // Translate format
    VkFormat format;
    auto success = cl_image_format_to_vulkan_format(m_format, format);
    if (!success) {
        return false; // TODO error code
    }

    // Create Image
    VkImageCreateInfo imageCreateInfo = {
        VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        nullptr,                 // pNext
        0,                       // flags
        image_type,              // imageType
        format,                  // format
        extent,                  // extent
        1,                       // mipLevels
        array_layers,            // arrayLayers
        VK_SAMPLE_COUNT_1_BIT,   // samples
        VK_IMAGE_TILING_OPTIMAL, // tiling
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_SHARING_MODE_EXCLUSIVE, // sharingMode
        0,                         // queueFamilyIndexCount
        nullptr,                   // pQueueFamilyIndices
        VK_IMAGE_LAYOUT_UNDEFINED, // initialLayout
    };

    auto device = m_context->device();
    auto vkdev = device->vulkan_device();

    auto res = vkCreateImage(vkdev, &imageCreateInfo, nullptr, &m_image);
    if (res != VK_SUCCESS) {
        cvk_error_fn("Could not create image!");
        return false;
    }

    // Get memory requirements
    VkMemoryRequirements memreqs;
    vkGetImageMemoryRequirements(vkdev, m_image, &memreqs);
    cvk_debug_fn("Required memory type bits: %x", memreqs.memoryTypeBits);

    // Select memory type
    uint32_t memoryTypeIndex =
        device->memory_type_index_for_image(memreqs.memoryTypeBits);

    if (memoryTypeIndex == VK_MAX_MEMORY_TYPES) {
        cvk_error_fn("Could not get memory type!");
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
        cvk_error_fn("Could not allocate memory!");
        return false;
    }

    // Bind the buffer to memory
    res = vkBindImageMemory(vkdev, m_image, m_memory, 0);

    if (res != VK_SUCCESS) {
        return false;
    }

    // Create image view
    VkComponentMapping components = {
        VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};

    VkImageSubresourceRange subresource = {
        VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
        0,                         // baseMipLevel
        1,                         // levelCount
        0,                         // baseArrayLayer
        array_layers,              // layerCount
    };

    VkImageViewCreateInfo imageViewCreateInfo = {
        VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        nullptr,     // pNext
        0,           // flags
        m_image,     // image
        view_type,   // viewType;
        format,      // format
        components,  // components
        subresource, // subresourceRange
    };

    res =
        vkCreateImageView(vkdev, &imageViewCreateInfo, nullptr, &m_image_view);

    return res == VK_SUCCESS;
}

void cvk_image::prepare_fill_pattern(const void* input_pattern,
                                     fill_pattern_array& pattern,
                                     size_t* size_ret) const {

    auto pat_float = static_cast<const cl_float*>(input_pattern);
    auto pat_int = static_cast<const cl_int*>(input_pattern);
    auto pat_uint = static_cast<const cl_uint*>(input_pattern);

    cl_uchar pat_uchar[4] = {
        static_cast<cl_uchar>(pat_uint[0]),
        static_cast<cl_uchar>(pat_uint[1]),
        static_cast<cl_uchar>(pat_uint[2]),
        static_cast<cl_uchar>(pat_uint[3]),
    };
    cl_ushort pat_ushort[4] = {
        static_cast<cl_ushort>(pat_uint[0]),
        static_cast<cl_ushort>(pat_uint[1]),
        static_cast<cl_ushort>(pat_uint[2]),
        static_cast<cl_ushort>(pat_uint[3]),
    };
    cl_char pat_char[4] = {
        static_cast<cl_char>(pat_int[0]),
        static_cast<cl_char>(pat_int[1]),
        static_cast<cl_char>(pat_int[2]),
        static_cast<cl_char>(pat_int[3]),
    };
    cl_short pat_short[4] = {
        static_cast<cl_short>(pat_int[0]),
        static_cast<cl_short>(pat_int[1]),
        static_cast<cl_short>(pat_int[2]),
        static_cast<cl_short>(pat_int[3]),
    };

    size_t size = element_size();
    *size_ret = size;

    cl_uchar pat_unorm_int8[4] = {static_cast<cl_uchar>(pat_float[0] * 255.0f),
                                  static_cast<cl_uchar>(pat_float[1] * 255.0f),
                                  static_cast<cl_uchar>(pat_float[2] * 255.0f),
                                  static_cast<cl_uchar>(pat_float[3] * 255.0f)};
    cl_uchar pat_snorm_int8[4] = {static_cast<cl_uchar>(pat_float[0] * 127.0f),
                                  static_cast<cl_uchar>(pat_float[1] * 127.0f),
                                  static_cast<cl_uchar>(pat_float[2] * 127.0f),
                                  static_cast<cl_uchar>(pat_float[3] * 127.0f)};
    cl_ushort pat_unorm_int16[4] = {
        static_cast<cl_ushort>(pat_float[0] * 65535.0f),
        static_cast<cl_ushort>(pat_float[1] * 65535.0f),
        static_cast<cl_ushort>(pat_float[2] * 65535.0f),
        static_cast<cl_ushort>(pat_float[3] * 65535.0f)};
    cl_ushort pat_snorm_int16[4] = {
        static_cast<cl_ushort>(pat_float[0] * 32767.0f),
        static_cast<cl_ushort>(pat_float[1] * 32767.0f),
        static_cast<cl_ushort>(pat_float[2] * 32767.0f),
        static_cast<cl_ushort>(pat_float[3] * 32767.0f)};

    const void* cast_pattern = nullptr;
    switch (format().image_channel_data_type) {
    case CL_UNSIGNED_INT8:
        cast_pattern = &pat_uchar;
        break;
    case CL_UNSIGNED_INT16:
        cast_pattern = &pat_ushort;
        break;
    case CL_SIGNED_INT8:
        cast_pattern = &pat_char;
        break;
    case CL_SIGNED_INT16:
        cast_pattern = &pat_short;
        break;
    case CL_FLOAT:
    case CL_UNSIGNED_INT32:
    case CL_SIGNED_INT32:
        cast_pattern = input_pattern;
        break;
    case CL_UNORM_INT8:
        cast_pattern = pat_unorm_int8;
        break;
    case CL_UNORM_INT16:
        cast_pattern = pat_unorm_int16;
        break;
    case CL_SNORM_INT8:
        cast_pattern = pat_snorm_int8;
        break;
    case CL_SNORM_INT16:
        cast_pattern = pat_snorm_int16;
        break;
    case CL_HALF_FLOAT: // FIXME
    default:
        CVK_ASSERT(false);
        return;
    }

    size_t csize = element_size_per_channel();

    switch (format().image_channel_order) {
    case CL_R:
    case CL_RG:
    case CL_RGBA:
        memcpy(pattern.data(), cast_pattern, size);
        break;
    case CL_BGRA:
        memcpy(pattern.data() + 0 * csize,
               pointer_offset(cast_pattern, 2 * csize), csize);
        memcpy(pattern.data() + 1 * csize,
               pointer_offset(cast_pattern, 1 * csize), csize);
        memcpy(pattern.data() + 2 * csize,
               pointer_offset(cast_pattern, 0 * csize), csize);
        memcpy(pattern.data() + 3 * csize,
               pointer_offset(cast_pattern, 3 * csize), csize);
        break;
    default:
        CVK_ASSERT(false);
    }
}
