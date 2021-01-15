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

bool cvk_mem::map() {
    std::lock_guard<std::mutex> lock(m_map_lock);
    cvk_debug("%p::map", this);

    if (m_map_count == 0) {
        if (m_parent != nullptr) {
            if (!m_parent->map()) {
                return false;
            }
            m_map_ptr = pointer_offset(m_parent->host_va(), m_parent_offset);
            cvk_debug("%p::map, sub-buffer, map_ptr = %p", this, m_map_ptr);
        } else {
            auto res = m_memory->map(&m_map_ptr);
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
    if (m_map_count == 0) {
        if (m_parent != nullptr) {
            m_parent->unmap();
            cvk_debug("%p::unmap, sub-buffer", this);
        } else {
            m_memory->unmap();
            m_map_ptr = nullptr;
        }
    }
    cvk_debug("%p::unmap, new map_count = %u", this, m_map_count);
}

std::unique_ptr<cvk_buffer>
cvk_buffer::create(cvk_context* context, cl_mem_flags flags, size_t size,
                   void* host_ptr, std::vector<cl_mem_properties>&& properties,
                   cl_int* errcode_ret) {
    auto buffer = std::make_unique<cvk_buffer>(
        context, flags, size, host_ptr, nullptr, 0, std::move(properties));

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

    // Select memory type
    cvk_device::allocation_parameters params =
        device->select_memory_for(m_buffer, flags());
    if (params.memory_type_index == VK_MAX_MEMORY_TYPES) {
        return false;
    }

    // Allocate memory
    m_memory = std::make_unique<cvk_memory_allocation>(
        vkdev, params.size, params.memory_type_index);
    res = m_memory->allocate();

    if (res != VK_SUCCESS) {
        return false;
    }

    // Bind the buffer to memory
    res = vkBindBufferMemory(vkdev, m_buffer, m_memory->vulkan_memory(), 0);

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
    std::vector<cl_mem_properties> properties;
    auto buffer = std::make_unique<cvk_buffer>(
        m_context, flags, size, nullptr, this, origin, std::move(properties));

    return buffer.release();
}

cvk_sampler*
cvk_sampler::create(cvk_context* context, bool normalized_coords,
                    cl_addressing_mode addressing_mode,
                    cl_filter_mode filter_mode,
                    std::vector<cl_sampler_properties>&& properties) {
    auto sampler = std::make_unique<cvk_sampler>(context, normalized_coords,
                                                 addressing_mode, filter_mode,
                                                 std::move(properties));

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
                             const cl_image_format* format, void* host_ptr,
                             std::vector<cl_mem_properties>&& properties) {
    auto image = std::make_unique<cvk_image>(ctx, flags, desc, format, host_ptr,
                                             std::move(properties));

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

    m_extent.width = m_desc.image_width;
    m_extent.height = m_desc.image_height;
    m_extent.depth = m_desc.image_depth;

    uint32_t array_layers = 1;
    if ((m_desc.image_type == CL_MEM_OBJECT_IMAGE1D_ARRAY) ||
        (m_desc.image_type == CL_MEM_OBJECT_IMAGE2D_ARRAY)) {
        array_layers = m_desc.image_array_size;
    }

    uint32_t row_pitch = m_desc.image_row_pitch;
    if (row_pitch == 0) {
        row_pitch = m_extent.width * element_size();
    }
    uint32_t slice_pitch = m_desc.image_slice_pitch;
    if (slice_pitch == 0) {
        slice_pitch = row_pitch * m_extent.height;
    }

    size_t host_ptr_size = 0;

    switch (m_desc.image_type) {
    case CL_MEM_OBJECT_IMAGE1D:
        image_type = VK_IMAGE_TYPE_1D;
        view_type = VK_IMAGE_VIEW_TYPE_1D;
        m_extent.height = 1;
        m_extent.depth = 1;
        host_ptr_size = row_pitch;
        break;
    case CL_MEM_OBJECT_IMAGE1D_ARRAY:
        image_type = VK_IMAGE_TYPE_1D;
        view_type = VK_IMAGE_VIEW_TYPE_1D_ARRAY;
        m_extent.height = 1;
        m_extent.depth = 1;
        host_ptr_size = row_pitch * array_layers;
        break;
    case CL_MEM_OBJECT_IMAGE2D:
        image_type = VK_IMAGE_TYPE_2D;
        view_type = VK_IMAGE_VIEW_TYPE_2D;
        m_extent.depth = 1;
        host_ptr_size = slice_pitch;
        break;
    case CL_MEM_OBJECT_IMAGE2D_ARRAY:
        image_type = VK_IMAGE_TYPE_2D;
        view_type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        m_extent.depth = 1;
        host_ptr_size = slice_pitch * array_layers;
        break;
    case CL_MEM_OBJECT_IMAGE3D:
        image_type = VK_IMAGE_TYPE_3D;
        view_type = VK_IMAGE_VIEW_TYPE_3D;
        host_ptr_size = slice_pitch * m_extent.depth;
        break;
    case CL_MEM_OBJECT_IMAGE1D_BUFFER: // TODO support that
    default:
        CVK_ASSERT(false);
        image_type = VK_IMAGE_TYPE_MAX_ENUM;
        view_type = VK_IMAGE_VIEW_TYPE_MAX_ENUM;
        break;
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
        m_extent,                // extent
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

    // Select memory type
    cvk_device::allocation_parameters params =
        device->select_memory_for(m_image);
    if (params.memory_type_index == VK_MAX_MEMORY_TYPES) {
        cvk_error_fn("Could not get memory type!");
        return false;
    }

    // Allocate memory
    m_memory = std::make_unique<cvk_memory_allocation>(
        vkdev, params.size, params.memory_type_index);

    res = m_memory->allocate();

    if (res != VK_SUCCESS) {
        cvk_error_fn("Could not allocate memory!");
        return false;
    }

    // Bind the image to memory
    res = vkBindImageMemory(vkdev, m_image, m_memory->vulkan_memory(), 0);

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

    if (res != VK_SUCCESS) {
        return false;
    }

    if (has_any_flag(CL_MEM_COPY_HOST_PTR | CL_MEM_USE_HOST_PTR)) {
        // Create a staging buffer to copy to the device later.
        cl_int ret;
        m_init_data = cvk_buffer::create(m_context, CL_MEM_READ_ONLY,
                                         host_ptr_size, nullptr, &ret);
        if (ret != CL_SUCCESS) {
            cvk_error("Could not create staging buffer for image host_ptr");
            return false;
        }

        if (!m_init_data->map()) {
            cvk_error("Could not map staging buffer");
            return false;
        }

        // Copy data to staging buffer.
        void* dst = m_init_data->map_ptr(0);
        memcpy(dst, m_host_ptr, host_ptr_size);
        m_init_data->unmap();
    }

    return true;
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
    cl_half pat_half[4] = {
        cl_half_from_float(pat_float[0], CL_HALF_RTE),
        cl_half_from_float(pat_float[1], CL_HALF_RTE),
        cl_half_from_float(pat_float[2], CL_HALF_RTE),
        cl_half_from_float(pat_float[3], CL_HALF_RTE),
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
    case CL_HALF_FLOAT:
        cast_pattern = pat_half;
        break;
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

bool cvk_image::prepare_for_device(cvk_command_queue& queue) {
    std::lock_guard<std::mutex> lock(m_device_init_lock);

    // Check if we have already initialized image on the device.
    if (m_device_initialized) {
        return true;
    }

    cvk_info("Preparing image %p for use on device", this);

    // Create a command buffer and begin recording commands.
    cvk_command_buffer command_buffer(&queue);
    if (!command_buffer.begin()) {
        cvk_error("Could not create command buffer for image initialization");
        return false;
    }

    bool needs_copy = m_init_data != nullptr;

    // Transition image layout to GENERAL or TRANSFER_DST_OPTIMAL.
    VkImageSubresourceRange subresourceRange = {
        VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
        0,                         // baseMipLevel
        VK_REMAINING_MIP_LEVELS,   // levelCount
        0,                         // baseArrayLayer
        VK_REMAINING_ARRAY_LAYERS, // layerCount
    };
    VkImageLayout layout = needs_copy ? VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                                      : VK_IMAGE_LAYOUT_GENERAL;
    VkImageMemoryBarrier imageBarrier = {
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        nullptr,
        0,                                                      // srcAccessMask
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, // dstAccessMask
        VK_IMAGE_LAYOUT_UNDEFINED,                              // oldLayout
        layout,                                                 // newLayout
        0,                // srcQueueFamilyIndex
        0,                // dstQueueFamilyIndex
        vulkan_image(),   // image
        subresourceRange, // subresourceRange
    };
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0,              // dependencyFlags
                         0,              // memoryBarrierCount
                         nullptr,        // pMemoryBarriers
                         0,              // bufferMemoryBarrierCount
                         nullptr,        // pBufferMemoryBarriers
                         1,              // imageMemoryBarrierCount
                         &imageBarrier); // pImageMemoryBarriers

    // Set up a buffer->image copy to initialize the image contents.
    if (needs_copy) {
        uint32_t row_length = m_desc.image_row_pitch
                                  ? m_desc.image_row_pitch / element_size()
                                  : m_extent.width;
        uint32_t image_height =
            m_desc.image_slice_pitch
                ? m_desc.image_slice_pitch / row_length / element_size()
                : m_extent.height;
        uint32_t layer_count = 1;
        if ((m_desc.image_type == CL_MEM_OBJECT_IMAGE1D_ARRAY) ||
            (m_desc.image_type == CL_MEM_OBJECT_IMAGE2D_ARRAY)) {
            layer_count = m_desc.image_array_size;
        }
        VkImageSubresourceLayers subresource = {
            VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
            0,                         // mipLevel
            0,                         // baseArrayLayer
            layer_count,               // layerCount
        };
        VkBufferImageCopy copy = {
            0,            // bufferOffset
            row_length,   // bufferRowLength
            image_height, // bufferImageHeight
            subresource,  // imageSubresource
            {0, 0, 0},    // imageOffset
            m_extent,     // imageExtent
        };
        vkCmdCopyBufferToImage(command_buffer, m_init_data->vulkan_buffer(),
                               vulkan_image(),
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

        // Transition image layout to GENERAL.
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
            VK_ACCESS_TRANSFER_WRITE_BIT,         // srcAccessMask
            VK_ACCESS_MEMORY_READ_BIT,            // dstAccessMask
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, // oldLayout
            VK_IMAGE_LAYOUT_GENERAL,              // newLayout
            0,                                    // srcQueueFamilyIndex
            0,                                    // dstQueueFamilyIndex
            vulkan_image(),                       // image
            subresourceRange,                     // subresourceRange
        };
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0,              // dependencyFlags
                             0,              // memoryBarrierCount
                             nullptr,        // pMemoryBarriers
                             0,              // bufferMemoryBarrierCount
                             nullptr,        // pBufferMemoryBarriers
                             1,              // imageMemoryBarrierCount
                             &imageBarrier); // pImageMemoryBarriers
    }

    if (!command_buffer.end()) {
        cvk_error("Could not end image initialization command buffer");
        return false;
    }

    // Submit commands and wait for completion.
    if (!command_buffer.submit_and_wait()) {
        cvk_error("Could not execute image initialization commands");
        return false;
    }

    m_init_data.reset();

    m_device_initialized = true;

    return true;
}
