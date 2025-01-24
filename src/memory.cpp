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

#include <cmath>

#include "image_format.hpp"
#include "memory.hpp"
#include "queue.hpp"

bool cvk_mem::map_memory() {
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

void cvk_mem::unmap_memory() {
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

void cvk_mem::invalidate_memory(VkDeviceSize offset, VkDeviceSize size) {
    if (m_parent != nullptr) {
        m_parent->invalidate_memory(offset + m_parent_offset, size);
    } else {
        m_memory->invalidate(offset, size);
    }
}

void cvk_mem::flush_memory(VkDeviceSize offset, VkDeviceSize size) {
    if (m_parent != nullptr) {
        m_parent->flush_memory(offset + m_parent_offset, size);
    } else {
        m_memory->flush(offset, size);
    }
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
        prepare_usage_flags(), // usage
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
    m_memory = std::make_shared<cvk_memory_allocation>(
        vkdev, params.size, params.memory_type_index, params.memory_coherent);
    res = m_memory->allocate(device->uses_physical_addressing());

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

bool cvk_sampler::init(bool force_normalized_coordinates) {
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
    case CL_FILTER_LINEAR:
        if (config.supports_filter_linear) {
            filter = VK_FILTER_LINEAR;
            mipmap_mode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            break;
        } else {
            cvk_warn_fn(
                "linear filter is not supported, using nearest filter instead");
        }
        [[fallthrough]];
    default:
    case CL_FILTER_NEAREST:
        filter = VK_FILTER_NEAREST;
        mipmap_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        break;
    }

    // Translate coordinate type
    VkBool32 unnormalized_coordinates;
    if (m_normalized_coords || force_normalized_coordinates) {
        unnormalized_coordinates = VK_FALSE;
    } else {
        unnormalized_coordinates = VK_TRUE;
        // VUID-01073: unnormalized coords must use nearest mipmap filtering.
        mipmap_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        // VUID-01075: unnormalized coords must use clamp to edge or border
        // addressing
        if ((address_mode != VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE) &&
            (address_mode != VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER)) {
            address_mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        }
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

    VkSampler* sampler =
        force_normalized_coordinates ? &m_sampler_norm : &m_sampler;
    auto res = vkCreateSampler(vkdev, &create_info, nullptr, sampler);

    return (res == VK_SUCCESS);
}

VkFormatFeatureFlags
cvk_image::required_format_feature_flags_for(cl_mem_object_type type,
                                             cl_mem_flags flags) {
    // 1Dbuffer requires
    //  RW / RaW: STORAGE_TEXEL_BUFFER
    //  RO: UNIFORM_TEXEL_BUFFER
    // All other images require TRANSFER_SRC, TRANSFER_DST
    //  read-only: SAMPLED_IMAGE, SAMPLED_IMAGE_FILTER_LINEAR
    //  write-only: STORAGE_IMAGE
    //  read-write: STORAGE_IMAGE, SAMPLED_IMAGE, SAMPLED_IMAGE_FILTER_LINEAR
    //  read-and-write: STORAGE_IMAGE
    VkFormatFeatureFlags format_feature_flags = 0;
    if (type != CL_MEM_OBJECT_IMAGE1D_BUFFER) {
        format_feature_flags = VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                               VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    }
    VkFormatFeatureFlags format_feature_flags_RO;
    if (type == CL_MEM_OBJECT_IMAGE1D_BUFFER) {
        format_feature_flags_RO = VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT;
    } else {
        format_feature_flags_RO = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        if (config.supports_filter_linear()) {
            format_feature_flags_RO |=
                VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
        }
    }
    VkFormatFeatureFlags format_feature_flags_WO;
    if (type == CL_MEM_OBJECT_IMAGE1D_BUFFER) {
        format_feature_flags_WO = VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT;
    } else {
        format_feature_flags_WO = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
    }

    if (flags & (CL_MEM_KERNEL_READ_AND_WRITE | CL_MEM_WRITE_ONLY)) {
        format_feature_flags |= format_feature_flags_WO;
    } else if (flags & CL_MEM_READ_ONLY) {
        format_feature_flags |= format_feature_flags_RO;
    } else {
        format_feature_flags |=
            format_feature_flags_RO | format_feature_flags_WO;
    }

    return format_feature_flags;
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

bool cvk_image::init_vulkan_image() {
    // Translate image type and size
    VkImageType image_type;
    VkImageViewType view_type;
    VkExtent3D extent;

    extent.width = m_desc.image_width;
    extent.height = m_desc.image_height;
    extent.depth = m_desc.image_depth;

    uint32_t array_layers = 1;
    if ((m_desc.image_type == CL_MEM_OBJECT_IMAGE1D_ARRAY) ||
        (m_desc.image_type == CL_MEM_OBJECT_IMAGE2D_ARRAY)) {
        array_layers = m_desc.image_array_size;
    }

    uint32_t row_pitch = m_desc.image_row_pitch;
    if (row_pitch == 0) {
        row_pitch = m_desc.image_width * element_size();
    }
    uint32_t slice_pitch = m_desc.image_slice_pitch;
    if (slice_pitch == 0) {
        slice_pitch = row_pitch * m_desc.image_height;
    }

    size_t host_ptr_size = 0;

    switch (m_desc.image_type) {
    case CL_MEM_OBJECT_IMAGE1D:
        image_type = VK_IMAGE_TYPE_1D;
        view_type = VK_IMAGE_VIEW_TYPE_1D;
        extent.height = 1;
        extent.depth = 1;
        host_ptr_size = row_pitch;
        break;
    case CL_MEM_OBJECT_IMAGE1D_ARRAY:
        image_type = VK_IMAGE_TYPE_1D;
        view_type = VK_IMAGE_VIEW_TYPE_1D_ARRAY;
        extent.height = 1;
        extent.depth = 1;
        host_ptr_size = row_pitch * array_layers;
        break;
    case CL_MEM_OBJECT_IMAGE2D:
        image_type = VK_IMAGE_TYPE_2D;
        view_type = VK_IMAGE_VIEW_TYPE_2D;
        extent.depth = 1;
        host_ptr_size = slice_pitch;
        break;
    case CL_MEM_OBJECT_IMAGE2D_ARRAY:
        image_type = VK_IMAGE_TYPE_2D;
        view_type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        extent.depth = 1;
        host_ptr_size = slice_pitch * array_layers;
        break;
    case CL_MEM_OBJECT_IMAGE3D:
        image_type = VK_IMAGE_TYPE_3D;
        view_type = VK_IMAGE_VIEW_TYPE_3D;
        host_ptr_size = slice_pitch * m_desc.image_depth;
        break;
    default:
        CVK_ASSERT(false);
        image_type = VK_IMAGE_TYPE_MAX_ENUM;
        view_type = VK_IMAGE_VIEW_TYPE_MAX_ENUM;
        break;
    }

    // Translate format
    image_format_support fmt;
    VkComponentMapping components_sampled, components_storage;

    auto device = m_context->device();

    auto success = cl_image_format_to_vulkan_format(
        m_format, m_desc.image_type, device, &fmt, &components_sampled,
        &components_storage);
    if (!success) {
        return false; // TODO error code
    }

    // Create Image
    VkImageCreateInfo imageCreateInfo = {
        VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        nullptr,                   // pNext
        0,                         // flags
        image_type,                // imageType
        fmt.vkfmt,                 // format
        extent,                    // extent
        1,                         // mipLevels
        array_layers,              // arrayLayers
        VK_SAMPLE_COUNT_1_BIT,     // samples
        VK_IMAGE_TILING_OPTIMAL,   // tiling
        prepare_usage_flags(),     // usage
        VK_SHARING_MODE_EXCLUSIVE, // sharingMode
        0,                         // queueFamilyIndexCount
        nullptr,                   // pQueueFamilyIndices
        VK_IMAGE_LAYOUT_UNDEFINED, // initialLayout
    };

    auto vkdev = device->vulkan_device();

    auto res = vkCreateImage(vkdev, &imageCreateInfo, nullptr, &m_image);
    if (res != VK_SUCCESS) {
        cvk_error_fn("Could not create image!");
        return false;
    }

    CVK_ASSERT(m_desc.image_type != CL_MEM_OBJECT_IMAGE1D_BUFFER);
    // Select memory type
    cvk_device::allocation_parameters params =
        device->select_memory_for(m_image);
    if (params.memory_type_index == VK_MAX_MEMORY_TYPES) {
        cvk_error_fn("Could not get memory type!");
        return false;
    }

    // Allocate memory
    m_memory = std::make_unique<cvk_memory_allocation>(
        vkdev, params.size, params.memory_type_index, params.memory_coherent);

    res = m_memory->allocate(device->uses_physical_addressing());

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
    VkImageSubresourceRange subresource = {
        VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
        0,                         // baseMipLevel
        1,                         // levelCount
        0,                         // baseArrayLayer
        array_layers,              // layerCount
    };

    VkImageViewCreateInfo imageViewCreateInfo = {
        VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        nullptr,            // pNext
        0,                  // flags
        m_image,            // image
        view_type,          // viewType;
        fmt.vkfmt,          // format
        components_sampled, // components
        subresource,        // subresourceRange
    };

    res = vkCreateImageView(vkdev, &imageViewCreateInfo, nullptr,
                            &m_sampled_view);

    if (res != VK_SUCCESS) {
        return false;
    }

    imageViewCreateInfo.components = components_storage;

    res = vkCreateImageView(vkdev, &imageViewCreateInfo, nullptr,
                            &m_storage_view);

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

        if (!m_init_data->copy_from(m_host_ptr, 0, host_ptr_size)) {
            cvk_error(
                "Could not copy image host_ptr data to the staging buffer");
            return false;
        }

        if (config.init_image_at_creation()) {
            auto queue = m_context->get_or_create_image_init_command_queue();
            if (queue == nullptr) {
                return false;
            }

            auto initimage = new cvk_command_image_init(queue, this, m_context);
            ret = queue->enqueue_command_with_deps(initimage, 0, nullptr,
                                                   nullptr);
            if (ret != CL_SUCCESS) {
                return false;
            }
            ret = queue->finish();
            if (ret != CL_SUCCESS) {
                return false;
            }
            std::lock_guard<std::mutex> lock(m_init_tracker.mutex());
            m_init_tracker.set_state(cvk_mem_init_state::completed);
        }
    }

    return true;
}

bool cvk_image::init_vulkan_texel_buffer() {
    VkResult res;

    auto device = m_context->device();
    auto vkdev = device->vulkan_device();

    image_format_support fmt;
    VkComponentMapping components_sampled, components_storage;

    auto success = cl_image_format_to_vulkan_format(
        m_format, m_desc.image_type, device, &fmt, &components_sampled,
        &components_storage);
    if (!success) {
        return false;
    }

    CVK_ASSERT(buffer());
    CVK_ASSERT(buffer()->is_buffer_type());

    auto vkbuf = static_cast<cvk_buffer*>(buffer())->vulkan_buffer();
    auto offset = static_cast<cvk_buffer*>(buffer())->vulkan_buffer_offset();
    // The range should cover exactly the number of texels specified at
    // image creation time.  Don't use WHOLE_SIZE because the row pitch
    // might include additional padding large enough for one or more texels.
    auto range = element_size() * width();

    VkBufferViewCreateInfo createInfo = {
        VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
        nullptr,
        0,         // flags
        vkbuf,     // buffer
        fmt.vkfmt, // format
        offset,    // offset
        range      // range
    };

    res = vkCreateBufferView(vkdev, &createInfo, nullptr, &m_buffer_view);
    if (res != VK_SUCCESS) {
        cvk_error_fn("Could not create buffer view");
        return false;
    }

    buffer()->retain();

    return true;
}

bool cvk_image::init() {
    if (is_backed_by_buffer_view()) {
        return init_vulkan_texel_buffer();
    } else {
        return init_vulkan_image();
    }
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

    auto saturate = [](float x, float min, float max) {
        if (std::isnan(x)) {
            return 0.f;
        } else if (x < min) {
            return min;
        } else if (x > max) {
            return max;
        }
        return x;
    };
    cl_uchar pat_unorm_int8[4] = {
        static_cast<cl_uchar>(saturate(pat_float[0], 0.f, 1.f) * 255.0f),
        static_cast<cl_uchar>(saturate(pat_float[1], 0.f, 1.f) * 255.0f),
        static_cast<cl_uchar>(saturate(pat_float[2], 0.f, 1.f) * 255.0f),
        static_cast<cl_uchar>(saturate(pat_float[3], 0.f, 1.f) * 255.0f)};
    cl_char pat_snorm_int8[4] = {
        static_cast<cl_char>(saturate(pat_float[0], -1.f, 1.f) * 127.0f),
        static_cast<cl_char>(saturate(pat_float[1], -1.f, 1.f) * 127.0f),
        static_cast<cl_char>(saturate(pat_float[2], -1.f, 1.f) * 127.0f),
        static_cast<cl_char>(saturate(pat_float[3], -1.f, 1.f) * 127.0f)};
    cl_ushort pat_unorm_int16[4] = {
        static_cast<cl_ushort>(saturate(pat_float[0], 0.f, 1.f) * 65535.0f),
        static_cast<cl_ushort>(saturate(pat_float[1], 0.f, 1.f) * 65535.0f),
        static_cast<cl_ushort>(saturate(pat_float[2], 0.f, 1.f) * 65535.0f),
        static_cast<cl_ushort>(saturate(pat_float[3], 0.f, 1.f) * 65535.0f)};
    cl_short pat_snorm_int16[4] = {
        static_cast<cl_short>(saturate(pat_float[0], -1.f, 1.f) * 32767.0f),
        static_cast<cl_short>(saturate(pat_float[1], -1.f, 1.f) * 32767.0f),
        static_cast<cl_short>(saturate(pat_float[2], -1.f, 1.f) * 32767.0f),
        static_cast<cl_short>(saturate(pat_float[3], -1.f, 1.f) * 32767.0f)};
    cl_short pat_unorm_short_565 =
        (static_cast<cl_ushort>(saturate(pat_float[0], 0.f, 1.f) * 31.0f)
         << 11) |
        (static_cast<cl_ushort>(saturate(pat_float[1], 0.f, 1.f) * 63.0f)
         << 5) |
        (static_cast<cl_ushort>(saturate(pat_float[2], 0.f, 1.f) * 31.0f));
    cl_short pat_unorm_short_555 =
        (static_cast<cl_ushort>(saturate(pat_float[0], 0.f, 1.f) * 31.0f)
         << 10) |
        (static_cast<cl_ushort>(saturate(pat_float[1], 0.f, 1.f) * 31.0f)
         << 5) |
        (static_cast<cl_ushort>(saturate(pat_float[2], 0.f, 1.f) * 31.0f));
    cl_short pat_unorm_int_101010 =
        (static_cast<cl_uint>(saturate(pat_float[0], 0.f, 1.f) * 1023.0f)
         << 20) |
        (static_cast<cl_uint>(saturate(pat_float[1], 0.f, 1.f) * 1023.0f)
         << 10) |
        (static_cast<cl_uint>(saturate(pat_float[2], 0.f, 1.f) * 1023.0f));
    cl_short pat_unorm_int_101010_2 =
        (static_cast<cl_uint>(saturate(pat_float[0], 0.f, 1.f) * 1023.0f)
         << 22) |
        (static_cast<cl_uint>(saturate(pat_float[1], 0.f, 1.f) * 1023.0f)
         << 12) |
        (static_cast<cl_uint>(saturate(pat_float[1], 0.f, 1.f) * 1023.0f)
         << 2) |
        (static_cast<cl_uint>(saturate(pat_float[3], 0.f, 1.f) * 3.0f));

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
    case CL_UNORM_SHORT_565:
        cast_pattern = &pat_unorm_short_565;
        break;
    case CL_UNORM_SHORT_555:
        cast_pattern = &pat_unorm_short_555;
        break;
    case CL_UNORM_INT_101010:
        cast_pattern = &pat_unorm_int_101010;
        break;
    case CL_UNORM_INT_101010_2:
        cast_pattern = &pat_unorm_int_101010_2;
        break;
    default:
        CVK_ASSERT(false);
        return;
    }

    size_t csize = element_size_per_channel();

    switch (format().image_channel_order) {
    case CL_R:
    case CL_RG:
    case CL_RGB:
    case CL_RGBA:
    case CL_LUMINANCE:
    case CL_INTENSITY:
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
