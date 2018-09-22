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

#include "objects.hpp"

using cvk_mem_callback_pointer_type = void (*) (cl_mem mem, void *user_data);

struct cvk_mem_callback {
    cvk_mem_callback_pointer_type pointer;
    void *data;
};

typedef struct _cl_mem cvk_mem;

typedef struct _cl_mem : public api_object {

    static const VkBufferUsageFlags USAGE_FLAGS = \
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | \
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | \
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | \
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    _cl_mem(cvk_context *ctx, cl_mem_flags flags, size_t sz, void *hptr,
            cvk_mem *par, size_t par_off, cl_mem_object_type type = CL_MEM_OBJECT_BUFFER) :
        api_object(ctx),
        m_type(type),
        m_flags(flags),
        m_map_count(0),
        m_parent(par),
        m_parent_offset(par_off),
        m_host_ptr(hptr),
        m_size(sz),
        m_memory(VK_NULL_HANDLE),
        m_buffer(VK_NULL_HANDLE)
{
        if (m_parent != nullptr) {
            m_parent->retain();

            // Handle flag inheritance
            cl_mem_flags access_flags = CL_MEM_READ_WRITE | CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY;
            if ((m_flags & access_flags) == 0) {
                m_flags |= m_parent->m_flags & access_flags;
            }

            m_flags |= m_parent->m_flags & (CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR);

            cl_mem_flags host_access_flags = CL_MEM_HOST_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_NO_ACCESS;
            if ((m_flags & host_access_flags) == 0) {
                m_flags |= m_parent->m_flags & host_access_flags;
            }

            // Handle host_ptr
            m_host_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(m_parent->host_ptr()) + m_parent_offset);
        }
    }

    static std::unique_ptr<cvk_mem> create(cvk_context *context, cl_mem_flags, size_t size, void *host_ptr, cl_int *errcode_ret);
    cvk_mem* create_subbuffer(cl_mem_flags, size_t origin, size_t size);

    virtual ~_cl_mem() {
        if (m_parent != nullptr) {
            m_parent->release();
        }
        auto device = m_context->device()->vulkan_device();

        vkDestroyBuffer(device, m_buffer, nullptr);

        if (m_parent == nullptr) {
            vkFreeMemory(device, m_memory, nullptr);
        }

        for (auto cbi = m_callbacks.rbegin(); cbi != m_callbacks.rend(); ++cbi) {
            auto cb = *cbi;
            cb.pointer(this, cb.data);
        }
    }

    VkBuffer vulkan_buffer() const { return m_buffer; }
    uint32_t map_count() const { return m_map_count; }
    cvk_mem* parent() const { return m_parent; }
    size_t parent_offset() const { return m_parent_offset; }
    void* host_ptr() const { return m_host_ptr; }
    size_t size() const { return m_size; }
    cl_mem_object_type type() const { return m_type; }
    cl_mem_flags flags() const { return m_flags; }

    static bool is_image_type(cl_mem_object_type type) {
        return ((type == CL_MEM_OBJECT_IMAGE1D) ||
                (type == CL_MEM_OBJECT_IMAGE1D_ARRAY) ||
                (type == CL_MEM_OBJECT_IMAGE1D_BUFFER) ||
                (type == CL_MEM_OBJECT_IMAGE2D) ||
                (type == CL_MEM_OBJECT_IMAGE2D_ARRAY) ||
                (type == CL_MEM_OBJECT_IMAGE3D));
    }

    bool is_image_type() const {
        return is_image_type(type());
    }

    bool has_flags(cl_mem_flags flags) {
        return (m_flags & flags) == flags;
    }

    bool has_any_flag(cl_mem_flags flags) {
        return (m_flags & flags) != 0;
    }

    void add_destructor_callback(cvk_mem_callback_pointer_type ptr, void *user_data) {
        cvk_mem_callback cb = {ptr, user_data};
        m_callbacks.push_back(cb);
    }

    void* host_va() const {
        CVK_ASSERT(m_map_count > 0);
        return m_map_ptr;
    }

    bool CHECK_RETURN map() {
        std::lock_guard<std::mutex> lock(m_map_lock);
        cvk_debug("%p::map", this);

        if (m_parent != nullptr) {
            if (!m_parent->map()) {
                return false;
            }
            m_map_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(m_parent->host_va()) + m_parent_offset);
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

    void unmap() {
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

    bool CHECK_RETURN copy_to(void *dst, size_t offset, size_t size) {
        if (map()) {
            void *src = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(m_map_ptr) + offset);
            memcpy(dst, src, size);
            unmap();
            return true;
        }
        return false;
    }

    bool CHECK_RETURN copy_to(cvk_mem *dst, size_t src_offset, size_t dst_offset, size_t size) {
        if (map() && dst->map()) {
            void *src_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(m_map_ptr) + src_offset);
            void *dst_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(dst->host_va()) + dst_offset);
            memcpy(dst_ptr, src_ptr, size);
            dst->unmap();
            unmap();
            return true;
        }
        return false;
    }

    bool CHECK_RETURN copy_from(const void *src, size_t offset, size_t size) {
        if (map()) {
            void *dst = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(m_map_ptr) + offset);
            memcpy(dst, src, size);
            unmap();
            return true;
        }
        return false;
    }

private:
    bool init();
    bool init_subbuffer();

    cl_mem_object_type m_type;
    std::mutex m_map_lock;
    cl_mem_flags m_flags;
    uint32_t m_map_count;
    void *m_map_ptr;
    cvk_mem *m_parent;
    size_t m_parent_offset;
    void *m_host_ptr;
    size_t m_size;
    VkDeviceMemory m_memory;
    VkBuffer m_buffer;
    std::vector<cvk_mem_callback> m_callbacks;

} cvk_mem;

typedef struct _cl_sampler : public api_object {

    _cl_sampler(cvk_context *context, bool normalized_coords,
                cl_addressing_mode addressing_mode, cl_filter_mode filter_mode) :
        api_object(context),
        m_normalized_coords(normalized_coords),
        m_addressing_mode(addressing_mode),
        m_filter_mode(filter_mode) {} // TODO actually create a sampler

    bool normalized_coords() const { return m_normalized_coords; }
    cl_addressing_mode addressing_mode() const { return m_addressing_mode; }
    cl_filter_mode filter_mode() const { return m_filter_mode; }

private:
    bool m_normalized_coords;
    cl_addressing_mode m_addressing_mode;
    cl_filter_mode m_filter_mode;
} cvk_sampler;

struct cvk_image : public cvk_mem {

    cvk_image(cvk_context *ctx, cl_mem_flags flags, const cl_image_desc *desc,
              const cl_image_format *format, void *host_ptr)
        : cvk_mem(ctx, flags, /* FIXME size */ 0,
                  host_ptr, /* FIXME parent */ nullptr,
                  /* FIXME parent_offset */ 0, desc->image_type),
          m_desc(*desc),
          m_format(*format)
    {
    }

    const cl_image_format& format() const { return m_format; }
    size_t element_size() const {
        return num_channels() * element_size_per_channel();
    }
    size_t row_pitch() const { return m_desc.image_row_pitch; }
    size_t slice_pitch() const { return m_desc.image_slice_pitch; }
    size_t width() const { return m_desc.image_width; }
    size_t height() const { return m_desc.image_height; }
    size_t depth() const { return m_desc.image_depth; }
    size_t array_size() const { return m_desc.image_array_size; }
    cvk_mem* buffer() const { return m_desc.buffer; }
    cl_uint num_mip_levels() const { return m_desc.num_mip_levels; }
    cl_uint num_samples() const { return m_desc.num_samples; }

private:

    size_t num_channels() const {
        switch (m_format.image_channel_order) {
        case CL_R:
        case CL_Rx:
        case CL_A:
        case CL_INTENSITY:
        case CL_LUMINANCE:
            return 1;
        case CL_RG:
        case CL_RGx:
        case CL_RA:
            return 2;
        case CL_RGB:
        case CL_RGBx:
            return 3;
        case CL_RGBA:
        case CL_ARGB:
        case CL_BGRA:
            return 4;
        default:
            return 0;
        }
    }

    size_t element_size_per_channel() const {
        switch (m_format.image_channel_data_type) {
        case CL_SNORM_INT8:
        case CL_UNORM_INT8:
        case CL_SIGNED_INT8:
        case CL_UNSIGNED_INT8:
            return 1;
        case CL_SNORM_INT16:
        case CL_UNORM_INT16:
        case CL_SIGNED_INT16:
        case CL_UNSIGNED_INT16:
        case CL_UNORM_SHORT_565:
        case CL_UNORM_SHORT_555:
        case CL_HALF_FLOAT:
            return 2;
        case CL_UNORM_INT_101010:
        case CL_SIGNED_INT32:
        case CL_UNSIGNED_INT32:
        case CL_FLOAT:
            return 4;
        default:
            return 0;
        }
    }

    cl_image_desc m_desc;
    cl_image_format m_format;
};
