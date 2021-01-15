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

#include "objects.hpp"

struct cvk_memory_allocation {

    cvk_memory_allocation(VkDevice dev, VkDeviceSize size, uint32_t type_index)
        : m_device(dev), m_size(size), m_memory(VK_NULL_HANDLE),
          m_memory_type_index(type_index) {}

    ~cvk_memory_allocation() {
        if (m_memory != VK_NULL_HANDLE) {
            vkFreeMemory(m_device, m_memory, nullptr);
        }
    }

    VkResult allocate() {
        const VkMemoryAllocateInfo memoryAllocateInfo = {
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            nullptr,
            m_size,
            m_memory_type_index,
        };

        return vkAllocateMemory(m_device, &memoryAllocateInfo, 0, &m_memory);
    }

    VkResult map(void** map_ptr) {
        return vkMapMemory(m_device, m_memory, 0, m_size, 0, map_ptr);
    }

    void unmap() { vkUnmapMemory(m_device, m_memory); }

    VkDeviceMemory vulkan_memory() { return m_memory; }

private:
    VkDevice m_device;
    VkDeviceSize m_size;
    VkDeviceMemory m_memory;
    uint32_t m_memory_type_index;
};

using cvk_mem_callback_pointer_type = void(CL_CALLBACK*)(cl_mem mem,
                                                         void* user_data);

struct cvk_mem_callback {
    cvk_mem_callback_pointer_type pointer;
    void* data;
};

struct cvk_mem;
using cvk_mem_holder = refcounted_holder<cvk_mem>;

struct cvk_mem : public _cl_mem, api_object {

    cvk_mem(cvk_context* ctx, cl_mem_flags flags, size_t size, void* host_ptr,
            cvk_mem* parent, size_t parent_offset,
            std::vector<cl_mem_properties>&& properties,
            cl_mem_object_type type)
        : api_object(ctx), m_type(type), m_flags(flags), m_map_count(0),
          m_map_ptr(nullptr), m_properties(std::move(properties)), m_size(size),
          m_host_ptr(host_ptr), m_parent(parent),
          m_parent_offset(parent_offset) {

        if (m_parent != nullptr) {

            // Handle flag inheritance
            cl_mem_flags access_flags =
                CL_MEM_READ_WRITE | CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY;

            if ((m_flags & access_flags) == 0) {
                m_flags |= m_parent->m_flags & access_flags;
            }

            cl_mem_flags host_ptr_flags = CL_MEM_USE_HOST_PTR |
                                          CL_MEM_COPY_HOST_PTR |
                                          CL_MEM_ALLOC_HOST_PTR;

            m_flags |= m_parent->m_flags & host_ptr_flags;

            cl_mem_flags host_access_flags = CL_MEM_HOST_WRITE_ONLY |
                                             CL_MEM_HOST_READ_ONLY |
                                             CL_MEM_HOST_NO_ACCESS;

            if ((m_flags & host_access_flags) == 0) {
                m_flags |= m_parent->m_flags & host_access_flags;
            }

            // Handle host_ptr
            m_host_ptr = pointer_offset(m_parent->host_ptr(), m_parent_offset);
        }
    }

    virtual ~cvk_mem() {
        for (auto cbi = m_callbacks.rbegin(); cbi != m_callbacks.rend();
             ++cbi) {
            auto cb = *cbi;
            cb.pointer(this, cb.data);
        }
    }

    uint32_t map_count() const { return m_map_count; }
    cvk_mem* parent() const { return m_parent; }
    size_t parent_offset() const { return m_parent_offset; }
    void* host_ptr() const { return m_host_ptr; }
    size_t size() const { return m_size; }
    cl_mem_object_type type() const { return m_type; }
    cl_mem_flags flags() const { return m_flags; }
    const std::vector<cl_mem_properties>& properties() const {
        return m_properties;
    }

    static bool is_image_type(cl_mem_object_type type) {
        return ((type == CL_MEM_OBJECT_IMAGE1D) ||
                (type == CL_MEM_OBJECT_IMAGE1D_ARRAY) ||
                (type == CL_MEM_OBJECT_IMAGE1D_BUFFER) ||
                (type == CL_MEM_OBJECT_IMAGE2D) ||
                (type == CL_MEM_OBJECT_IMAGE2D_ARRAY) ||
                (type == CL_MEM_OBJECT_IMAGE3D));
    }

    bool is_image_type() const { return is_image_type(type()); }

    bool is_buffer_type() const { return type() == CL_MEM_OBJECT_BUFFER; }

    bool is_sub_buffer() const {
        return is_buffer_type() && (parent() != nullptr);
    }

    bool has_flags(cl_mem_flags flags) const {
        return (m_flags & flags) == flags;
    }

    bool has_any_flag(cl_mem_flags flags) const {
        return (m_flags & flags) != 0;
    }

    void add_destructor_callback(cvk_mem_callback_pointer_type ptr,
                                 void* user_data) {
        cvk_mem_callback cb = {ptr, user_data};
        std::lock_guard<std::mutex> lock(m_callbacks_lock);
        m_callbacks.push_back(cb);
    }

    void* host_va() const {
        CVK_ASSERT(m_map_ptr != nullptr);
        return m_map_ptr;
    }

    bool CHECK_RETURN map();
    void unmap();

    bool CHECK_RETURN copy_to(void* dst, size_t offset, size_t size) {
        if (map()) {
            void* src = pointer_offset(m_map_ptr, offset);
            memcpy(dst, src, size);
            unmap();
            return true;
        }
        return false;
    }

    bool CHECK_RETURN copy_to(cvk_mem* dst, size_t src_offset,
                              size_t dst_offset, size_t size) {
        if (map() && dst->map()) {
            void* src_ptr = pointer_offset(m_map_ptr, src_offset);
            void* dst_ptr = pointer_offset(dst->host_va(), dst_offset);
            memcpy(dst_ptr, src_ptr, size);
            dst->unmap();
            unmap();
            return true;
        }
        return false;
    }

    bool CHECK_RETURN copy_from(const void* src, size_t offset, size_t size) {
        if (map()) {
            void* dst = pointer_offset(m_map_ptr, offset);
            memcpy(dst, src, size);
            unmap();
            return true;
        }
        return false;
    }

private:
    cl_mem_object_type m_type;
    std::mutex m_map_lock;
    cl_mem_flags m_flags;
    uint32_t m_map_count;
    void* m_map_ptr;
    std::mutex m_callbacks_lock;
    std::vector<cvk_mem_callback> m_callbacks;
    std::vector<cl_mem_properties> m_properties;

protected:
    size_t m_size;
    void* m_host_ptr;
    cvk_mem_holder m_parent;
    size_t m_parent_offset;
    std::unique_ptr<cvk_memory_allocation> m_memory;
};

static inline cvk_mem* icd_downcast(cl_mem mem) {
    return static_cast<cvk_mem*>(mem);
}

struct cvk_buffer;

struct cvk_memobj_mappping {
    cvk_buffer* buffer;
    void* ptr;
    cl_map_flags flags;
};

struct cvk_buffer_mapping : public cvk_memobj_mappping {
    size_t offset;
    size_t size;
};

struct cvk_buffer : public cvk_mem {

    static const VkBufferUsageFlags USAGE_FLAGS =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    cvk_buffer(cvk_context* ctx, cl_mem_flags flags, size_t size,
               void* host_ptr, cvk_mem* parent, size_t parent_offset,
               std::vector<cl_mem_properties>&& properties)
        : cvk_mem(ctx, flags, size, host_ptr, parent, parent_offset,
                  std::move(properties), CL_MEM_OBJECT_BUFFER),
          m_buffer(VK_NULL_HANDLE) {}

    virtual ~cvk_buffer() {
        auto vkdev = m_context->device()->vulkan_device();
        vkDestroyBuffer(vkdev, m_buffer, nullptr);
    }

    static std::unique_ptr<cvk_buffer> create(cvk_context* context,
                                              cl_mem_flags flags, size_t size,
                                              void* host_ptr,
                                              cl_int* errcode_ret) {
        std::vector<cl_mem_properties> properties;
        return create(context, flags, size, host_ptr, std::move(properties),
                      errcode_ret);
    }

    static std::unique_ptr<cvk_buffer>
    create(cvk_context* context, cl_mem_flags, size_t size, void* host_ptr,
           std::vector<cl_mem_properties>&& properties, cl_int* errcode_ret);
    cvk_mem* create_subbuffer(cl_mem_flags, size_t origin, size_t size);

    VkBuffer vulkan_buffer() const {
        if (m_parent == nullptr) {
            return m_buffer;
        } else {
            const cvk_mem *parent = m_parent;
            return static_cast<const cvk_buffer*>(parent)->vulkan_buffer();
        }
    }

    size_t vulkan_buffer_offset() const {
        return m_parent_offset;
    }

    void* map_ptr(size_t offset) const {
        void* ptr;
        if (has_flags(CL_MEM_USE_HOST_PTR)) {
            ptr = host_ptr();
        } else {
            ptr = host_va();
        }

        ptr = pointer_offset(ptr, offset);

        return ptr;
    }

    bool find_or_create_mapping(cvk_buffer_mapping& mapping, size_t offset,
                                size_t size, cl_map_flags flags) {

        if (!map()) {
            return false;
        }

        mapping.buffer = this;
        mapping.offset = offset;
        mapping.size = size;
        mapping.ptr = this->map_ptr(offset);
        mapping.flags = flags;

        return true;
    }

    bool insert_mapping(const cvk_buffer_mapping& mapping) {
        auto num_mappings_with_same_pointer = m_mappings.count(mapping.ptr);
        // TODO support multiple mappings with the same pointer
        if (num_mappings_with_same_pointer != 0) {
            return false;
        }

        m_mappings[mapping.ptr] = mapping;

        return true;
    }

    cvk_buffer_mapping remove_mapping(void* ptr) {
        CVK_ASSERT(m_mappings.count(ptr) > 0);
        auto mapping = m_mappings.at(ptr);
        m_mappings.erase(ptr);
        mapping.buffer->unmap();
        return mapping;
    }

private:
    bool init();

    VkBuffer m_buffer;
    std::unordered_map<void*, cvk_buffer_mapping> m_mappings;
};

using cvk_buffer_holder = refcounted_holder<cvk_buffer>;

struct cvk_sampler;
using cvk_sampler_holder = refcounted_holder<cvk_sampler>;

struct cvk_sampler : public _cl_sampler, api_object {

    cvk_sampler(cvk_context* context, bool normalized_coords,
                cl_addressing_mode addressing_mode, cl_filter_mode filter_mode,
                std::vector<cl_sampler_properties>&& properties)
        : api_object(context), m_normalized_coords(normalized_coords),
          m_addressing_mode(addressing_mode), m_filter_mode(filter_mode),
          m_properties(std::move(properties)), m_sampler(VK_NULL_HANDLE) {}

    ~cvk_sampler() {
        if (m_sampler != VK_NULL_HANDLE) {
            auto vkdev = context()->device()->vulkan_device();
            vkDestroySampler(vkdev, m_sampler, nullptr);
        }
    }

    static cvk_sampler* create(cvk_context* context, bool normalized_coords,
                               cl_addressing_mode addressing_mode,
                               cl_filter_mode filter_mode,
                               std::vector<cl_sampler_properties>&& properties);
    static cvk_sampler* create(cvk_context* context, bool normalized_coords,
                               cl_addressing_mode addressing_mode,
                               cl_filter_mode filter_mode) {
        std::vector<cl_sampler_properties> properties;
        return create(context, normalized_coords, addressing_mode, filter_mode,
                      std::move(properties));
    }

    bool normalized_coords() const { return m_normalized_coords; }
    cl_addressing_mode addressing_mode() const { return m_addressing_mode; }
    cl_filter_mode filter_mode() const { return m_filter_mode; }
    VkSampler vulkan_sampler() const { return m_sampler; }
    const std::vector<cl_sampler_properties>& properties() const {
        return m_properties;
    }

private:
    bool init();
    bool m_normalized_coords;
    cl_addressing_mode m_addressing_mode;
    cl_filter_mode m_filter_mode;
    const std::vector<cl_sampler_properties> m_properties;
    VkSampler m_sampler;
};

static inline cvk_sampler* icd_downcast(cl_sampler sampler) {
    return static_cast<cvk_sampler*>(sampler);
}

struct cvk_image_mapping : public cvk_memobj_mappping {
    std::array<size_t, 3> origin;
    std::array<size_t, 3> region;
};

struct cvk_image : public cvk_mem {

    cvk_image(cvk_context* ctx, cl_mem_flags flags, const cl_image_desc* desc,
              const cl_image_format* format, void* host_ptr,
              std::vector<cl_mem_properties>&& properties)
        : cvk_mem(ctx, flags, /* FIXME size */ 0, host_ptr,
                  /* FIXME parent */ nullptr,
                  /* FIXME parent_offset */ 0, std::move(properties),
                  desc->image_type),
          m_desc(*desc), m_format(*format), m_image(VK_NULL_HANDLE),
          m_image_view(VK_NULL_HANDLE) {}

    ~cvk_image() {
        auto vkdev = m_context->device()->vulkan_device();
        if (m_image != VK_NULL_HANDLE) {
            vkDestroyImage(vkdev, m_image, nullptr);
        }
        if (m_image_view != VK_NULL_HANDLE) {
            vkDestroyImageView(vkdev, m_image_view, nullptr);
        }
    }

    static cvk_image* create(cvk_context* ctx, cl_mem_flags flags,
                             const cl_image_desc* desc,
                             const cl_image_format* format, void* host_ptr,
                             std::vector<cl_mem_properties>&& properties);

    VkImage vulkan_image() const { return m_image; }
    VkImageView vulkan_image_view() const { return m_image_view; }
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
    cvk_mem* buffer() const { return icd_downcast(m_desc.buffer); }
    cl_uint num_mip_levels() const { return m_desc.num_mip_levels; }
    cl_uint num_samples() const { return m_desc.num_samples; }

    bool has_same_format(const cvk_image* other) const {
        auto fmt = format();
        auto ofmt = other->format();
        return fmt.image_channel_order == ofmt.image_channel_order &&
               fmt.image_channel_data_type == ofmt.image_channel_data_type;
    }

    bool find_or_create_mapping(cvk_image_mapping& mapping,
                                std::array<size_t, 3> origin,
                                std::array<size_t, 3> region,
                                cl_map_flags flags) {
        // TODO try to reuse existing mappings
        // TODO add overlap checks

        // Create a buffer
        // TODO adapt flags depending on the map flags
        auto buffer_size = element_size() * region[0] * region[1] * region[2];
        cl_int err;
        auto buffer = cvk_buffer::create(context(), CL_MEM_READ_WRITE,
                                         buffer_size, nullptr, &err);

        if (err != CL_SUCCESS) {
            return false;
        }

        if (!buffer->map()) {
            return false;
        }

        mapping.buffer = buffer.release();
        mapping.origin = origin;
        mapping.region = region;
        mapping.ptr = mapping.buffer->map_ptr(0);
        mapping.flags = flags;

        auto num_mappings_with_same_pointer = m_mappings.count(mapping.ptr);
        // TODO support multiple mappings with the same pointer
        if (num_mappings_with_same_pointer != 0) {
            return false;
        }

        m_mappings[mapping.ptr] = mapping;

        return true;
    }

    cvk_image_mapping remove_mapping(void* ptr) {
        CVK_ASSERT(m_mappings.count(ptr) > 0);
        auto mapping = m_mappings.at(ptr);
        m_mappings.erase(ptr);
        mapping.buffer->unmap();
        mapping.buffer->release();
        mapping.buffer = nullptr;
        mapping.ptr = nullptr;
        return mapping;
    }

    cvk_image_mapping mapping_for(void* ptr) {
        CVK_ASSERT(m_mappings.count(ptr) > 0);
        auto mapping = m_mappings.at(ptr);
        return mapping;
    }

    static constexpr int MAX_NUM_CHANNELS = 4;
    static constexpr int MAX_CHANNEL_SIZE = 4;
    static constexpr int FILL_PATTERN_MAX_SIZE =
        MAX_NUM_CHANNELS * MAX_CHANNEL_SIZE;
    using fill_pattern_array = std::array<char, FILL_PATTERN_MAX_SIZE>;
    void prepare_fill_pattern(const void* input_pattern,
                              fill_pattern_array& pattern,
                              size_t* size_ret) const;

private:
    bool init();

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

    const cl_image_desc m_desc;
    const cl_image_format m_format;
    VkImage m_image;
    VkImageView m_image_view;
    std::unordered_map<void*, cvk_image_mapping> m_mappings;
};

using cvk_image_holder = refcounted_holder<cvk_image>;
