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
#include <list>

#include "device.hpp"
#include "event.hpp"
#include "objects.hpp"
#include "utils.hpp"

struct cvk_memory_allocation {

    cvk_memory_allocation(VkDevice dev, VkDeviceSize size, uint32_t type_index,
                          bool coherent)
        : m_device(dev), m_size(size), m_memory(VK_NULL_HANDLE),
          m_memory_type_index(type_index), m_coherent(coherent) {}

    ~cvk_memory_allocation() {
        if (m_memory != VK_NULL_HANDLE) {
            vkFreeMemory(m_device, m_memory, nullptr);
        }
    }

    VkResult allocate(bool physical_addressing) {
        const VkMemoryAllocateFlagsInfo flagsInfo = {
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, nullptr,
            VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};

        const VkMemoryAllocateInfo memoryAllocateInfo = {
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            physical_addressing ? &flagsInfo : nullptr,
            m_size,
            m_memory_type_index,
        };

        return vkAllocateMemory(m_device, &memoryAllocateInfo, 0, &m_memory);
    }

    void invalidate(VkDeviceSize offset, VkDeviceSize size) {
        if (!m_coherent) {
            TRACE_BEGIN("invalidate_memory", "offset", offset, "size", size);
            const VkMappedMemoryRange range = {
                VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, m_memory,
                offset, size};
            vkInvalidateMappedMemoryRanges(m_device, 1, &range);
            TRACE_END();
        }
    }

    void flush(VkDeviceSize offset, VkDeviceSize size) {
        if (!m_coherent) {
            TRACE_BEGIN("flush_memory", "offset", offset, "size", size);
            const VkMappedMemoryRange range = {
                VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, m_memory,
                offset, size};
            vkFlushMappedMemoryRanges(m_device, 1, &range);
            TRACE_END();
        }
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
    bool m_coherent;
};

using cvk_mem_callback_pointer_type = void(CL_CALLBACK*)(cl_mem mem,
                                                         void* user_data);

struct cvk_mem_callback {
    cvk_mem_callback_pointer_type pointer;
    void* data;
};

struct cvk_mem;
using cvk_mem_holder = refcounted_holder<cvk_mem>;

enum class cvk_mem_init_state
{
    required,
    scheduled,
    completed
};

struct cvk_mem_init_tracker {
    cvk_mem_init_tracker()
        : m_state(cvk_mem_init_state::required), m_event(nullptr) {}

    cvk_mem_init_state state() const { return m_state; }
    void set_state(cvk_mem_init_state state) { m_state = state; }
    cvk_event* event() const { return m_event; }
    void set_event(cvk_event* event) {
        CVK_ASSERT(m_event == nullptr);
        CVK_ASSERT(state() == cvk_mem_init_state::required);
        set_state(cvk_mem_init_state::scheduled);
        m_event.reset(event);
    }

    std::mutex& mutex() { return m_mutex; }

private:
    std::mutex m_mutex;
    cvk_mem_init_state m_state;
    cvk_event_holder m_event;
};

struct cvk_mem : public _cl_mem, api_object<object_magic::memory_object> {

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

    std::shared_ptr<cvk_memory_allocation> memory() const {
        if (m_parent == nullptr) {
            return m_memory;
        } else {
            return m_parent->memory();
        }
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

    bool CHECK_RETURN map() {
        auto ret = map_memory();
        if (!ret) {
            return ret;
        }
        invalidate_memory(0, m_size);
        return true;
    }
    bool CHECK_RETURN map_write_only() { return map_memory(); }
    bool CHECK_RETURN map_to_read(VkDeviceSize offset, VkDeviceSize size) {
        auto ret = map_memory();
        if (!ret) {
            return ret;
        }
        invalidate_memory(offset, size);
        return true;
    }
    void unmap() {
        flush_memory(0, m_size);
        unmap_memory();
    }
    void unmap_read_only() { unmap_memory(); }
    void unmap_to_write(VkDeviceSize offset, VkDeviceSize size) {
        flush_memory(offset, size);
        unmap_memory();
    }

    bool CHECK_RETURN copy_to(void* dst, size_t offset, size_t size) {
        if (map_to_read(offset, size)) {
            void* src = pointer_offset(m_map_ptr, offset);
            memcpy(dst, src, size);
            unmap_read_only();
            return true;
        }
        return false;
    }

    bool CHECK_RETURN copy_to(cvk_mem* dst, size_t src_offset,
                              size_t dst_offset, size_t size) {
        if (map_to_read(src_offset, size) && dst->map_write_only()) {
            void* src_ptr = pointer_offset(m_map_ptr, src_offset);
            void* dst_ptr = pointer_offset(dst->host_va(), dst_offset);
            memcpy(dst_ptr, src_ptr, size);
            dst->unmap_to_write(dst_offset, size);
            unmap_read_only();
            return true;
        }
        return false;
    }

    bool CHECK_RETURN copy_from(const void* src, size_t offset, size_t size) {
        if (map_write_only()) {
            void* dst = pointer_offset(m_map_ptr, offset);
            memcpy(dst, src, size);
            unmap_to_write(offset, size);
            return true;
        }
        return false;
    }

    cvk_mem_init_tracker& init_tracker() { return m_init_tracker; }

    void invalidate_memory(VkDeviceSize offset, VkDeviceSize size);

private:
    bool CHECK_RETURN map_memory();
    void unmap_memory();
    void flush_memory(VkDeviceSize offset, VkDeviceSize size);

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
    std::shared_ptr<cvk_memory_allocation> m_memory;
    cvk_mem_init_tracker m_init_tracker{};
};

static inline cvk_mem* icd_downcast(cl_mem mem) {
    return static_cast<cvk_mem*>(mem);
}

struct cvk_buffer;
struct cvk_image;

using cvk_image_holder = refcounted_holder<cvk_image>;

struct cvk_memobj_mappping {
    cvk_buffer* buffer;
    void* ptr;
    cl_map_flags flags;
};

struct cvk_buffer_mapping : public cvk_memobj_mappping {
    size_t offset;
    size_t size;

    // Needed for buffer mapped through clEnqueueMapImage with a
    // CL_MEM_OBJECT_IMAGE1D_BUFFER.
    cvk_image_holder image;
};

struct cvk_buffer : public cvk_mem {

    cvk_buffer(cvk_context* ctx, cl_mem_flags flags, size_t size,
               void* host_ptr, cvk_mem* parent, size_t parent_offset,
               std::vector<cl_mem_properties>&& properties)
        : cvk_mem(ctx, flags, size, host_ptr, parent, parent_offset,
                  std::move(properties), CL_MEM_OBJECT_BUFFER),
          m_buffer(VK_NULL_HANDLE) {
        // Buffers currently do not require any asynchronous initialisation
        m_init_tracker.set_state(cvk_mem_init_state::completed);
    }

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

    VkBufferUsageFlags prepare_usage_flags() {
        VkBufferUsageFlags usage_flags =
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT |
            VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT;
        if (m_context->device()->uses_physical_addressing()) {
            usage_flags |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        }
        return usage_flags;
    }

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
                                size_t size, cl_map_flags flags,
                                cvk_image* image) {

        if (!map()) {
            return false;
        }

        mapping.buffer = this;
        mapping.offset = offset;
        mapping.size = size;
        mapping.ptr = this->map_ptr(offset);
        mapping.flags = flags;
        mapping.image.reset(image);

        return true;
    }

    bool insert_mapping(const cvk_buffer_mapping& mapping) {
        std::lock_guard<std::mutex> lock(m_mappings_lock);
        auto num_mappings_with_same_pointer = m_mappings.count(mapping.ptr);
        // TODO support multiple mappings with the same pointer
        if (num_mappings_with_same_pointer != 0) {
            return false;
        }

        // memory has been mapped when the mapping has been created (when the
        // enqueue command has been created). We need to invalidate it before
        // the command execution to make sure of the content of the memory.
        invalidate_memory(mapping.offset, mapping.size);

        m_mappings.insert({mapping.ptr, mapping});

        return true;
    }

    cvk_buffer_mapping remove_mapping(void* ptr) {
        std::lock_guard<std::mutex> lock(m_mappings_lock);
        CVK_ASSERT(m_mappings.count(ptr) > 0);
        auto mapping = m_mappings.at(ptr);
        m_mappings.erase(ptr);
        mapping.buffer->unmap();
        mapping.image.reset(nullptr);
        return mapping;
    }

    void cleanup_mapping(cvk_buffer_mapping& mapping) {
        std::lock_guard<std::mutex> lock(m_mappings_lock);
        if (m_mappings.count(mapping.ptr)) {
            m_mappings.erase(mapping.ptr);
        }
        mapping.buffer->unmap();
    }

    uint64_t device_address() const {
        VkBufferDeviceAddressInfo info{};
        info.buffer = vulkan_buffer();
        info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        info.pNext = NULL;
        auto device = context()->device();
        auto vkdev = device->vulkan_device();
        auto device_address =
            device->vkfns().vkGetBufferDeviceAddressKHR(vkdev, &info) +
            vulkan_buffer_offset();
        device->device_to_buffer_map[(void*)device_address] = (void*)this;
        return device_address;
    }

private:
    bool init();

    VkBuffer m_buffer;
    std::unordered_map<void*, cvk_buffer_mapping> m_mappings;
    std::mutex m_mappings_lock;
};

using cvk_buffer_holder = refcounted_holder<cvk_buffer>;

struct cvk_sampler;
using cvk_sampler_holder = refcounted_holder<cvk_sampler>;

struct cvk_sampler : public _cl_sampler, api_object<object_magic::sampler> {

    cvk_sampler(cvk_context* context, bool normalized_coords,
                cl_addressing_mode addressing_mode, cl_filter_mode filter_mode,
                std::vector<cl_sampler_properties>&& properties)
        : api_object(context), m_normalized_coords(normalized_coords),
          m_addressing_mode(addressing_mode), m_filter_mode(filter_mode),
          m_properties(std::move(properties)), m_sampler(VK_NULL_HANDLE),
          m_sampler_norm(VK_NULL_HANDLE) {}

    ~cvk_sampler() {
        auto vkdev = context()->device()->vulkan_device();
        if (m_sampler != VK_NULL_HANDLE) {
            vkDestroySampler(vkdev, m_sampler, nullptr);
        }
        if (m_sampler_norm != VK_NULL_HANDLE) {
            vkDestroySampler(vkdev, m_sampler_norm, nullptr);
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
    VkSampler get_or_create_vulkan_sampler_with_normalized_coords() {
        if (m_sampler_norm == VK_NULL_HANDLE) {
            if (!init(true)) {
                return VK_NULL_HANDLE;
            }
        }
        return m_sampler_norm;
    }
    const std::vector<cl_sampler_properties>& properties() const {
        return m_properties;
    }

private:
    bool init(bool force_normalized_coordinates = false);
    bool m_normalized_coords;
    cl_addressing_mode m_addressing_mode;
    cl_filter_mode m_filter_mode;
    const std::vector<cl_sampler_properties> m_properties;
    VkSampler m_sampler;
    VkSampler m_sampler_norm;
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
          m_sampled_view(VK_NULL_HANDLE), m_storage_view(VK_NULL_HANDLE),
          m_buffer_view(VK_NULL_HANDLE) {
        // All images require asynchronous initialiation for the initial
        // layout transition (and copy/use host ptr init) apart from
        // those backed by a texel buffer
        if (is_backed_by_buffer_view()) {
            m_init_tracker.set_state(cvk_mem_init_state::completed);
        } else {
            m_init_tracker.set_state(cvk_mem_init_state::required);
        }
    }

    ~cvk_image() {
        auto vkdev = m_context->device()->vulkan_device();
        if (m_image != VK_NULL_HANDLE) {
            vkDestroyImage(vkdev, m_image, nullptr);
        }
        if (m_sampled_view != VK_NULL_HANDLE) {
            vkDestroyImageView(vkdev, m_sampled_view, nullptr);
        }
        if (m_storage_view != VK_NULL_HANDLE) {
            vkDestroyImageView(vkdev, m_storage_view, nullptr);
        }
        if (m_buffer_view != VK_NULL_HANDLE) {
            vkDestroyBufferView(vkdev, m_buffer_view, nullptr);
        }
        if (buffer() != nullptr) {
            buffer()->release();
        }
    }

    static VkFormatFeatureFlags
    required_format_feature_flags_for(cl_mem_object_type type,
                                      cl_mem_flags flags);
    VkImageUsageFlags prepare_usage_flags() {
        VkImageUsageFlags usage_flags =
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

        if (flags() & (CL_MEM_KERNEL_READ_AND_WRITE | CL_MEM_WRITE_ONLY)) {
            usage_flags |= VK_IMAGE_USAGE_STORAGE_BIT;
        } else if (flags() & CL_MEM_READ_ONLY) {
            usage_flags |= VK_IMAGE_USAGE_SAMPLED_BIT;
        } else {
            usage_flags |=
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
        }
        return usage_flags;
    }

    static cvk_image* create(cvk_context* ctx, cl_mem_flags flags,
                             const cl_image_desc* desc,
                             const cl_image_format* format, void* host_ptr,
                             std::vector<cl_mem_properties>&& properties);

    bool is_backed_by_buffer_view() const {
        return type() == CL_MEM_OBJECT_IMAGE1D_BUFFER;
    }

    VkImage vulkan_image() const {
        CVK_ASSERT(!is_backed_by_buffer_view());
        return m_image;
    }
    VkImageView vulkan_sampled_view() const {
        CVK_ASSERT(!is_backed_by_buffer_view());
        return m_sampled_view;
    }
    VkImageView vulkan_storage_view() const {
        CVK_ASSERT(!is_backed_by_buffer_view());
        return m_storage_view;
    }
    VkBufferView vulkan_buffer_view() const {
        CVK_ASSERT(is_backed_by_buffer_view());
        return m_buffer_view;
    }
    const cl_image_format& format() const { return m_format; }
    size_t element_size() const {
        switch (m_format.image_channel_data_type) {
        case CL_UNORM_SHORT_555:
        case CL_UNORM_SHORT_565:
            return 2;
        case CL_UNORM_INT_101010:
        case CL_UNORM_INT_101010_2:
            return 4;
        default:
            return num_channels() * element_size_per_channel();
        }
    }
    size_t row_pitch() const {
        if (m_desc.image_row_pitch == 0) {
            return element_size() * width();
        } else {
            return m_desc.image_row_pitch;
        }
    }
    size_t slice_pitch() const {
        if (m_desc.image_slice_pitch == 0) {
            switch (type()) {
            case CL_MEM_OBJECT_IMAGE1D:
            case CL_MEM_OBJECT_IMAGE1D_BUFFER:
            case CL_MEM_OBJECT_IMAGE2D:
                return 0;
            default:
                return row_pitch() * height();
            }
        } else {
            return m_desc.image_slice_pitch;
        }
    }
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
                                cl_map_flags flags, bool handle_host_ptr) {
        std::lock_guard<std::mutex> lock(m_mappings_lock);
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
        mapping.flags = flags;

        if (handle_host_ptr && has_flags(CL_MEM_USE_HOST_PTR)) {
            uintptr_t offset = slice_pitch() * origin[2] +
                               row_pitch() * origin[1] +
                               origin[0] * element_size();
            mapping.ptr = pointer_offset(host_ptr(), offset);
        } else {
            mapping.ptr = mapping.buffer->map_ptr(0);
        }

        auto num_mappings_with_same_pointer = m_mappings.count(mapping.ptr);
        // TODO support multiple mappings with the same pointer
        if (num_mappings_with_same_pointer != 0 &&
            !has_flags(CL_MEM_USE_HOST_PTR)) {
            cvk_error_fn(
                "creating multiple image mappings with the same "
                "pointer is not supported for image without a host ptr");
            return false;
        }

        // TODO should insertion be deferred, as done for buffers?
        m_mappings[mapping.ptr].push_back(mapping);

        return true;
    }

    cvk_image_mapping remove_mapping(void* ptr) {
        std::lock_guard<std::mutex> lock(m_mappings_lock);
        CVK_ASSERT(m_mappings.count(ptr) > 0);
        auto mapping = m_mappings.at(ptr).front();
        m_mappings.at(ptr).pop_front();
        if (m_mappings.at(ptr).size() == 0) {
            m_mappings.erase(ptr);
        }
        mapping.buffer->unmap();
        mapping.buffer->release();
        mapping.buffer = nullptr;
        mapping.ptr = nullptr;
        return mapping;
    }

    cvk_image_mapping mapping_for(void* ptr) {
        std::lock_guard<std::mutex> lock(m_mappings_lock);
        CVK_ASSERT(m_mappings.count(ptr) > 0);
        auto mapping = m_mappings.at(ptr).front();
        return mapping;
    }

    size_t map_buffer_row_pitch(const std::array<size_t, 3>& region) const {
        return region[0] * element_size();
    }

    size_t map_buffer_row_pitch(const cvk_image_mapping& mapping) const {
        return map_buffer_row_pitch(mapping.region);
    }

    size_t map_buffer_slice_pitch(const cvk_image_mapping& mapping) const {
        return map_buffer_slice_pitch(mapping.region);
    }

    size_t map_buffer_slice_pitch(const std::array<size_t, 3>& region) const {
        switch (type()) {
        case CL_MEM_OBJECT_IMAGE1D_ARRAY:
            return map_buffer_row_pitch(region);
            break;
        default:
            return map_buffer_row_pitch(region) * region[1];
        }
    }

    const cvk_buffer* init_data() const { return m_init_data.get(); }

    void discard_init_data() { m_init_data.reset(); }

    static constexpr int MAX_NUM_CHANNELS = 4;
    static constexpr int MAX_CHANNEL_SIZE = 4;
    static constexpr int FILL_PATTERN_MAX_SIZE =
        MAX_NUM_CHANNELS * MAX_CHANNEL_SIZE;
    using fill_pattern_array = std::array<char, FILL_PATTERN_MAX_SIZE>;
    void prepare_fill_pattern(const void* input_pattern,
                              fill_pattern_array& pattern,
                              size_t* size_ret) const;

private:
    bool init_vulkan_image();
    bool init_vulkan_texel_buffer();
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
        case CL_HALF_FLOAT:
            return 2;
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
    VkImageView m_sampled_view;
    VkImageView m_storage_view;
    VkBufferView m_buffer_view;
    std::unordered_map<void*, std::list<cvk_image_mapping>> m_mappings;
    std::mutex m_mappings_lock;
    std::unique_ptr<cvk_buffer> m_init_data;
};
