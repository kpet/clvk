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

#include <limits>

#include "memory.hpp"
#include "objects.hpp"
#include "program.hpp"

struct cvk_kernel_arg_storage {
    char *ptr;
    size_t size;
};

struct cvk_kernel_pipeline_cache_entry {
    uint32_t lws[3];
    VkPipeline pipeline;
};

typedef struct _cl_kernel cvk_kernel;
using cvk_kernel_holder = refcounted_holder<cvk_kernel>;

struct cvk_kernel_pipeline_cache {

    cvk_kernel_pipeline_cache(cvk_kernel *kernel, cvk_device *dev) : m_kernel(kernel), m_device(dev) {};

    ~cvk_kernel_pipeline_cache() {
        for (auto &entry : m_entries) {
            vkDestroyPipeline(m_device->vulkan_device(), entry.pipeline, nullptr);
        }
    }

    CHECK_RETURN VkPipeline get_pipeline(uint32_t x, uint32_t y, uint32_t z);

private:

    void insert_pipeline(uint32_t x, uint32_t y, uint32_t z, VkPipeline pipeline) {
        cvk_kernel_pipeline_cache_entry entry = {{x,y,z}, pipeline};
        m_entries.push_back(entry);
    }

    CHECK_RETURN VkPipeline create_and_insert_pipeline(uint32_t x, uint32_t y, uint32_t z);

    std::mutex m_lock;
    cvk_kernel *m_kernel;
    cvk_device *m_device;
    // TODO use map instead?
    std::list<cvk_kernel_pipeline_cache_entry> m_entries;
};

typedef struct _cl_kernel : public api_object {

    const uint32_t MAX_INSTANCES = 16*1024; // FIXME find a better definition

    _cl_kernel(cvk_program *program, const char* name) :
        api_object(program->context()),
        m_program(program),
        m_name(name),
        m_pod_buffer(nullptr),
        m_pod_descriptor_type(VK_DESCRIPTOR_TYPE_MAX_ENUM),
        m_pod_binding(INVALID_POD_BINDING),
        m_descriptor_pool(VK_NULL_HANDLE),
        m_descriptor_set_layout(VK_NULL_HANDLE),
        m_pipeline_layout(VK_NULL_HANDLE),
        m_pipeline_cache(this, m_context->device())
    {
        m_program->retain();
    }

    CHECK_RETURN cl_int init();

    virtual ~_cl_kernel() {
        VkDevice dev = m_context->device()->vulkan_device();
        if (m_descriptor_pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(dev, m_descriptor_pool, nullptr);
        }
        if (m_pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(dev, m_pipeline_layout, nullptr);
        }

        if (m_descriptor_set_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(dev, m_descriptor_set_layout, nullptr);
        }
        m_program->release();
    }

    CHECK_RETURN bool setup_descriptor_set(VkDescriptorSet *ds, std::unique_ptr<cvk_mem> &pod_buffer);

    void free_descriptor_set(VkDescriptorSet ds) {
        auto vkdev = m_context->device()->vulkan_device();
        vkFreeDescriptorSets(vkdev, m_descriptor_pool, 1, &ds);
    }

    CHECK_RETURN cl_int set_arg(cl_uint index, size_t size, const void *value);

    CHECK_RETURN VkPipeline get_pipeline(uint32_t x, uint32_t y, uint32_t z) {
        return m_pipeline_cache.get_pipeline(x, y, z);
    }

    bool has_pod_arguments() const {
        for (auto &arg : m_args) {
            if (arg.is_pod()) {
                return true;
            }
        }
        return false;
    }

    size_t pod_size() const {
        int max_offset = 0;

        for (auto &arg : m_args) {
            max_offset = std::max(max_offset, arg.offset);
        }

        max_offset += MAX_POD_ARGUMENT_SIZE;

        return max_offset;
    }

    const std::string& name() const { return m_name; }
    uint32_t num_args() const { return m_args.size(); }
    VkPipelineLayout pipeline_layout() const { return m_pipeline_layout; }
    cvk_program* program() const { return m_program; }

    cl_ulong local_mem_size() const {
        cl_ulong ret = 0; // FIXME

        return ret;
    }

private:

    const uint32_t INVALID_POD_BINDING = std::numeric_limits<uint32_t>::max();
    const uint32_t MAX_POD_ARGUMENT_SIZE = 1024; // FIXME shouldn't need that
    void build_descriptor_sets_layout_bindings();
    bool allocate_pod_buffer();

    std::mutex m_lock;
    cvk_program *m_program;
    std::string m_name;
    std::unique_ptr<cvk_mem> m_pod_buffer;
    VkDescriptorType m_pod_descriptor_type;
    uint32_t m_pod_binding;
    std::vector<kernel_argument> m_args;
    std::vector<cvk_kernel_arg_storage> m_args_storage;
    std::vector<VkDescriptorSetLayoutBinding> m_layout_bindings;
    VkDescriptorPool m_descriptor_pool;
    VkDescriptorSetLayout m_descriptor_set_layout;
    VkPipelineLayout m_pipeline_layout;
    cvk_kernel_pipeline_cache m_pipeline_cache;
} cvk_kernel;

