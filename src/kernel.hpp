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

struct cvk_kernel_argument_values;

typedef struct _cl_kernel : public api_object {

    const uint32_t MAX_INSTANCES = 16*1024; // FIXME find a better definition

    _cl_kernel(cvk_program *program, const char* name) :
        api_object(program->context()),
        m_program(program),
        m_name(name),
        m_pod_descriptor_type(VK_DESCRIPTOR_TYPE_MAX_ENUM),
        m_pod_binding(INVALID_POD_BINDING),
        m_pod_buffer_size(0u),
        m_has_pod_arguments(false),
        m_descriptor_pool(VK_NULL_HANDLE),
        m_descriptor_set_layout(VK_NULL_HANDLE),
        m_pipeline_layout(VK_NULL_HANDLE),
        m_pipeline_cache(VK_NULL_HANDLE)
    {
        m_program->retain();
    }

    CHECK_RETURN cl_int init();

    virtual ~_cl_kernel() {
        VkDevice dev = m_context->device()->vulkan_device();
        if (m_pipeline_cache != VK_NULL_HANDLE) {
            vkDestroyPipelineCache(dev, m_pipeline_cache, nullptr);
        }
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

    CHECK_RETURN bool setup_descriptor_set(
        VkDescriptorSet *ds, std::unique_ptr<cvk_kernel_argument_values> &arg_values);

    void free_descriptor_set(VkDescriptorSet ds) {
        auto vkdev = m_context->device()->vulkan_device();
        vkFreeDescriptorSets(vkdev, m_descriptor_pool, 1, &ds);
    }

    CHECK_RETURN cl_int set_arg(cl_uint index, size_t size, const void *value);
    CHECK_RETURN VkPipeline create_pipeline(const VkSpecializationInfo &info);

    bool has_pod_arguments() const { return m_has_pod_arguments; }
    const std::string& name() const { return m_name; }
    uint32_t num_args() const { return m_args.size(); }
    uint32_t num_bindings() const { return m_layout_bindings.size(); }
    VkPipelineLayout pipeline_layout() const { return m_pipeline_layout; }
    cvk_program* program() const { return m_program; }

    kernel_argument_kind arg_kind(int index) const {
        return m_args[index].kind;
    }

    cl_ulong local_mem_size() const;

private:

    const uint32_t INVALID_POD_BINDING = std::numeric_limits<uint32_t>::max();
    void build_descriptor_sets_layout_bindings();
    std::unique_ptr<cvk_buffer> allocate_pod_buffer();
    friend cvk_kernel_argument_values;

    std::mutex m_lock;
    cvk_program *m_program;
    std::string m_name;
    VkDescriptorType m_pod_descriptor_type;
    uint32_t m_pod_binding;
    uint32_t m_pod_buffer_size;
    bool m_has_pod_arguments;
    std::vector<kernel_argument> m_args;
    std::unique_ptr<cvk_kernel_argument_values> m_argument_values;
    std::vector<VkDescriptorSetLayoutBinding> m_layout_bindings;
    VkDescriptorPool m_descriptor_pool;
    VkDescriptorSetLayout m_descriptor_set_layout;
    VkPipelineLayout m_pipeline_layout;
    VkPipelineCache m_pipeline_cache;
} cvk_kernel;

using cvk_kernel_holder = refcounted_holder<cvk_kernel>;

struct cvk_kernel_argument_values {

    cvk_kernel_argument_values(cvk_kernel *kernel) :
        m_kernel(kernel),
        m_pod_buffer(nullptr),
        m_kernel_resources(m_kernel->num_bindings()),
        m_local_args_size(m_kernel->num_args(), 0) {}

    cvk_kernel_argument_values(const cvk_kernel_argument_values &other) :
        m_kernel(other.m_kernel),
        m_pod_buffer(nullptr),
        m_kernel_resources(other.m_kernel_resources),
        m_local_args_size(other.m_local_args_size) {}

    static std::unique_ptr<cvk_kernel_argument_values> create(cvk_kernel *kernel) {
        auto val = std::make_unique<cvk_kernel_argument_values>(kernel);

        if (!val->init()) {
            return nullptr;
        }

        return val;
    }

    static std::unique_ptr<cvk_kernel_argument_values> create(const cvk_kernel_argument_values &other) {
        auto val = std::make_unique<cvk_kernel_argument_values>(other);

        if (!val->init()) {
            return nullptr;
        }

        if (!val->init_copy(other)) {
            return nullptr;
        }

        return val;
    }

    bool init() {
        if (m_kernel->has_pod_arguments()) {
            auto buffer = m_kernel->allocate_pod_buffer();
            if (buffer == nullptr) {
                return false;
            }

            m_pod_buffer = std::move(buffer);
        }

        return true;
    }

    bool init_copy(const cvk_kernel_argument_values &other) {
        if (m_kernel->has_pod_arguments()) {
            return other.m_pod_buffer->copy_to(m_pod_buffer.get(), 0, 0, m_pod_buffer->size());
        } else {
            return true;
        }
    }

    cl_int set_arg(const kernel_argument& arg, size_t size, const void *value) {

        if (arg.is_pod()) {
            if (size != arg.size) {
                return CL_INVALID_ARG_SIZE;
            }

            if (!m_pod_buffer->copy_from(value, arg.offset, size)) {
                return CL_OUT_OF_RESOURCES;
            }
        } else if (arg.kind == kernel_argument_kind::local) {
            CVK_ASSERT(value == nullptr);
            m_local_args_size[arg.pos] = size;
            CVK_ASSERT(size % arg.local_elem_size == 0);
            m_specialization_constants[arg.local_spec_id] = size / arg.local_elem_size;
        } else {
            // We only expect cl_mem or cl_sampler here
            if (size != sizeof(void*)) {
                return CL_INVALID_ARG_SIZE;
            }

            auto refc = *reinterpret_cast<refcounted*const*>(value);

            m_kernel_resources[arg.binding].reset(refc);
        }

        return CL_SUCCESS;
    }

    refcounted* get_arg_value(const kernel_argument& arg) {
        return m_kernel_resources[arg.binding];
    }

    VkBuffer pod_vulkan_buffer() const {
        return m_pod_buffer->vulkan_buffer();
    }

    size_t local_arg_size(int pos) const { return m_local_args_size[pos]; }

    const std::unordered_map<uint32_t, uint32_t>& specialization_constants() const {
        return m_specialization_constants;
    }

private:
    cvk_kernel *m_kernel;
    std::unique_ptr<cvk_buffer> m_pod_buffer;
    std::vector<refcounted_holder<refcounted>> m_kernel_resources;
    std::vector<size_t> m_local_args_size;
    std::unordered_map<uint32_t, uint32_t> m_specialization_constants;
};
