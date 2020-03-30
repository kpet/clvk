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
#include <unordered_map>

#include "memory.hpp"
#include "objects.hpp"
#include "program.hpp"

struct cvk_kernel_argument_values;
struct cvk_command_buffer;

struct cvk_kernel : public _cl_kernel, api_object {

    cvk_kernel(cvk_program* program, const char* name)
        : api_object(program->context()), m_program(program),
          m_entry_point(nullptr), m_name(name), m_pod_arg(nullptr) {
        m_program->retain();
    }

    CHECK_RETURN cl_int init();

    virtual ~cvk_kernel() { m_program->release(); }

    CHECK_RETURN bool setup_descriptor_sets(
        VkDescriptorSet* ds,
        std::unique_ptr<cvk_kernel_argument_values>& arg_values);

    void free_descriptor_set(VkDescriptorSet ds) {
        m_entry_point->free_descriptor_set(ds);
    }

    CHECK_RETURN cl_int set_arg(cl_uint index, size_t size, const void* value);
    CHECK_RETURN VkPipeline
    create_pipeline(const cvk_spec_constant_map& spec_constants);

    bool has_pod_arguments() const {
        return m_entry_point->has_pod_arguments();
    }

    bool has_pod_buffer_arguments() const {
        return m_entry_point->has_pod_buffer_arguments();
    }

    const std::string& name() const { return m_name; }
    uint32_t num_args() const { return m_args.size(); }
    uint32_t num_set_layouts() const {
        return m_entry_point->num_set_layouts();
    }
    VkPipelineLayout pipeline_layout() const {
        return m_entry_point->pipeline_layout();
    }
    cvk_program* program() const { return m_program; }

    const std::vector<kernel_argument>& arguments() const { return m_args; }

    kernel_argument_kind arg_kind(int index) const {
        return m_args[index].kind;
    }

    cl_ulong local_mem_size() const;

private:
    std::unique_ptr<cvk_buffer> allocate_pod_buffer();
    std::unique_ptr<std::vector<uint8_t>> allocate_pod_pushconstant_buffer();
    friend cvk_kernel_argument_values;

    std::mutex m_lock;
    cvk_program* m_program;
    cvk_entry_point* m_entry_point;
    std::string m_name;
    const kernel_argument* m_pod_arg;
    std::vector<kernel_argument> m_args;
    std::unique_ptr<cvk_kernel_argument_values> m_argument_values;
};

static inline cvk_kernel* icd_downcast(cl_kernel kernel) {
    return static_cast<cvk_kernel*>(kernel);
}

using cvk_kernel_holder = refcounted_holder<cvk_kernel>;

struct cvk_kernel_argument_values {

    cvk_kernel_argument_values(cvk_kernel* kernel, uint32_t num_resources)
        : m_kernel(kernel), m_pod_buffer(nullptr), m_owns_resources(false),
          m_kernel_resources(num_resources),
          m_local_args_size(m_kernel->num_args(), 0) {}

    cvk_kernel_argument_values(const cvk_kernel_argument_values& other)
        : m_kernel(other.m_kernel), m_pod_buffer(nullptr),
          m_owns_resources(false), m_kernel_resources(other.m_kernel_resources),
          m_local_args_size(other.m_local_args_size) {}

    ~cvk_kernel_argument_values() { release_resources(); }

    static std::unique_ptr<cvk_kernel_argument_values>
    create(cvk_kernel* kernel, uint32_t num_resources) {
        auto val =
            std::make_unique<cvk_kernel_argument_values>(kernel, num_resources);

        if (!val->init()) {
            return nullptr;
        }

        return val;
    }

    static std::unique_ptr<cvk_kernel_argument_values>
    create(const cvk_kernel_argument_values& other) {
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
        if (m_kernel->has_pod_buffer_arguments()) {
            auto buffer = m_kernel->allocate_pod_buffer();
            if (buffer == nullptr) {
                return false;
            }

            m_pod_buffer = std::move(buffer);
        } else if (m_kernel->has_pod_arguments()) {
            auto buffer = m_kernel->allocate_pod_pushconstant_buffer();
            if (buffer == nullptr) {
                return false;
            }

            m_pod_pushconstant_buffer = std::move(buffer);
        }

        return true;
    }

    bool init_copy(const cvk_kernel_argument_values& other) {
        if (m_kernel->has_pod_buffer_arguments()) {
            return other.m_pod_buffer->copy_to(m_pod_buffer.get(), 0, 0,
                                               m_pod_buffer->size());
        } else if (m_kernel->has_pod_arguments()) {
            memcpy(&pod_pushconstant_buffer()[0],
                   &other.pod_pushconstant_buffer()[0],
                   pod_pushconstant_buffer().size());
            return true;
        } else {
            return true;
        }
    }

    cl_int set_arg(const kernel_argument& arg, size_t size, const void* value) {

        if (arg.is_pod_buffer()) {
            if (size != arg.size) {
                return CL_INVALID_ARG_SIZE;
            }

            if (!m_pod_buffer->copy_from(value, arg.offset, size)) {
                return CL_OUT_OF_RESOURCES;
            }
        } else if (arg.is_pod()) {
            if (size != arg.size) {
                return CL_INVALID_ARG_SIZE;
            }

            memcpy(&pod_pushconstant_buffer()[arg.offset], value, size);
        } else if (arg.kind == kernel_argument_kind::local) {
            CVK_ASSERT(value == nullptr);
            m_local_args_size[arg.pos] = size;
            CVK_ASSERT(size % arg.local_elem_size == 0);
            m_specialization_constants[arg.local_spec_id] =
                size / arg.local_elem_size;
        } else {
            // We only expect cl_mem or cl_sampler here
            if (size != sizeof(void*)) {
                return CL_INVALID_ARG_SIZE;
            }
            if (arg.kind == kernel_argument_kind::sampler) {
                auto sampler = *reinterpret_cast<const cl_sampler*>(value);
                m_kernel_resources[arg.binding] = icd_downcast(sampler);
            } else {
                auto mem = *reinterpret_cast<const cl_mem*>(value);
                m_kernel_resources[arg.binding] = icd_downcast(mem);
            }
        }

        return CL_SUCCESS;
    }

    refcounted* get_arg_value(const kernel_argument& arg) {
        return m_kernel_resources[arg.binding];
    }

    VkBuffer pod_vulkan_buffer() const { return m_pod_buffer->vulkan_buffer(); }

    std::vector<uint8_t>& pod_pushconstant_buffer() const {
        return *m_pod_pushconstant_buffer;
    }

    size_t local_arg_size(int pos) const { return m_local_args_size[pos]; }

    const std::unordered_map<uint32_t, uint32_t>&
    specialization_constants() const {
        return m_specialization_constants;
    }

    // Take ownership of resources and retain them.
    void retain_resources() {
        if (!m_owns_resources) {
            m_owns_resources = true;
            for (auto& resource : m_kernel_resources) {
                if (resource)
                    resource->retain();
            }
        }
    }

    // Release all resources owned resources.
    void release_resources() {
        if (m_owns_resources) {
            for (auto& resource : m_kernel_resources) {
                if (resource)
                    resource->release();
            }
        }
    }

private:
    cvk_kernel* m_kernel;
    std::unique_ptr<cvk_buffer> m_pod_buffer;
    std::unique_ptr<std::vector<uint8_t>> m_pod_pushconstant_buffer;
    bool m_owns_resources;
    std::vector<refcounted*> m_kernel_resources;
    std::vector<size_t> m_local_args_size;
    std::unordered_map<uint32_t, uint32_t> m_specialization_constants;
};
