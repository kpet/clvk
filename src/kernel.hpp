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

struct cvk_kernel : public _cl_kernel, api_object {

    cvk_kernel(cvk_program* program, const char* name)
        : api_object(program->context()), m_program(program),
          m_entry_point(nullptr), m_name(name) {}

    CHECK_RETURN cl_int init();
    std::unique_ptr<cvk_kernel> clone(cl_int* errcode_ret) const;

    virtual ~cvk_kernel() {
        m_argument_values.reset();
    }

    std::shared_ptr<cvk_kernel_argument_values> argument_values() const {
        return m_argument_values;
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

    const std::array<size_t, 3>& required_work_group_size() const {
        return m_program->required_work_group_size(m_name);
    }

private:
    friend cvk_kernel_argument_values;

    std::mutex m_lock;
    cvk_program_holder m_program;
    cvk_entry_point* m_entry_point;
    std::string m_name;
    std::vector<kernel_argument> m_args;
    std::shared_ptr<cvk_kernel_argument_values> m_argument_values;
};

static inline cvk_kernel* icd_downcast(cl_kernel kernel) {
    return static_cast<cvk_kernel*>(kernel);
}

using cvk_kernel_holder = refcounted_holder<cvk_kernel>;

struct cvk_kernel_argument_values {

    cvk_kernel_argument_values(cvk_entry_point* entry_point)
        : m_entry_point(entry_point), m_is_enqueued(false),
          m_args(m_entry_point->args()), m_pod_arg(nullptr),
          m_kernel_resources(m_entry_point->num_resource_slots()),
          m_local_args_size(m_entry_point->args().size(), 0),
          m_descriptor_sets{VK_NULL_HANDLE} {}

    cvk_kernel_argument_values(const cvk_kernel_argument_values& other)
        : m_entry_point(other.m_entry_point), m_is_enqueued(false),
          m_args(m_entry_point->args()), m_pod_arg(nullptr),
          m_kernel_resources(other.m_kernel_resources),
          m_local_args_size(other.m_local_args_size), m_descriptor_sets{
                                                          VK_NULL_HANDLE} {}

    ~cvk_kernel_argument_values() {
        for (auto ds : m_descriptor_sets) {
            if (ds != VK_NULL_HANDLE) {
                m_entry_point->free_descriptor_set(ds);
            }
        }
    }

    static std::shared_ptr<cvk_kernel_argument_values>
    create(cvk_entry_point* entry_point) {
        auto val = std::make_shared<cvk_kernel_argument_values>(entry_point);

        if (!val->init()) {
            return nullptr;
        }

        return val;
    }

    static std::shared_ptr<cvk_kernel_argument_values>
    create(const cvk_kernel_argument_values& other) {
        auto val = std::make_shared<cvk_kernel_argument_values>(other);

        if (!val->init()) {
            return nullptr;
        }

        if (!val->init_copy(other)) {
            return nullptr;
        }

        return val;
    }

    bool init() {
        // Init POD arguments
        if (m_entry_point->has_pod_arguments()) {
            // Find out POD binding
            for (auto& arg : m_args) {
                if (arg.is_pod()) {
                    m_pod_arg = &arg;
                    break;
                }
            }

            if (m_pod_arg == nullptr) {
                return CL_INVALID_PROGRAM;
            }

            // TODO(#101): host out-of-memory errors are currently unhandled.
            auto buffer = std::make_unique<std::vector<uint8_t>>(
                m_entry_point->pod_buffer_size());
            m_pod_data = std::move(buffer);
        }

        return true;
    }

    bool init_copy(const cvk_kernel_argument_values& other) {
        if (m_entry_point->has_pod_arguments()) {
            memcpy(&pod_data()[0], &other.pod_data()[0], pod_data().size());
            return true;
        } else {
            return true;
        }
    }

    cl_int set_arg(const kernel_argument& arg, size_t size, const void* value) {

        if (arg.is_pod()) {
            if (size != arg.size) {
                return CL_INVALID_ARG_SIZE;
            }

            memcpy(&pod_data()[arg.offset], value, size);
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

    bool is_enqueued() const { return m_is_enqueued; }

    const std::vector<uint8_t>& pod_data() const { return *m_pod_data; }
    std::vector<uint8_t>& pod_data() { return *m_pod_data; }

    size_t local_arg_size(int pos) const { return m_local_args_size[pos]; }

    const std::unordered_map<uint32_t, uint32_t>&
    specialization_constants() const {
        return m_specialization_constants;
    }

    CHECK_RETURN bool setup_descriptor_sets();

    VkDescriptorSet* descriptor_sets() { return m_descriptor_sets.data(); }

    // Take ownership of resources and retain them.
    void retain_resources() {
        for (auto& resource : m_kernel_resources) {
            if (resource)
                resource->retain();
        }
    }

    // Release all resources owned resources.
    void release_resources() {
        for (auto& resource : m_kernel_resources) {
            if (resource)
                resource->release();
        }
    }

private:
    bool create_pod_buffer() {
        CVK_ASSERT(m_pod_data->size() >= m_entry_point->pod_buffer_size());

        // Create POD buffer and copy data to it
        m_pod_buffer = m_entry_point->allocate_pod_buffer();
        if (m_pod_buffer == nullptr) {
            return false;
        }
        return m_pod_buffer->copy_from(m_pod_data->data(), 0,
                                       m_entry_point->pod_buffer_size());
    }

    std::mutex m_lock;
    cvk_entry_point* m_entry_point;
    std::unique_ptr<std::vector<uint8_t>> m_pod_data;
    bool m_is_enqueued;
    const std::vector<kernel_argument>& m_args;
    const kernel_argument* m_pod_arg;
    std::vector<refcounted*> m_kernel_resources;
    std::vector<size_t> m_local_args_size;
    std::unordered_map<uint32_t, uint32_t> m_specialization_constants;

    std::unique_ptr<cvk_buffer> m_pod_buffer;
    std::array<VkDescriptorSet, spir_binary::MAX_DESCRIPTOR_SETS>
        m_descriptor_sets;
};
