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

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <vector>

#include "memory.hpp"
#include "objects.hpp"
#include "program.hpp"

struct cvk_kernel_argument_values;

struct cvk_kernel : public _cl_kernel, api_object<object_magic::kernel> {

    cvk_kernel(cvk_program* program, const char* name)
        : api_object(program->context()), m_program(program),
          m_entry_point(nullptr), m_name(name), m_image_metadata(nullptr) {}

    CHECK_RETURN cl_int init();
    std::unique_ptr<cvk_kernel> clone(cl_int* errcode_ret) const;

    virtual ~cvk_kernel() { m_argument_values.reset(); }

    std::shared_ptr<cvk_kernel_argument_values> argument_values() const {
        return m_argument_values;
    }

    const kernel_image_metadata_map* get_image_metadata() const {
        return m_image_metadata;
    }

    void set_image_metadata(cl_uint index, const void* image);

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
    const std::string& attributes() const {
        return m_program->kernel_attributes(m_name);
    }
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

    size_t max_work_group_size(const cvk_device* device) const {
        return device->max_work_group_size();
    }

    size_t max_sub_group_size_for_ndrange(const cvk_device* device) const {
        return device->sub_group_size();
    }

    size_t
    sub_group_count_for_ndrange(const cvk_device* device,
                                const std::array<uint32_t, 3>& lws) const {
        uint32_t work_items_per_work_group = lws[0] * lws[1] * lws[2];
        return ceil_div(work_items_per_work_group, device->sub_group_size());
    }
    std::array<size_t, 3>
    local_size_for_sub_group_count(const cvk_device* device,
                                   size_t num_sub_groups) const {
        std::array<size_t, 3> ret = {1, 1, 1};
        size_t wgs = num_sub_groups * device->sub_group_size();
        if (wgs > max_work_group_size(device)) {
            ret = {0, 0, 0};
        } else {
            ret[0] = wgs;
        }
        return ret;
    }

    size_t max_num_sub_groups(const cvk_device* device) const {
        return ceil_div(max_work_group_size(device),
                        static_cast<size_t>(device->sub_group_size()));
    }

    const std::array<uint32_t, 3>& required_work_group_size() const {
        return m_program->required_work_group_size(m_name);
    }

    bool args_valid() const;

    bool has_extended_arg_info(cl_uint arg_index) const {
        return m_args.at(arg_index).info.extended_valid;
    }

    const std::string arg_name(cl_uint arg_index) const {
        return m_args.at(arg_index).info.name;
    }

    const std::string arg_type_name(cl_uint arg_index) const {
        return m_args.at(arg_index).info.type_name;
    }

    cl_kernel_arg_address_qualifier
    arg_address_qualifier(cl_uint arg_index) const {
        return m_args.at(arg_index).info.address_qualifier;
    }

    cl_kernel_arg_access_qualifier
    arg_access_qualifier(cl_uint arg_index) const {
        return m_args.at(arg_index).info.access_qualifier;
    }

    cl_kernel_arg_type_qualifier arg_type_qualifier(cl_uint arg_index) const {
        return m_args.at(arg_index).info.type_qualifier;
    }

private:
    friend cvk_kernel_argument_values;

    std::mutex m_lock;
    cvk_program_holder m_program;
    cvk_entry_point* m_entry_point;
    std::string m_name;
    std::vector<kernel_argument> m_args;
    std::shared_ptr<cvk_kernel_argument_values> m_argument_values;
    const kernel_image_metadata_map* m_image_metadata;
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
          m_args_set(m_args.size(), false), m_descriptor_sets{VK_NULL_HANDLE},
          m_descriptor_sets_refcount(0) {}

    cvk_kernel_argument_values(const cvk_kernel_argument_values& other)
        : m_entry_point(other.m_entry_point), m_is_enqueued(false),
          m_args(m_entry_point->args()), m_pod_arg(nullptr),
          m_kernel_resources(other.m_kernel_resources),
          m_local_args_size(other.m_local_args_size),
          m_specialization_constants(other.m_specialization_constants),
          m_args_set(other.m_args_set), m_descriptor_sets{VK_NULL_HANDLE},
          m_descriptor_sets_refcount(0) {}

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
        }

        if (m_entry_point->has_pod_arguments() ||
            m_entry_point->has_image_metadata()) {
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

    void set_pod_data(uint32_t offset, size_t size, const void* value) {
        memcpy(&pod_data()[offset], value, size);
    }

    cl_int set_arg(const kernel_argument& arg, size_t size, const void* value) {

        if (arg.is_pod_pointer()) {
            auto mem = *reinterpret_cast<const cl_mem*>(value);
            if (mem == NULL) {
                // OpenCL permits cl_mem to be NULL
                uint64_t null = 0;
                set_pod_data(arg.offset, arg.size, &null);
            } else {
                auto mem_downcast = icd_downcast(mem);
                auto buff = reinterpret_cast<const cvk_buffer*>(mem_downcast);
                auto dev_addr = buff->device_address();
                set_pod_data(arg.offset, arg.size, &dev_addr);
            }
        } else if (arg.is_pod()) {
            // If the argument is a vec3, OpenCL requires to call clSetKernelArg
            // with a size of 4 times the element size. But clspv arg size is
            // only 3 times the element size. When size and arg.size do not
            // match, make sure that we are not in this unusual case.
            if (size != arg.size &&
                !(arg.is_vec3() && (size == arg.size * 4 / 3))) {
                return CL_INVALID_ARG_SIZE;
            }

            set_pod_data(arg.offset, arg.size, value);
        } else if (arg.kind == kernel_argument_kind::local) {
            CVK_ASSERT(value == nullptr);
            m_local_args_size[arg.pos] = size;
            CVK_ASSERT(size % arg.local_elem_size == 0);
            m_specialization_constants[arg.local_spec_id] =
                size / arg.local_elem_size;
        } else if (!arg.is_unused()) {
            // We only expect cl_mem or cl_sampler here
            if (size != sizeof(void*)) {
                return CL_INVALID_ARG_SIZE;
            }
            if (arg.kind == kernel_argument_kind::sampler) {
                auto apisampler = *reinterpret_cast<const cl_sampler*>(value);
                if (apisampler == nullptr) {
                    return CL_INVALID_SAMPLER;
                }
                auto sampler = icd_downcast(apisampler);
                if (!sampler->is_valid()) {
                    return CL_INVALID_SAMPLER;
                }

                m_kernel_resources[arg.binding] = sampler;
            } else {
                auto apimem = *reinterpret_cast<const cl_mem*>(value);
                if (apimem == nullptr) {
                    return CL_INVALID_MEM_OBJECT;
                }
                auto mem = icd_downcast(apimem);
                if (!mem->is_valid()) {
                    return CL_INVALID_MEM_OBJECT;
                }
                m_kernel_resources[arg.binding] = mem;
            }
        }

        m_args_set[arg.pos] = true;
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
        std::lock_guard<std::mutex> lock(m_lock);
        m_descriptor_sets_refcount++;
    }

    // Release all resources owned resources.
    void release_resources() {
        for (auto& resource : m_kernel_resources) {
            if (resource)
                resource->release();
        }
        std::lock_guard<std::mutex> lock(m_lock);
        if (--m_descriptor_sets_refcount == 0) {
            m_is_enqueued = false;
            for (auto& ds : m_descriptor_sets) {
                if (ds != VK_NULL_HANDLE) {
                    m_entry_point->free_descriptor_set(ds);
                    ds = VK_NULL_HANDLE;
                }
            }
        }
    }

    const std::vector<cvk_mem*> memory_objects() const {
        std::vector<cvk_mem*> mems;
        mems.reserve(m_args.size());
        for (auto& arg : m_args) {
            if (arg.is_mem_object_backed()) {
                auto mem =
                    static_cast<cvk_mem*>(m_kernel_resources[arg.binding]);
                mems.push_back(mem);
            }
        }
        return mems;
    }

    bool args_valid() const {
        return std::all_of(m_args_set.cbegin(), m_args_set.cend(),
                           [](bool b) { return b; });
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
    std::vector<bool> m_args_set;

    std::unique_ptr<cvk_buffer> m_pod_buffer;
    std::array<VkDescriptorSet, spir_binary::MAX_DESCRIPTOR_SETS>
        m_descriptor_sets;
    uint32_t m_descriptor_sets_refcount;
};
