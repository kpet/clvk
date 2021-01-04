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
#include <atomic>
#include <cstdint>
#include <fstream>
#include <map>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

#include "spirv-tools/libspirv.h"
#include "spirv/1.0/spirv.hpp"

#include "memory.hpp"
#include "objects.hpp"

const int SPIR_WORD_SIZE = 4;

enum class kernel_argument_kind
{
    buffer,
    buffer_ubo,
    pod,
    pod_ubo,
    pod_pushconstant,
    ro_image,
    wo_image,
    sampler,
    local,
};

struct kernel_argument {
    std::string name;
    uint32_t pos;
    uint32_t descriptorSet;
    uint32_t binding;
    uint32_t offset;
    uint32_t size;
    kernel_argument_kind kind;
    uint32_t local_spec_id;
    uint32_t local_elem_size;

    bool is_pod() const {
        return (kind == kernel_argument_kind::pod) ||
               (kind == kernel_argument_kind::pod_ubo) ||
               (kind == kernel_argument_kind::pod_pushconstant);
    }

    bool is_pod_buffer() const {
        return (kind == kernel_argument_kind::pod) ||
               (kind == kernel_argument_kind::pod_ubo);
    }
};

struct sampler_desc {
    uint32_t descriptorSet;
    uint32_t binding;
    bool normalized_coords;
    cl_addressing_mode addressing_mode;
    cl_filter_mode filter_mode;
};

enum class pushconstant
{
    global_offset,
    enqueued_local_size,
    global_size,
    region_offset,
    num_workgroups,
    region_group_offset,
};

struct pushconstant_desc {
    uint32_t offset;
    uint32_t size;
};

enum class spec_constant
{
    workgroup_size_x,
    workgroup_size_y,
    workgroup_size_z,
    work_dim,
    global_offset_x,
    global_offset_y,
    global_offset_z,
};

class spir_binary {

    using kernels_arguments_map =
        std::unordered_map<std::string, std::vector<kernel_argument>>;
    using kernels_reqd_work_group_size_map =
        std::unordered_map<std::string, std::array<size_t, 3>>;
    const uint32_t MAGIC = 0x00BEEF00;

public:
    spir_binary(spv_target_env env)
        : m_loaded_from_binary(false), m_target_env(env) {
        m_context = spvContextCreate(env);
    }
    ~spir_binary() { spvContextDestroy(m_context); }
    CHECK_RETURN bool load_spir(const char* fname);
    CHECK_RETURN bool load_spir(std::istream& istream, uint32_t size);
    CHECK_RETURN bool load_descriptor_map();
    void insert_descriptor_map(const spir_binary& other);
    CHECK_RETURN bool save_spir(const char* fname) const;
    CHECK_RETURN bool load(std::istream& istream);
    CHECK_RETURN bool save(std::ostream& ostream) const;
    CHECK_RETURN bool save(const char* fname) const;
    CHECK_RETURN bool read(const unsigned char* src, size_t size);
    CHECK_RETURN bool write(unsigned char* dst) const;
    size_t size() const;
    bool loaded_from_binary() const { return m_loaded_from_binary; }
    size_t spir_size() const { return m_code.size() * sizeof(uint32_t); }
    const uint32_t* spir_data() const { return m_code.data(); }
    void use(std::vector<uint32_t>&& src);
    void set_target_env(spv_target_env env);
    const std::vector<uint32_t>& code() const { return m_code; };
    CHECK_RETURN bool validate() const;
    size_t num_kernels() const { return m_dmaps.size(); }
    const kernels_arguments_map& kernels_arguments() const { return m_dmaps; }
    std::vector<uint32_t>* raw_binary() { return &m_code; }
    const std::vector<sampler_desc>& literal_samplers() {
        return m_literal_samplers;
    }
    const std::array<size_t, 3>&
    required_work_group_size(const std::string& kernel) const {
        return m_reqd_work_group_sizes.at(kernel);
    }
    CHECK_RETURN bool
    get_capabilities(std::vector<spv::Capability>& capabilities) const;
    static constexpr uint32_t MAX_DESCRIPTOR_SETS = 2;

    const std::unordered_map<pushconstant, pushconstant_desc>&
    push_constants() const {
        return m_push_constants;
    }

    const std::unordered_map<spec_constant, uint32_t>& spec_constants() const {
        return m_spec_constants;
    }

    CHECK_RETURN const pushconstant_desc* push_constant(pushconstant pc) const {
        if (m_push_constants.count(pc) != 0) {
            return &m_push_constants.at(pc);
        } else {
            return nullptr;
        }
    }

    void add_kernel(const std::string& name) {
        m_dmaps[name] = {};
        m_reqd_work_group_sizes[name] = {0, 0, 0};
    }

    void add_kernel_argument(const std::string& name, kernel_argument&& arg) {
        m_dmaps[name].push_back(arg);
    }

    void add_spec_constant(spec_constant constant, uint32_t spec_id) {
        m_spec_constants[constant] = spec_id;
    }

    void add_push_constant(pushconstant pc, pushconstant_desc&& desc) {
        m_push_constants[pc] = desc;
    }

    void add_literal_sampler(sampler_desc&& desc) {
        m_literal_samplers.push_back(desc);
    }

    void set_required_work_group_size(const std::string& kernel, uint32_t x,
                                      uint32_t y, uint32_t z) {
        m_reqd_work_group_sizes[kernel] = {x, y, z};
    }

    bool strip_reflection(std::vector<uint32_t>* stripped);

private:
    spv_context m_context;
    std::vector<uint32_t> m_code;
    std::vector<sampler_desc> m_literal_samplers;
    std::unordered_map<pushconstant, pushconstant_desc> m_push_constants;
    std::unordered_map<spec_constant, uint32_t> m_spec_constants;
    kernels_arguments_map m_dmaps;
    kernels_reqd_work_group_size_map m_reqd_work_group_sizes;
    std::string m_dmaps_text;
    bool m_loaded_from_binary;
    spv_target_env m_target_env;
};

enum class build_operation
{
    build,
    compile,
    link
};

using cvk_program_callback = void(CL_CALLBACK*)(cl_program, void*);

using cvk_spec_constant_map = std::map<uint32_t, uint32_t>;

struct cvk_program;

class cvk_entry_point {
public:
    cvk_entry_point(VkDevice dev, cvk_program* program,
                    const std::string& name);

    ~cvk_entry_point() {
        for (auto pipeline : m_pipelines) {
            cvk_info("destroying pipeline %p for kernel %s", pipeline.second,
                     m_name.c_str());
            vkDestroyPipeline(m_device, pipeline.second, nullptr);
        }
        if (m_descriptor_pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(m_device, m_descriptor_pool, nullptr);
        }
        if (m_pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(m_device, m_pipeline_layout, nullptr);
        }
        for (auto layout : m_descriptor_set_layouts) {
            vkDestroyDescriptorSetLayout(m_device, layout, nullptr);
        }
    }

    CHECK_RETURN cl_int init();

    CHECK_RETURN VkPipeline
    create_pipeline(const cvk_spec_constant_map& spec_constants);

    CHECK_RETURN bool allocate_descriptor_sets(VkDescriptorSet* ds);

    void free_descriptor_set(VkDescriptorSet ds) {
        std::lock_guard<std::mutex> lock(m_descriptor_pool_lock);
        vkFreeDescriptorSets(m_device, m_descriptor_pool, 1, &ds);
    }

    uint32_t num_set_layouts() const { return m_descriptor_set_layouts.size(); }

    std::unique_ptr<cvk_buffer> allocate_pod_buffer();

    const std::vector<kernel_argument>& args() const { return m_args; }

    bool has_pod_arguments() const { return m_has_pod_arguments; }

    bool has_pod_buffer_arguments() const { return m_has_pod_buffer_arguments; }

    uint32_t pod_buffer_size() const { return m_pod_buffer_size; }

    uint32_t num_resource_slots() const { return m_num_resource_slots; }

    VkPipelineLayout pipeline_layout() const { return m_pipeline_layout; }

    VkDescriptorType pod_descriptor_type() const {
        return m_pod_descriptor_type;
    }

    cvk_program* program() const { return m_program; }

private:
    const uint32_t MAX_INSTANCES = 16 * 1024; // FIXME find a better definition

    VkDevice m_device;
    cvk_context* m_context;
    cvk_program* m_program;
    std::string m_name;
    VkDescriptorType m_pod_descriptor_type;
    uint32_t m_pod_buffer_size;
    bool m_has_pod_arguments;
    bool m_has_pod_buffer_arguments;
    std::vector<kernel_argument> m_args;
    uint32_t m_num_resource_slots;
    VkDescriptorPool m_descriptor_pool;
    std::vector<VkDescriptorSetLayout> m_descriptor_set_layouts;
    VkPipelineLayout m_pipeline_layout;

    std::mutex m_pipeline_cache_lock;
    std::mutex m_descriptor_pool_lock;

    using binding_stat_map = std::unordered_map<VkDescriptorType, uint32_t>;
    bool build_descriptor_set_layout(
        const std::vector<VkDescriptorSetLayoutBinding>& bindings);
    bool build_descriptor_sets_layout_bindings_for_arguments(
        binding_stat_map& smap, uint32_t& num_resource_slots);
    bool build_descriptor_sets_layout_bindings_for_literal_samplers(
        binding_stat_map& smap);

    // Structures for caching pipelines based on specialization constants
    struct SpecConstantMapHash {
        size_t operator()(const cvk_spec_constant_map& spec_constants) const {
            // TODO: better hash?
            size_t result = 0;
            for (auto& entry : spec_constants) {
                result ^= std::hash<uint32_t>{}(entry.first) * 31;
                result ^= std::hash<uint32_t>{}(entry.second) * 59;
            }
            return result;
        }
    };
    struct SpecConstantMapEqual {
        bool operator()(const cvk_spec_constant_map& lhs,
                        const cvk_spec_constant_map& rhs) const {
            if (lhs.size() != rhs.size())
                return false;
            for (auto& lhs_entry : lhs) {
                if (!rhs.count(lhs_entry.first))
                    return false;
                if (lhs_entry.second != rhs.at(lhs_entry.first))
                    return false;
            }
            return true;
        }
    };
    std::unordered_map<cvk_spec_constant_map, VkPipeline, SpecConstantMapHash,
                       SpecConstantMapEqual>
        m_pipelines;
};

struct cvk_program : public _cl_program, api_object {

    cvk_program(cvk_context* ctx)
        : api_object(ctx), m_num_devices(1U),
          m_binary_type(CL_PROGRAM_BINARY_TYPE_NONE),
          m_shader_module(VK_NULL_HANDLE) {
        m_dev_status[m_context->device()] = CL_BUILD_NONE;
    }

    cvk_program(cvk_context* ctx, const void* il, size_t length)
        : cvk_program(ctx) {
        m_il.resize(length);
        memcpy(m_il.data(), il, length);
    }

    virtual ~cvk_program() {
        if (m_shader_module != VK_NULL_HANDLE) {
            auto vkdev = m_context->device()->vulkan_device();
            vkDestroyShaderModule(vkdev, m_shader_module, nullptr);
        }
        for (auto& s : m_literal_samplers) {
            s->release();
        }
    }

    void append_source(const char* src, size_t len) {
        if (len != 0) {
            m_source.append(src, len);
        } else {
            m_source.append(src);
        }
    }

    const std::string& source() const { return m_source; }

    const std::vector<uint8_t>& il() const { return m_il; }

    uint32_t num_devices() const { return m_num_devices; }

    cl_program_binary_type binary_type(const cvk_device*) const {
        return m_binary_type;
    }

    bool can_be_linked() const {
        auto dev = m_context->device();
        return ((build_status() == CL_BUILD_SUCCESS) &&
                ((binary_type(dev) == CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT) ||
                 (binary_type(dev) == CL_PROGRAM_BINARY_TYPE_LIBRARY)));
    }

    CHECK_RETURN bool build(build_operation operation, cl_uint num_devices,
                            const cl_device_id* device_list,
                            const char* options, cl_uint num_input_programs,
                            const cl_program* input_programs,
                            const char** header_include_names,
                            cvk_program_callback cb, void* data);

    const std::string& build_options() const { return m_build_options; }

    cl_build_status build_status(const cvk_device* device) const {
        return m_dev_status.at(device);
    }

    cl_build_status build_status() const {
        for (auto& dev_st : m_dev_status) {
            if (dev_st.second != CL_BUILD_SUCCESS) {
                return dev_st.second;
            }
        }

        return CL_BUILD_SUCCESS;
    }

    const std::string& build_log(const cvk_device* device) const {
        UNUSED(device); // TODO support per-device build log
        return m_build_log;
    }

    std::vector<const cvk_device*> devices() const {
        std::vector<const cvk_device*> ret;

        for (auto& dev_st : m_dev_status) {
            ret.push_back(dev_st.first);
        }

        return ret;
    }

    VkShaderModule shader_module() const { return m_shader_module; }

    void wait_for_operation() {
        CVK_ASSERT(m_thread->joinable());
        m_thread->join();
    }

    void complete_operation(cvk_device* device, cl_build_status status) {
        m_dev_status[device] = status;
        if (m_operation_callback != nullptr) {
            m_operation_callback(this, m_operation_callback_data);
        }
        release();
    }

    unsigned num_kernels() const { return m_binary.num_kernels(); }
    bool loaded_from_binary() const { return m_binary.loaded_from_binary(); }

    const std::vector<kernel_argument>* args_for_kernel(std::string& name) {
        auto const& args = m_binary.kernels_arguments().find(name);
        if (args != m_binary.kernels_arguments().end()) {
            return &args->second;
        } else {
            return nullptr;
        }
    }

    CHECK_RETURN bool read(const unsigned char* src, size_t size) {
        bool success = m_binary.read(src, size);
        if (success) {
            // TODO support loading other program types
            m_binary_type = CL_PROGRAM_BINARY_TYPE_EXECUTABLE;
        }
        return success;
    }

    CHECK_RETURN bool write(unsigned char* dst) const {
        return m_binary.write(dst);
    }

    size_t binary_size() const { return m_binary.size(); }

    std::vector<const char*> kernel_names() const {
        std::vector<const char*> ret;
        for (auto& kname_args : m_binary.kernels_arguments()) {
            ret.push_back(kname_args.first.c_str());
        }
        return ret;
    }

    const std::vector<sampler_desc>& literal_sampler_descs() {
        return m_binary.literal_samplers();
    }

    const std::vector<cvk_sampler_holder>& literal_samplers() {
        return m_literal_samplers;
    }

    const VkPushConstantRange& push_constant_range() const {
        return m_push_constant_range;
    }

    CHECK_RETURN const pushconstant_desc* push_constant(pushconstant pc) const {
        return m_binary.push_constant(pc);
    }

    CHECK_RETURN const std::unordered_map<spec_constant, uint32_t>&
    spec_constants() const {
        return m_binary.spec_constants();
    }

    const std::array<size_t, 3>&
    required_work_group_size(const std::string& kernel) const {
        return m_binary.required_work_group_size(kernel);
    }

    CHECK_RETURN cvk_entry_point* get_entry_point(std::string& name,
                                                  cl_int* errcode_ret);

private:
    void do_build();
    CHECK_RETURN cl_build_status compile_source(const cvk_device* device);
    CHECK_RETURN cl_build_status link();
    void prepare_push_constant_range();

    /// Check if all of the capabilities required by the SPIR-V module are
    /// supported by `device`.
    CHECK_RETURN bool check_capabilities(const cvk_device* device) const;

    uint32_t m_num_devices;
    cl_uint m_num_input_programs;
    std::vector<const cvk_program*> m_input_programs;
    std::vector<const char*> m_header_include_names;
    build_operation m_operation;
    cl_program_binary_type m_binary_type;
    cvk_program_callback m_operation_callback;
    void* m_operation_callback_data;
    std::mutex m_lock;
    std::unique_ptr<std::thread> m_thread;
    std::string m_source;
    std::vector<uint8_t> m_il;
    VkShaderModule m_shader_module;
    std::unordered_map<const cvk_device*, std::atomic<cl_build_status>>
        m_dev_status;
    std::string m_build_options;
    spir_binary m_binary{SPV_ENV_VULKAN_1_0};
    std::string m_build_log;
    std::vector<cvk_sampler_holder> m_literal_samplers;
    VkPushConstantRange m_push_constant_range;
    std::unordered_map<std::string, std::unique_ptr<cvk_entry_point>>
        m_entry_points;
    std::vector<uint32_t> m_stripped_binary;
};

static inline cvk_program* icd_downcast(cl_program program) {
    return static_cast<cvk_program*>(program);
}

using cvk_program_holder = refcounted_holder<cvk_program>;
