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
#include <climits>
#include <cstdint>
#include <fstream>
#include <map>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

#include "spirv-tools/libspirv.h"
#include "spirv/1.0/spirv.hpp"

#include "config.hpp"
#include "init.hpp"
#include "log.hpp"
#include "memory.hpp"
#include "objects.hpp"
#include "printf.hpp"

const int SPIR_WORD_SIZE = 4;

enum class kernel_argument_kind
{
    buffer,
    buffer_ubo,
    pod,
    pod_ubo,
    pod_pushconstant,
    pointer_ubo,
    pointer_pushconstant,
    sampled_image,
    storage_image,
    storage_texel_buffer,
    uniform_texel_buffer,
    sampler,
    local,
    unused,
};

struct kernel_argument_info {
    std::string name;
    bool extended_valid = false;
    std::string type_name;
    uint32_t address_qualifier;
    uint32_t access_qualifier;
    uint32_t type_qualifier;

    bool is_vec3() const {
        return extended_valid && type_name[type_name.length() - 1] == '3';
    }
};

struct kernel_argument {
    kernel_argument_info info;
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
               (kind == kernel_argument_kind::pod_pushconstant) ||
               (kind == kernel_argument_kind::pointer_ubo) ||
               (kind == kernel_argument_kind::pointer_pushconstant);
    }

    bool is_pod_buffer() const {
        return (kind == kernel_argument_kind::pod) ||
               (kind == kernel_argument_kind::pod_ubo) ||
               (kind == kernel_argument_kind::pointer_ubo);
    }

    bool is_pod_pointer() const {
        return (kind == kernel_argument_kind::pointer_pushconstant) ||
               (kind == kernel_argument_kind::pointer_ubo);
    }

    bool is_vec3() const { return info.is_vec3(); }

    bool is_mem_object_backed() const {
        return (kind == kernel_argument_kind::buffer) ||
               (kind == kernel_argument_kind::buffer_ubo) ||
               (kind == kernel_argument_kind::sampled_image) ||
               (kind == kernel_argument_kind::storage_image) ||
               (kind == kernel_argument_kind::storage_texel_buffer) ||
               (kind == kernel_argument_kind::uniform_texel_buffer);
    }

    bool is_unused() const { return kind == kernel_argument_kind::unused; }
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
    image_metadata,
    module_constants_pointer,
    printf_buffer_pointer,
    normalized_sampler_mask,
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
    subgroup_max_size,
};

enum class module_buffer_type
{
    storage_buffer,
    pointer_push_constant,
};

struct user_spec_constant_data {
    std::string type;
    uint32_t size;
    bool set;
    union {
        uint8_t i8;
        uint16_t i16;
        uint32_t i32;
        uint64_t i64;
    } data;

    user_spec_constant_data(std::string type, uint32_t size)
        : type(type), size(size), set(false) {}

    void init_data(size_t size, const void* value) {
        if (type == "i8" || type == "i1") {
            std::memcpy(&data.i8, value, size);
        } else if (type == "i16" || type == "f16") {
            std::memcpy(&data.i16, value, size);
        } else if (type == "i32" || type == "f32") {
            std::memcpy(&data.i32, value, size);
        } else if (type == "i64" || type == "f64") {
            std::memcpy(&data.i64, value, size);
        } else {
            CVK_ASSERT(false && "Unexpected specialisation constant type");
        }
        set = true;
    }
};

struct constant_data_buffer_info {
    module_buffer_type type;
    uint32_t set;
    uint32_t binding;
    uint32_t pc_offset;
    std::vector<char> data;
};

struct printf_buffer_desc_info {
    module_buffer_type type;
    uint32_t set;
    uint32_t binding;
    uint32_t pc_offset;
    uint32_t size = 0;
};

struct spirv_validation_options {
    bool uniform_buffer_std_layout = false;
};

struct image_metadata {
    image_metadata() : order_offset(UINT_MAX), data_type_offset(UINT_MAX) {}
    uint32_t order_offset;
    uint32_t data_type_offset;
    void set_order(uint32_t order) { order_offset = order; }
    void set_data_type(uint32_t data_type) { data_type_offset = data_type; }
    bool has_valid_order() const { return order_offset != UINT_MAX; }
    bool has_valid_data_type() const { return data_type_offset != UINT_MAX; }
};

using kernel_image_metadata_map =
    std::unordered_map<uint32_t, struct image_metadata>;
using image_metadata_map =
    std::unordered_map<std::string, kernel_image_metadata_map>;

using kernel_sampler_metadata_map = std::unordered_map<uint32_t, uint32_t>;
using sampler_metadata_map =
    std::unordered_map<std::string, kernel_sampler_metadata_map>;

class spir_binary {

    using kernels_arguments_map =
        std::unordered_map<std::string, std::vector<kernel_argument>>;
    using kernels_reqd_work_group_size_map =
        std::unordered_map<std::string, std::array<uint32_t, 3>>;
    using kernels_flags_map = std::unordered_map<std::string, uint32_t>;

public:
    spir_binary(spv_target_env env)
        : m_loaded_from_binary(false), m_target_env(env) {
        m_context = spvContextCreate(env);
    }
    ~spir_binary() { spvContextDestroy(m_context); }
    CHECK_RETURN bool load(const char* fname);
    CHECK_RETURN bool load(std::istream& istream, uint32_t size);
    CHECK_RETURN bool load_descriptor_map();
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
    CHECK_RETURN bool validate(const spirv_validation_options&) const;
    size_t num_kernels() const { return m_dmaps.size(); }
    const kernels_arguments_map& kernels_arguments() const { return m_dmaps; }
    const sampler_metadata_map& sampler_metadata() const {
        return m_sampler_metadata;
    }
    const image_metadata_map& image_metadata() const {
        return m_image_metadata;
    }
    std::vector<uint32_t>* raw_binary() { return &m_code; }
    const std::vector<sampler_desc>& literal_samplers() {
        return m_literal_samplers;
    }
    const std::array<uint32_t, 3>&
    required_work_group_size(const std::string& kernel) const {
        return m_reqd_work_group_sizes.at(kernel);
    }
    CHECK_RETURN bool
    get_capabilities(std::vector<spv::Capability>& capabilities) const;
    static constexpr uint32_t MAX_DESCRIPTOR_SETS = 3;

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

    const printf_descriptor_map_t& printf_descriptors() const {
        return m_printf_descriptors;
    }

    void add_kernel(const std::string& name, uint32_t num_args,
                    const std::string& attributes, uint32_t flags) {
        m_flags[name] = flags;
        auto& args = m_dmaps[name];
        kernel_argument unused = {
            {}, 0, 0, 0, 0, 0, kernel_argument_kind::unused, 0, 0};
        // Generate a placeholder for each argument in the kernel.
        args.resize(num_args, unused);
        uint32_t pos = 0;
        // Assign the argument ordinals. Any used argument will overwrite these,
        // but they are necessary for unused arguments.
        for (auto& arg : args) {
            arg.pos = pos++;
        }
        m_reqd_work_group_sizes[name] = {0, 0, 0};
        m_kernels_attributes[name] = attributes;
    }

    const std::unordered_map<std::string, std::string>&
    kernels_attributes() const {
        return m_kernels_attributes;
    }

    void add_kernel_argument(const std::string& name, kernel_argument&& arg) {
        // Overwrite the placeholder argument.
        m_dmaps[name][arg.pos] = std::move(arg);
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

    const constant_data_buffer_info* constant_data_buffer() const {
        return m_constant_data_buffer.get();
    }

    void set_constant_data_buffer(const constant_data_buffer_info& info) {
        m_constant_data_buffer.reset(new constant_data_buffer_info(info));
    }

    void add_sampler_metadata(const std::string& name, uint32_t ordinal,
                              uint32_t offset) {
        m_sampler_metadata[name][ordinal] = offset;
    }

    void add_image_channel_order_metadata(const std::string& name,
                                          uint32_t ordinal, uint32_t offset) {
        m_image_metadata[name][ordinal].set_order(offset);
    }
    void add_image_channel_data_type_metadata(const std::string& name,
                                              uint32_t ordinal,
                                              uint32_t offset) {
        m_image_metadata[name][ordinal].set_data_type(offset);
    }

    void add_printf_descriptor(printf_descriptor&& desc) {
        m_printf_descriptors[desc.printf_id] = desc;
    }

    void set_printf_buffer_info(const printf_buffer_desc_info& info) {
        m_printf_buffer_info = info;
    }

    const printf_buffer_desc_info& printf_buffer_info() const {
        return m_printf_buffer_info;
    }

    const printf_descriptor_map_t& get_printf_descriptors() const {
        return m_printf_descriptors;
    }

    const kernels_flags_map& kernels_flags() const { return m_flags; }

private:
    spv_context m_context;
    std::vector<uint32_t> m_code;
    std::vector<sampler_desc> m_literal_samplers;
    std::unordered_map<pushconstant, pushconstant_desc> m_push_constants;
    std::unordered_map<spec_constant, uint32_t> m_spec_constants;
    sampler_metadata_map m_sampler_metadata;
    image_metadata_map m_image_metadata;
    std::unordered_map<uint32_t, printf_descriptor> m_printf_descriptors;
    printf_buffer_desc_info m_printf_buffer_info;
    std::unique_ptr<constant_data_buffer_info> m_constant_data_buffer;
    kernels_arguments_map m_dmaps;
    kernels_reqd_work_group_size_map m_reqd_work_group_sizes;
    std::unordered_map<std::string, std::string> m_kernels_attributes;
    kernels_flags_map m_flags;
    bool m_loaded_from_binary;
    spv_target_env m_target_env;
};

enum class build_operation
{
    build,
    build_binary,
    compile,
    link
};

using cvk_program_callback = void(CL_CALLBACK*)(cl_program, void*);

using cvk_spec_constant_map = std::map<uint32_t, uint32_t>;

struct cvk_program;

class cvk_entry_point {
public:
    cvk_entry_point(cvk_device* dev, cvk_program* program,
                    const std::string& name);

    ~cvk_entry_point() {
        VkDevice vkdev = m_device->vulkan_device();
        for (auto pipeline : m_pipelines) {
            cvk_info("destroying pipeline %p for kernel %s", pipeline.second,
                     m_name.c_str());
            vkDestroyPipeline(vkdev, pipeline.second, nullptr);
        }
        if (m_descriptor_pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(vkdev, m_descriptor_pool, nullptr);
        }
        if (m_pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(vkdev, m_pipeline_layout, nullptr);
        }
        for (auto layout : m_descriptor_set_layouts) {
            vkDestroyDescriptorSetLayout(vkdev, layout, nullptr);
        }
    }

    CHECK_RETURN cl_int init();

    CHECK_RETURN VkPipeline
    create_pipeline(const cvk_spec_constant_map& spec_constants);

    CHECK_RETURN bool allocate_descriptor_sets(VkDescriptorSet* ds);

    void free_descriptor_set(VkDescriptorSet ds) {
        TRACE_FUNCTION();
        std::lock_guard<std::mutex> lock(m_descriptor_pool_lock);
        vkFreeDescriptorSets(m_device->vulkan_device(), m_descriptor_pool, 1,
                             &ds);
        m_nb_descriptor_set_allocated--;
        TRACE_CNT(descriptor_set_allocated_counter,
                  m_nb_descriptor_set_allocated);
    }

    uint32_t num_set_layouts() const { return m_descriptor_set_layouts.size(); }

    std::unique_ptr<cvk_buffer> allocate_pod_buffer();

    const std::vector<kernel_argument>& args() const { return m_args; }

    const kernel_sampler_metadata_map* sampler_metadata() const {
        return m_sampler_metadata;
    }

    const kernel_image_metadata_map* image_metadata() const {
        return m_image_metadata;
    }

    bool has_pod_arguments() const { return m_has_pod_arguments; }

    bool has_pod_buffer_arguments() const { return m_has_pod_buffer_arguments; }

    bool has_sampler_metadata() const { return m_sampler_metadata != nullptr; }

    bool has_image_metadata() const { return m_image_metadata != nullptr; }

    uint32_t pod_buffer_size() const { return m_pod_buffer_size; }

    uint32_t num_resource_slots() const { return m_num_resource_slots; }

    VkPipelineLayout pipeline_layout() const { return m_pipeline_layout; }

    VkDescriptorType pod_descriptor_type() const {
        return m_pod_descriptor_type;
    }

    cvk_program* program() const { return m_program; }

    bool uses_printf() const;

private:
    const uint32_t MAX_INSTANCES = config.max_entry_points_instances;

    cvk_device* m_device;
    cvk_context* m_context;
    cvk_program* m_program;
    std::string m_name;
    VkDescriptorType m_pod_descriptor_type;
    uint32_t m_pod_buffer_size;
    bool m_has_pod_arguments;
    bool m_has_pod_buffer_arguments;
    std::vector<kernel_argument> m_args;
    const kernel_sampler_metadata_map* m_sampler_metadata;
    const kernel_image_metadata_map* m_image_metadata;
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
    bool build_descriptor_sets_layout_bindings_for_program_scope_buffers(
        binding_stat_map& smap);
    bool build_descriptor_sets_layout_bindings_for_printf_buffer(
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

    uint32_t m_nb_descriptor_set_allocated;
    TRACE_CNT_VAR(descriptor_set_allocated_counter);

    bool m_first_allocation_failure;
};

struct cvk_program : public _cl_program, api_object<object_magic::program> {

    cvk_program(cvk_context* ctx)
        : api_object(ctx), m_num_devices(1U),
          m_binary_type(CL_PROGRAM_BINARY_TYPE_NONE),
          m_shader_module(VK_NULL_HANDLE),
          m_binary(m_context->device()->vulkan_spirv_env()) {
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

    cl_int set_user_spec_constant(uint32_t spec_id, size_t spec_size,
                                  const void* spec_value) {
        auto spec_const_iter = m_user_spec_constants.find(spec_id);
        if (spec_const_iter == m_user_spec_constants.end()) {
            return CL_INVALID_SPEC_ID;
        }
        if (spec_const_iter->second.size != spec_size) {
            return CL_INVALID_VALUE;
        }
        spec_const_iter->second.init_data(spec_size, spec_value);
        return CL_SUCCESS;
    }

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
    bool uses_printf() { return !m_binary.printf_descriptors().empty(); }
    const std::unordered_map<uint32_t, printf_descriptor>&
    printf_descriptors() {
        return m_binary.get_printf_descriptors();
    }

    const std::vector<kernel_argument>* args_for_kernel(std::string& name) {
        auto const& args = m_binary.kernels_arguments().find(name);
        if (args != m_binary.kernels_arguments().end()) {
            return &args->second;
        } else {
            return nullptr;
        }
    }

    const kernel_sampler_metadata_map* sampler_metadata(std::string& name) {
        auto const& md = m_binary.sampler_metadata().find(name);
        if (md != m_binary.sampler_metadata().end()) {
            return &md->second;
        } else {
            return nullptr;
        }
    }

    const kernel_image_metadata_map* image_metadata(std::string& name) {
        auto const& md = m_binary.image_metadata().find(name);
        if (md != m_binary.image_metadata().end()) {
            return &md->second;
        } else {
            return nullptr;
        }
    }

private:
    bool read_llvm_bitcode(const unsigned char* src, size_t size);

    void write_binary_header(unsigned char* dst) const;
    CHECK_RETURN cl_program_binary_type

    read_binary_header(const unsigned char* src, size_t size);

public:
    CHECK_RETURN bool read(const unsigned char* src, size_t size);

    CHECK_RETURN bool write(unsigned char* dst) const;

    size_t binary_size() const;

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

    const std::array<uint32_t, 3>&
    required_work_group_size(const std::string& kernel) const {
        return m_binary.required_work_group_size(kernel);
    }

    uint32_t required_sub_group_size(std::string& kernel) const {
        auto forced_subgroup_size_or_zero = []() {
            if (config.force_subgroup_size.set) {
                return config.force_subgroup_size();
            }
            return 0u;
        };

        const char* sub_group_size_attr = "intel_reqd_sub_group_size(";
        auto attrs = kernel_attributes(kernel);
        auto it = attrs.find(sub_group_size_attr);

        if (it == std::string::npos) {
            return forced_subgroup_size_or_zero();
        }
        it += strlen(sub_group_size_attr);
        auto it2 = attrs.substr(it).find(")");
        if (it2 == std::string::npos) {
            return forced_subgroup_size_or_zero();
        }

        uint32_t kernel_subgroup_size = atoi(attrs.substr(it, it2).c_str());

        if (config.force_subgroup_size.set) {
            uint32_t force_subgroup_size = config.force_subgroup_size();
            if (force_subgroup_size != kernel_subgroup_size) {
                cvk_warn_fn("overriding subgroup size specified inside kernel "
                            "'%s', using '%u' instead of '%u'",
                            kernel.c_str(), force_subgroup_size,
                            kernel_subgroup_size);
            }
            return config.force_subgroup_size();
        }

        return kernel_subgroup_size;
    }

    const VkPipelineCache& pipeline_cache() const { return m_pipeline_cache; }

    CHECK_RETURN cvk_entry_point* get_entry_point(std::string& name,
                                                  cl_int* errcode_ret);

    bool create_module_constant_data_buffer() {
        cl_int err;
        if (m_binary.constant_data_buffer() != nullptr) {
            auto& init_data = m_binary.constant_data_buffer()->data;
            void* init_data_ptr =
                reinterpret_cast<void*>(const_cast<char*>(init_data.data()));
            m_module_constant_data_buffer =
                cvk_buffer::create(m_context, CL_MEM_COPY_HOST_PTR,
                                   init_data.size(), init_data_ptr, &err);
            if (m_module_constant_data_buffer == nullptr) {
                return false;
            }
        }
        return true;
    }

    const cvk_buffer* module_constant_data_buffer() const {
        return m_module_constant_data_buffer.get();
    }

    const constant_data_buffer_info* module_constant_data_buffer_info() const {
        return m_binary.constant_data_buffer();
    }

    const printf_buffer_desc_info& printf_buffer_info() const {
        return m_binary.printf_buffer_info();
    }

    bool options_allow_split_region(std::string options) {
        if (options.find("-uniform-workgroup-size") != std::string::npos)
            return false;
        return true;
    }

    bool can_split_region() {
        int status = options_allow_split_region(m_build_options);
#if COMPILER_AVAILABLE
        status &= options_allow_split_region(config.clspv_options);
#endif
        return status;
    }

    CHECK_RETURN cl_int parse_user_spec_constants();

    const std::string& kernel_attributes(const std::string& kernel_name) const {
        return m_binary.kernels_attributes().at(kernel_name);
    }

    uint32_t kernel_flags(const std::string& kernel) const {
        return m_binary.kernels_flags().at(kernel);
    }

private:
    void do_build();
    std::string prepare_build_options(const cvk_device* device) const;
    CHECK_RETURN cl_build_status do_build_inner(const cvk_device* device);

#if COMPILER_AVAILABLE
#ifndef CLSPV_ONLINE_COMPILER
    CHECK_RETURN cl_build_status
    do_build_inner_offline(bool build_to_ir, bool build_from_il,
                           std::string& build_options, std::string& tmp_folder);
#else
    CHECK_RETURN cl_build_status do_build_inner_online(
        bool build_to_ir, bool build_from_il, std::string& build_options);
#endif
#endif

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
    std::vector<uint8_t> m_ir;
    std::vector<uint8_t> m_il;
    VkShaderModule m_shader_module;
    std::unordered_map<const cvk_device*, std::atomic<cl_build_status>>
        m_dev_status;
    std::string m_build_options;
    spir_binary m_binary;
    std::string m_build_log;
    std::vector<cvk_sampler_holder> m_literal_samplers;
    VkPushConstantRange m_push_constant_range;
    std::unordered_map<std::string, std::unique_ptr<cvk_entry_point>>
        m_entry_points;
    std::vector<uint32_t> m_stripped_binary;
    VkPipelineCache m_pipeline_cache;
    std::unique_ptr<cvk_buffer> m_module_constant_data_buffer;
    std::unordered_map<uint32_t, user_spec_constant_data> m_user_spec_constants;
};

static inline cvk_program* icd_downcast(cl_program program) {
    return static_cast<cvk_program*>(program);
}

using cvk_program_holder = refcounted_holder<cvk_program>;
