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

#include <cstdint>
#include <fstream>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

#include "spirv-tools/libspirv.h"
#include "spirv/1.0/spirv.hpp"

#include "objects.hpp"

const int SPIR_WORD_SIZE = 4;

enum class kernel_argument_kind {
    buffer,
    pod,
    pod_ubo,
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
    int offset;
    uint32_t size;
    kernel_argument_kind kind;
    uint32_t local_spec_id;
    uint32_t local_elem_size;

    bool is_pod() const {
        return (kind == kernel_argument_kind::pod) || (kind == kernel_argument_kind::pod_ubo);
    }
};

class spir_binary {

    using kernels_arguments_map = std::unordered_map<std::string, std::vector<kernel_argument>>;
    const uint32_t MAGIC = 0x00BEEF00;

public:
    spir_binary(spv_target_env env) : m_loaded_from_binary(false) {
        m_context = spvContextCreate(env);
    }
    ~spir_binary() {
        spvContextDestroy(m_context);
    }
    CHECK_RETURN bool load_spir(const char *fname);
    CHECK_RETURN bool load_spir(std::istream &istream, uint32_t size);
    CHECK_RETURN bool load_descriptor_map(const char *fname);
    CHECK_RETURN bool load_descriptor_map(std::istream &istream);
    void insert_descriptor_map(const spir_binary &other);
    CHECK_RETURN bool save_spir(const char *fname) const;
    CHECK_RETURN bool load(std::istream &istream);
    CHECK_RETURN bool save(std::ostream &ostream) const;
    CHECK_RETURN bool save(const char *fname) const;
    CHECK_RETURN bool read(const unsigned char *src, size_t size);
    CHECK_RETURN bool write(unsigned char *dst) const;
    size_t size() const;
    bool loaded_from_binary() const { return m_loaded_from_binary; }
    size_t spir_size() const { return m_code.size() * sizeof(uint32_t); }
    const uint32_t* spir_data() const { return m_code.data(); }
    void use(std::vector<uint32_t> &&src);
    void set_target_env(spv_target_env env);
    const std::vector<uint32_t>& code() const { return m_code; };
    CHECK_RETURN bool validate() const;
    size_t num_kernels() const { return m_dmaps.size(); }
    const kernels_arguments_map& kernels_arguments() const { return m_dmaps; }

private:
    spv_context m_context;
    std::vector<uint32_t> m_code;
    kernels_arguments_map m_dmaps;
    std::string m_dmaps_text;
    bool m_loaded_from_binary;
};

extern std::string gCLSPVPath;

enum class build_operation {
    build,
    compile,
    link
};

using cvk_program_callback = void (*)(cl_program, void*);

typedef struct _cl_program cvk_program;

typedef struct _cl_program : public api_object {

    _cl_program(cvk_context *ctx) :
        api_object(ctx),
        m_num_devices(1U),
        m_binary_type(CL_PROGRAM_BINARY_TYPE_NONE),
        m_shader_module(VK_NULL_HANDLE)
    {
        m_dev_status[m_context->device()] = CL_BUILD_NONE;
    }

    virtual ~_cl_program() {
        if (m_shader_module != VK_NULL_HANDLE) {
            auto vkdev = m_context->device()->vulkan_device();
            vkDestroyShaderModule(vkdev, m_shader_module, nullptr);
        }
    }

    void append_source(const char *src, size_t len) {
        if (len != 0) {
            m_source.append(src, len);
        } else {
            m_source.append(src);
        }
    }

    const std::string& source() const {
        return m_source;
    }

    uint32_t num_devices() const { return m_num_devices; }

    cl_program_binary_type binary_type(const cvk_device *) const {
        return m_binary_type;
    }

    bool can_be_linked() const {
        auto dev = m_context->device();
        return ((build_status() == CL_BUILD_SUCCESS) &&
                ((binary_type(dev) == CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT) ||
                 (binary_type(dev) == CL_PROGRAM_BINARY_TYPE_LIBRARY)));
    }

    CHECK_RETURN bool build(build_operation operation, cl_uint num_devices, const cvk_device *const*device_list, const char *options, cl_uint num_input_programs, const cvk_program *const*input_programs, const char **header_include_names, cvk_program_callback cb, void *data);

    const std::string& build_options() const { return m_build_options; }

    cl_build_status build_status(const cvk_device *device) const {
        return m_dev_status.at(device);
    }

    cl_build_status build_status() const {
        for (auto dev_st : m_dev_status) {
            if (dev_st.second != CL_BUILD_SUCCESS) {
                return dev_st.second;
            }
        }

        return CL_BUILD_SUCCESS;
    }

    std::vector<const cvk_device*> devices() const {
        std::vector<const cvk_device*> ret;

        for (auto &dev_st : m_dev_status) {
            ret.push_back(dev_st.first);
        }

        return std::move(ret);
    }

    VkShaderModule shader_module() const {
        return m_shader_module;
    }

    void wait_for_operation() {
        m_thread->join();
        delete m_thread;
    }

    void complete_operation(cvk_device *device, cl_build_status status) {
        m_dev_status[device] = status;
        m_lock.unlock();
        if (m_operation_callback != nullptr) {
            m_operation_callback(this, m_operation_callback_data);
        }
        release();
    }

    unsigned num_kernels() const { return m_binary.num_kernels(); }
    bool loaded_from_binary() const { return m_binary.loaded_from_binary(); }

    const std::vector<kernel_argument>* args_for_kernel(std::string& name) {
        auto const &args = m_binary.kernels_arguments().find(name);
        if (args != m_binary.kernels_arguments().end()) {
            return &args->second;
        } else {
            return nullptr;
        }
    }

    CHECK_RETURN bool read(const unsigned char *src, size_t size) {
        return m_binary.read(src, size);
    }

    CHECK_RETURN bool write(unsigned char *dst) const {
        return m_binary.write(dst);
    }

    size_t binary_size() const {
        return m_binary.size();
    }

    std::vector<const char*> kernel_names() const {
        std::vector<const char*> ret;
        for (auto &kname_args : m_binary.kernels_arguments()) {
            ret.push_back(kname_args.first.c_str());
        }
        return ret;
    }

private:
    void do_build();
    CHECK_RETURN cl_build_status compile_source();
    CHECK_RETURN cl_build_status link();

    uint32_t m_num_devices;
    cl_uint m_num_input_programs;
    std::vector<const cvk_program*> m_input_programs;
    std::vector<const char *> m_header_include_names;
    build_operation m_operation;
    cl_program_binary_type m_binary_type;
    cvk_program_callback m_operation_callback;
    void *m_operation_callback_data;
    std::mutex m_lock;
    std::thread *m_thread;
    std::string m_source;
    VkShaderModule m_shader_module;
    std::unordered_map<const cvk_device*, cl_build_status> m_dev_status;
    std::string m_build_options;
    spir_binary m_binary{SPV_ENV_VULKAN_1_0};

} cvk_program;

