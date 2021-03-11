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

#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#include <sstream>
#include <utility>
#include <vector>

#include <vulkan/vulkan.h>

#include "clspv/Sampler.h"
#ifdef CLSPV_ONLINE_COMPILER
#include "clspv/Compiler.h"
#endif
#include "spirv-tools/linker.hpp"
#include "spirv-tools/optimizer.hpp"
#include "spirv/unified1/NonSemanticClspvReflection.h"
#include "spirv/unified1/spirv.hpp"

#include "init.hpp"
#include "log.hpp"
#include "program.hpp"

struct membuf : public std::streambuf {
    membuf(const unsigned char* begin, const unsigned char* end) {
        auto sbegin =
            reinterpret_cast<char*>(const_cast<unsigned char*>(begin));
        auto send = reinterpret_cast<char*>(const_cast<unsigned char*>(end));
        setg(sbegin, sbegin, send);
    }
    membuf(unsigned char* begin, unsigned char* end) {
        auto sbegin = reinterpret_cast<char*>(begin);
        auto send = reinterpret_cast<char*>(end);
        setp(sbegin, send);
    }
};

struct reflection_parse_data {
    uint32_t uint_id = 0;
    std::unordered_map<uint32_t, uint32_t> constants;
    std::unordered_map<uint32_t, std::string> strings;
    spir_binary* binary;
};

spv_result_t parse_reflection(void* user_data,
                              const spv_parsed_instruction_t* inst) {
    // Helper function to map instruction to argument type.
    auto inst_to_arg_kind = [](uint32_t inst) {
        switch (static_cast<NonSemanticClspvReflectionInstructions>(inst)) {
        case NonSemanticClspvReflectionArgumentStorageBuffer:
            return kernel_argument_kind::buffer;
        case NonSemanticClspvReflectionArgumentUniform:
            return kernel_argument_kind::buffer_ubo;
        case NonSemanticClspvReflectionArgumentPodStorageBuffer:
            return kernel_argument_kind::pod;
        case NonSemanticClspvReflectionArgumentPodUniform:
            return kernel_argument_kind::pod_ubo;
        case NonSemanticClspvReflectionArgumentPodPushConstant:
            return kernel_argument_kind::pod_pushconstant;
        case NonSemanticClspvReflectionArgumentSampledImage:
            return kernel_argument_kind::ro_image;
        case NonSemanticClspvReflectionArgumentStorageImage:
            return kernel_argument_kind::wo_image;
        case NonSemanticClspvReflectionArgumentSampler:
            return kernel_argument_kind::sampler;
        case NonSemanticClspvReflectionArgumentWorkgroup:
            return kernel_argument_kind::local;
        default:
            cvk_error_fn("Unhandled reflection instruction for arg kind");
            break;
        }
        return kernel_argument_kind::buffer;
    };

    // Helper function to map instruction to push constant type.
    auto inst_to_push_constant = [](uint32_t inst) {
        switch (static_cast<NonSemanticClspvReflectionInstructions>(inst)) {
        case NonSemanticClspvReflectionPushConstantGlobalOffset:
            return pushconstant::global_offset;
        case NonSemanticClspvReflectionPushConstantEnqueuedLocalSize:
            return pushconstant::enqueued_local_size;
        case NonSemanticClspvReflectionPushConstantGlobalSize:
            return pushconstant::global_size;
        case NonSemanticClspvReflectionPushConstantRegionOffset:
            return pushconstant::region_offset;
        case NonSemanticClspvReflectionPushConstantNumWorkgroups:
            return pushconstant::num_workgroups;
        case NonSemanticClspvReflectionPushConstantRegionGroupOffset:
            return pushconstant::region_group_offset;
        default:
            cvk_error_fn("Unhandled reflection instruction for push constant");
            break;
        }
        return pushconstant::global_offset;
    };

    auto* helper = reinterpret_cast<reflection_parse_data*>(user_data);
    switch (inst->opcode) {
    case spv::OpTypeInt:
        if (inst->words[2] == 32 && inst->words[3] == 0) {
            helper->uint_id = inst->result_id;
        }
        break;
    case spv::OpConstant:
        if (inst->words[1] == helper->uint_id) {
            helper->constants[inst->result_id] = inst->words[3];
        }
        break;
    case spv::OpString:
        helper->strings[inst->result_id] =
            std::string(reinterpret_cast<const char*>(&inst->words[2]));
        break;
    case spv::OpExtInst:
        if (inst->ext_inst_type ==
            SPV_EXT_INST_TYPE_NONSEMANTIC_CLSPVREFLECTION) {
            auto ext_inst = inst->words[4];
            switch (ext_inst) {
            case NonSemanticClspvReflectionKernel: {
                // Record the kernel name.
                const auto& name = helper->strings[inst->words[6]];
                helper->strings[inst->result_id] = name;
                helper->binary->add_kernel(name);
                break;
            }
            case NonSemanticClspvReflectionArgumentInfo: {
                // Record the argument name.
                // TODO: parse the rest of the information when clspv produces
                // it.
                const auto& name = helper->strings[inst->words[5]];
                helper->strings[inst->result_id] = name;
                break;
            }
            case NonSemanticClspvReflectionArgumentStorageBuffer:
            case NonSemanticClspvReflectionArgumentUniform:
            case NonSemanticClspvReflectionArgumentSampledImage:
            case NonSemanticClspvReflectionArgumentStorageImage:
            case NonSemanticClspvReflectionArgumentSampler: {
                // These arguments have descriptor set, binding and an optional
                // arg info.
                auto kernel = helper->strings[inst->words[5]];
                auto ordinal = helper->constants[inst->words[6]];
                auto descriptor_set = helper->constants[inst->words[7]];
                if (descriptor_set >= spir_binary::MAX_DESCRIPTOR_SETS)
                    return SPV_ERROR_INVALID_DATA;
                auto binding = helper->constants[inst->words[8]];
                std::string arg_name;
                if (inst->num_operands == 9) {
                    arg_name = helper->strings[inst->words[9]];
                }
                auto kind = inst_to_arg_kind(ext_inst);
                kernel_argument arg = {arg_name, ordinal, descriptor_set,
                                       binding,  0,       0,
                                       kind,     0,       0};
                helper->binary->add_kernel_argument(kernel, std::move(arg));
                break;
            }
            case NonSemanticClspvReflectionArgumentPodStorageBuffer:
            case NonSemanticClspvReflectionArgumentPodUniform: {
                // These arguments have descriptor set, binding, offset, size
                // and an optional arg info.
                auto kernel = helper->strings[inst->words[5]];
                auto ordinal = helper->constants[inst->words[6]];
                auto descriptor_set = helper->constants[inst->words[7]];
                if (descriptor_set >= spir_binary::MAX_DESCRIPTOR_SETS)
                    return SPV_ERROR_INVALID_DATA;
                auto binding = helper->constants[inst->words[8]];
                auto offset = helper->constants[inst->words[9]];
                auto size = helper->constants[inst->words[10]];
                std::string arg_name;
                if (inst->num_operands == 11) {
                    arg_name = helper->strings[inst->words[11]];
                }
                auto kind = inst_to_arg_kind(ext_inst);
                kernel_argument arg = {arg_name, ordinal, descriptor_set,
                                       binding,  offset,  size,
                                       kind,     0,       0};
                helper->binary->add_kernel_argument(kernel, std::move(arg));
                break;
            }
            case NonSemanticClspvReflectionArgumentPodPushConstant: {
                // These arguments have offset, size and an optional arg info.
                auto kernel = helper->strings[inst->words[5]];
                auto ordinal = helper->constants[inst->words[6]];
                auto offset = helper->constants[inst->words[7]];
                auto size = helper->constants[inst->words[8]];
                std::string arg_name;
                if (inst->num_operands == 9) {
                    arg_name = helper->strings[inst->words[9]];
                }
                auto kind = inst_to_arg_kind(ext_inst);
                kernel_argument arg = {arg_name, ordinal, 0, 0, offset,
                                       size,     kind,    0, 0};
                helper->binary->add_kernel_argument(kernel, std::move(arg));
                break;
            }
            case NonSemanticClspvReflectionArgumentWorkgroup: {
                // These arguments have spec id, elem size and an optional arg
                // info.
                auto kernel = helper->strings[inst->words[5]];
                auto ordinal = helper->constants[inst->words[6]];
                auto spec_id = helper->constants[inst->words[7]];
                auto size = helper->constants[inst->words[8]];
                std::string arg_name;
                if (inst->num_operands == 9) {
                    arg_name = helper->strings[inst->words[9]];
                }
                auto kind = inst_to_arg_kind(ext_inst);
                kernel_argument arg = {arg_name, ordinal, 0,       0,   0,
                                       0,        kind,    spec_id, size};
                helper->binary->add_kernel_argument(kernel, std::move(arg));
                break;
            }
            case NonSemanticClspvReflectionSpecConstantWorkgroupSize: {
                // Reflection encodes all three spec ids in a single
                // instruction.
                auto x_id = helper->constants[inst->words[5]];
                auto y_id = helper->constants[inst->words[6]];
                auto z_id = helper->constants[inst->words[7]];
                helper->binary->add_spec_constant(
                    spec_constant::workgroup_size_x, x_id);
                helper->binary->add_spec_constant(
                    spec_constant::workgroup_size_y, y_id);
                helper->binary->add_spec_constant(
                    spec_constant::workgroup_size_z, z_id);
                break;
            }
            case NonSemanticClspvReflectionSpecConstantGlobalOffset: {
                // Reflection encodes all three spec ids in a single
                // instruction.
                auto x_id = helper->constants[inst->words[5]];
                auto y_id = helper->constants[inst->words[6]];
                auto z_id = helper->constants[inst->words[7]];
                helper->binary->add_spec_constant(
                    spec_constant::global_offset_x, x_id);
                helper->binary->add_spec_constant(
                    spec_constant::global_offset_y, y_id);
                helper->binary->add_spec_constant(
                    spec_constant::global_offset_z, z_id);
                break;
            }
            case NonSemanticClspvReflectionSpecConstantWorkDim: {
                auto dim_id = helper->constants[inst->words[5]];
                helper->binary->add_spec_constant(spec_constant::work_dim,
                                                  dim_id);
                break;
            }
            case NonSemanticClspvReflectionPushConstantGlobalOffset:
            case NonSemanticClspvReflectionPushConstantEnqueuedLocalSize:
            case NonSemanticClspvReflectionPushConstantGlobalSize:
            case NonSemanticClspvReflectionPushConstantRegionOffset:
            case NonSemanticClspvReflectionPushConstantNumWorkgroups:
            case NonSemanticClspvReflectionPushConstantRegionGroupOffset: {
                auto offset = helper->constants[inst->words[5]];
                auto size = helper->constants[inst->words[6]];
                auto pc = inst_to_push_constant(ext_inst);
                helper->binary->add_push_constant(pc, {offset, size});
                break;
            }
            case NonSemanticClspvReflectionLiteralSampler: {
                // Track descriptor set and binding. Decode the sampler mask.
                auto descriptor_set = helper->constants[inst->words[5]];
                if (descriptor_set >= spir_binary::MAX_DESCRIPTOR_SETS)
                    return SPV_ERROR_INVALID_DATA;
                auto binding = helper->constants[inst->words[6]];
                auto mask = helper->constants[inst->words[7]];
                uint32_t coords = mask & clspv::kSamplerNormalizedCoordsMask;
                bool normalized_coords =
                    coords == clspv::CLK_NORMALIZED_COORDS_TRUE;
                cl_addressing_mode addressing;
                switch (mask & clspv::kSamplerAddressMask) {
                case clspv::CLK_ADDRESS_NONE:
                default:
                    addressing = CL_ADDRESS_NONE;
                    break;
                case clspv::CLK_ADDRESS_CLAMP_TO_EDGE:
                    addressing = CL_ADDRESS_CLAMP_TO_EDGE;
                    break;
                case clspv::CLK_ADDRESS_CLAMP:
                    addressing = CL_ADDRESS_CLAMP;
                    break;
                case clspv::CLK_ADDRESS_MIRRORED_REPEAT:
                    addressing = CL_ADDRESS_MIRRORED_REPEAT;
                    break;
                case clspv::CLK_ADDRESS_REPEAT:
                    addressing = CL_ADDRESS_REPEAT;
                    break;
                }
                cl_filter_mode filter;
                switch (mask & clspv::kSamplerFilterMask) {
                case clspv::CLK_FILTER_NEAREST:
                default:
                    filter = CL_FILTER_NEAREST;
                    break;
                case clspv::CLK_FILTER_LINEAR:
                    filter = CL_FILTER_LINEAR;
                    break;
                }
                helper->binary->add_literal_sampler({descriptor_set, binding,
                                                     normalized_coords,
                                                     addressing, filter});
                break;
            }
            case NonSemanticClspvReflectionPropertyRequiredWorkgroupSize: {
                auto kernel = helper->strings[inst->words[5]];
                auto x = helper->constants[inst->words[6]];
                auto y = helper->constants[inst->words[7]];
                auto z = helper->constants[inst->words[8]];
                helper->binary->set_required_work_group_size(kernel, x, y, z);
                break;
            }
            default:
                break;
            }
        }
        break;
    default:
        break;
    }

    return SPV_SUCCESS;
}

/*
 * BINARY FILE FORMAT
 * +---------+-----------------------+
 * | U32     | Version               |
 * | U32     | SPIR Size             |
 * | N * U32 | SPIR                  |
 * +---------+-----------------------+
 */

bool spir_binary::load_spir(std::istream& istream, uint32_t size) {
    m_code.assign(size / SPIR_WORD_SIZE, 0);
    istream.read(reinterpret_cast<char*>(m_code.data()), size);
    return istream.good();
}

bool spir_binary::load_spir(const char* fname) {
    std::ifstream ifile;

    ifile.open(fname, std::ios::in | std::ios::binary);

    if (!ifile.is_open()) {
        cvk_error("Failed to open %s", fname);
        return false;
    }

    ifile.seekg(0, std::ios::end);
    uint32_t size = ifile.tellg();
    ifile.seekg(0, std::ios::beg);

    return load_spir(ifile, size);
}

bool spir_binary::load(std::istream& istream) {
    m_loaded_from_binary = true;

    // Check magic
    uint32_t magic;
    istream.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));

    if (magic != MAGIC) {
        return false;
    }

    // Load SPIR
    uint32_t size_spir;
    istream.read(reinterpret_cast<char*>(&size_spir), sizeof(uint32_t));

    if (size_spir % SPIR_WORD_SIZE != 0) {
        return false;
    }

    if (!load_spir(istream, size_spir)) {
        return false;
    }

    return load_descriptor_map();
}

bool spir_binary::read(const unsigned char* src, size_t size) {
    membuf bufview(src, src + size);
    std::istream istream(&bufview);
    return load(istream);
}

bool spir_binary::save_spir(const char* fname) const {
    std::ofstream ofile;

    ofile.open(fname, std::ios::out | std::ios::binary);

    if (!ofile.is_open()) {
        return false;
    }

    ofile.write(reinterpret_cast<const char*>(m_code.data()),
                m_code.size() * sizeof(SPIR_WORD_SIZE));

    return ofile.good();
}

bool spir_binary::save(std::ostream& ostream) const {
    // Write magic
    ostream.write(reinterpret_cast<const char*>(&MAGIC), sizeof(MAGIC));

    // Write SPIR
    uint32_t spir_size = m_code.size() * SPIR_WORD_SIZE;
    ostream.write(reinterpret_cast<const char*>(&spir_size), sizeof(spir_size));
    ostream.write(reinterpret_cast<const char*>(m_code.data()), spir_size);

    return ostream.good();
}

bool spir_binary::save(const char* fname) const {
    std::ofstream ofile;

    ofile.open(fname, std::ios::out | std::ios::binary);

    if (!ofile.is_open()) {
        return false;
    }

    return save(ofile);
}

size_t spir_binary::size() const {
    return sizeof(MAGIC) + sizeof(uint32_t) + (m_code.size() * SPIR_WORD_SIZE);
}

bool spir_binary::write(unsigned char* dst) const {
    membuf bufview(dst, dst + size());
    std::ostream ostream(&bufview);
    return save(ostream);
}

void spir_binary::use(std::vector<uint32_t>&& src) { m_code = std::move(src); }

void spir_binary::set_target_env(spv_target_env env) {
    spvContextDestroy(m_context);
    m_context = spvContextCreate(env);
}

bool spir_binary::validate() const {
    spv_diagnostic diag;
    spv_result_t res =
        spvValidateBinary(m_context, m_code.data(), m_code.size(), &diag);
    spvDiagnosticPrint(diag);
    spvDiagnosticDestroy(diag);
    return res == SPV_SUCCESS;
}

bool spir_binary::strip_reflection(std::vector<uint32_t>* stripped) {
    const spvtools::MessageConsumer consumer =
        [](spv_message_level_t level, const char*,
           const spv_position_t& position, const char* message) {

#define msgtpl "spvtools says '%s' at position %zu"
            switch (level) {
            case SPV_MSG_FATAL:
            case SPV_MSG_INTERNAL_ERROR:
            case SPV_MSG_ERROR:
                cvk_error(msgtpl, message, position.index);
                break;
            case SPV_MSG_WARNING:
                cvk_warn(msgtpl, message, position.index);
                break;
            case SPV_MSG_INFO:
                cvk_info(msgtpl, message, position.index);
                break;
            case SPV_MSG_DEBUG:
                cvk_debug(msgtpl, message, position.index);
                break;
            }
#undef msgtpl
        };

    spvtools::Optimizer opt(m_target_env);
    opt.SetMessageConsumer(consumer);
    opt.RegisterPass(spvtools::CreateStripReflectInfoPass());
    spvtools::OptimizerOptions options;
    options.set_run_validator(false);
    if (!opt.Run(m_code.data(), m_code.size(), stripped, options)) {
        return false;
    }
    return true;
}

bool spir_binary::load_descriptor_map() {
    reflection_parse_data helper;
    helper.binary = this;

    // TODO: The parser assumes a valid SPIR-V module, but validation is not
    // run until later.
    auto result =
        spvBinaryParse(m_context, &helper, m_code.data(), m_code.size(),
                       nullptr, parse_reflection, nullptr);
    if (result != SPV_SUCCESS) {
        cvk_error_fn("Parsing SPIR-V module reflection failed: %d", result);
        return false;
    }

    return true;
}

bool spir_binary::get_capabilities(
    std::vector<spv::Capability>& capabilities) const {
    // Callback for receiving parsed instructions.
    // The `user_data` parameter will be a pointer to the vector of
    // capabilities (we cannot use a lambda capture for this as it prevents the
    // lambda from being able to be converted to a function pointer).
    auto parse_inst = [](void* user_data,
                         const spv_parsed_instruction_t* inst) {
        // Stop parsing at first instruction that is not an OpCapability.
        if (inst->opcode != spv::Op::OpCapability) {
            return SPV_END_OF_STREAM;
        }

        // Add the capability to the list.
        uint32_t capability = inst->words[inst->operands[0].offset];
        auto capabilities =
            reinterpret_cast<std::vector<spv::Capability>*>(user_data);
        capabilities->push_back(static_cast<spv::Capability>(capability));

        return SPV_SUCCESS;
    };

    // Parse the SPIR-V binary to build the list of required capabilities.
    spv_result_t result =
        spvBinaryParse(m_context, &capabilities, m_code.data(), m_code.size(),
                       nullptr, parse_inst, nullptr);
    if (result != SPV_SUCCESS && result != SPV_END_OF_STREAM) {
        cvk_error_fn("Parsing SPIR-V module failed: %d", result);
        return false;
    }
    return true;
}

namespace {

bool save_string_to_file(const std::string& fname, const std::string& text) {
    std::ofstream ofile{fname};

    if (!ofile.is_open()) {
        return false;
    }

    ofile.write(text.c_str(), text.size());
    ofile.close();

    return ofile.good();
}

bool save_il_to_file(const std::string& fname, const std::vector<uint8_t>& il) {
    std::ofstream ofile{fname, std::ios::binary};

    if (!ofile.is_open()) {
        return false;
    }

    ofile.write(reinterpret_cast<const char*>(il.data()), il.size());
    ofile.close();

    return ofile.good();
}

struct temp_file_deletion_stack {

    ~temp_file_deletion_stack() {
        if (!gKeepTemporaries) {
            for (auto path = m_paths.rbegin(); path < m_paths.rend(); ++path) {
                std::remove((*path).c_str());
            }
        }
    }

    void push(const std::string& path) { m_paths.push_back(path); }

private:
    std::vector<std::string> m_paths;
};

} // namespace

cl_build_status cvk_program::compile_source(const cvk_device* device) {
    bool use_tmp_folder = true;
    bool save_headers = true;
    temp_file_deletion_stack temps;
#ifdef CLSPV_ONLINE_COMPILER
    use_tmp_folder =
        m_operation == build_operation::compile && m_num_input_programs > 0;
    save_headers = m_operation == build_operation::compile;
#endif

    std::string tmp_folder;
    if (use_tmp_folder) {
        // Create temporary folder
        std::string tmp_template{"clvk-XXXXXX"};
        const char* tmp = cvk_mkdtemp(tmp_template);
        if (tmp == nullptr) {
            return CL_BUILD_ERROR;
        }
        tmp_folder = tmp;
        cvk_info("Created temporary folder \"%s\"", tmp_folder.c_str());
        temps.push(tmp_folder);
    }

#ifndef CLSPV_ONLINE_COMPILER
    bool build_from_il = m_il.size() > 0;
    std::string clspv_input_file{tmp_folder + "/source"};
    std::string llvmspirv_input_file{tmp_folder + "/source.spv"};
    // Save input program to a file
    if (build_from_il) {
        clspv_input_file += ".bc";
        temps.push(llvmspirv_input_file);
        if (!save_il_to_file(llvmspirv_input_file, m_il)) {
            cvk_error_fn("Couldn't save IL to file!");
            return CL_BUILD_ERROR;
        }
    } else {
        clspv_input_file += ".cl";
        temps.push(clspv_input_file);
        if (!save_string_to_file(clspv_input_file, m_source)) {
            cvk_error_fn("Couldn't save source to file!");
            return CL_BUILD_ERROR;
        }
    }
#endif

    // Save headers
    if (save_headers) {
        for (cl_uint i = 0; i < m_num_input_programs; i++) {
            std::string fname{tmp_folder + "/" + m_header_include_names[i]};
            temps.push(fname);
            if (!save_string_to_file(fname, m_input_programs[i]->source())) {
                cvk_error_fn("Couldn't save header to file!");
                return CL_BUILD_ERROR;
            }
        }
    }

    // Strip off a few options we can't handle
    std::string processed_options;
    if (m_build_options.size() > 0) {
        processed_options += " ";
        processed_options += m_build_options;
    }
    std::vector<std::pair<std::string, std::string>> option_substitutions = {
        // TODO Enable in clspv and figure out interface
        {"-cl-kernel-arg-info", ""},
        // FIXME The 1.2 conformance tests shouldn't pass this option.
        //       It doesn't exist after OpenCL 1.0.
        {"-cl-strict-aliasing", ""},
        // clspv require entrypoint inlining for OpenCL 2.0
        {"-cl-std=CL2.0", "-cl-std=CL2.0 -inline-entry-points"},
    };

    for (auto& subst : option_substitutions) {
        size_t loc = processed_options.find(subst.first);
        if (loc != std::string::npos) {
            processed_options.replace(loc, subst.first.length(), subst.second);
        }
    }

    // Select operation
    // TODO support building a library with clBuildProgram
    if (m_operation == build_operation::compile) {
        m_binary_type = CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT;
    } else {
        m_binary_type = CL_PROGRAM_BINARY_TYPE_EXECUTABLE;
    }

    // Prepare options
    std::string options = processed_options;
    std::string single_precision_option = "-cl-single-precision-constant";
    if (processed_options.find(single_precision_option) == std::string::npos) {
        options += " " + single_precision_option + " ";
    }
    if (!devices_support_images()) {
        options += " -images=0 ";
    }

    // 8-bit storage capability restrictions.
    const auto& features_8bit_storage =
        m_context->device()->device_8bit_storage_features();
    if (features_8bit_storage.storageBuffer8BitAccess == VK_FALSE) {
        options += " -no-8bit-storage=ssbo ";
    }
    if (features_8bit_storage.uniformAndStorageBuffer8BitAccess == VK_FALSE) {
        options += " -no-8bit-storage=ubo ";
    }
    if (features_8bit_storage.storagePushConstant8 == VK_FALSE) {
        options += " -no-8bit-storage=pushconstant ";
    }

    // 16-bit storage capability restrictions.
    const auto& features_16bit_storage =
        m_context->device()->device_16bit_storage_features();
    if (features_16bit_storage.storageBuffer16BitAccess == VK_FALSE) {
        options += " -no-16bit-storage=ssbo ";
    }
    if (features_16bit_storage.uniformAndStorageBuffer16BitAccess == VK_FALSE) {
        options += " -no-16bit-storage=ubo ";
    }
    if (features_16bit_storage.storagePushConstant16 == VK_FALSE) {
        options += " -no-16bit-storage=pushconstant ";
    }

    options += " -max-pushconstant-size=" +
               std::to_string(device->vulkan_max_push_constants_size()) + " ";
    options += " -int8 ";
    if (device->supports_ubo_stdlayout()) {
        options += " -std430-ubo-layout ";
    }
    options += " -global-offset ";
    options += " " + gCLSPVOptions + " ";

#ifdef CLSPV_ONLINE_COMPILER
    cvk_info("About to compile \"%s\"", options.c_str());
    auto result = clspv::CompileFromSourceString(
        m_source, "", options, m_binary.raw_binary(), &m_build_log);
    cvk_info("Return code was: %d", result);
    if (result != 0) {
        cvk_error_fn("failed to compile the program");
        return CL_BUILD_ERROR;
    }

    // Load descriptor map
    if (!m_binary.load_descriptor_map()) {
        cvk_error("Could not load descriptor map for SPIR-V binary.");
        return CL_BUILD_ERROR;
    }
#else
    if (build_from_il) {
        // Compose llvm-spirv command-line
        std::string cmd{gLLVMSPIRVPath};
        cmd += " -r ";
        cmd += " -o ";
        cmd += clspv_input_file;
        cmd += " ";
        cmd += llvmspirv_input_file;

        temps.push(clspv_input_file);

        // Call the translator
        int status = cvk_exec(cmd);
        cvk_info("Return code was: %d", status);

        if (status != 0) {
            cvk_error_fn("failed to translate SPIR-V to LLVM IR");
            return CL_BUILD_ERROR;
        }
    }

    // Compose clspv command-line
    std::string cmd{gCLSPVPath};
    std::string spirv_file{tmp_folder + "/compiled.spv"};

    temps.push(spirv_file);

    if (build_from_il) {
        cmd += " -x ir ";
    }
    cmd += " ";
    cmd += clspv_input_file;
    cmd += " ";
    cmd += options;
    cmd += " -o ";
    cmd += spirv_file;
    cvk_info("About to run \"%s\"", cmd.c_str());

    // Call clspv
    int status = cvk_exec(cmd, &m_build_log);
    cvk_info("Return code was: %d", status);

    if (status != 0) {
        cvk_error_fn("failed to compile the program");
        return CL_BUILD_ERROR;
    }

    // Load SPIR-V program
    const char* filename = spirv_file.c_str();
    if (!m_binary.load_spir(filename)) {
        cvk_error("Could not load SPIR-V binary from \"%s\"", filename);
        return CL_BUILD_ERROR;
    } else {
        cvk_info("Loaded SPIR-V binary from \"%s\", size = %zu words", filename,
                 m_binary.code().size());
    }

    // Load descriptor map
    if (!m_binary.load_descriptor_map()) {
        cvk_error("Could not load descriptor map for SPIR-V binary.");
        return CL_BUILD_ERROR;
    }
#endif

    return CL_BUILD_SUCCESS;
}

cl_build_status cvk_program::link() {
    spvtools::Context context(SPV_ENV_VULKAN_1_0);
    std::vector<uint32_t> linked;
    std::vector<std::vector<uint32_t>> binaries(m_num_input_programs);

    const spvtools::MessageConsumer consumer =
        [](spv_message_level_t level, const char*,
           const spv_position_t& position, const char* message) {

#define msgtpl "spvtools says '%s' at position %zu"
            switch (level) {
            case SPV_MSG_FATAL:
            case SPV_MSG_INTERNAL_ERROR:
            case SPV_MSG_ERROR:
                cvk_error(msgtpl, message, position.index);
                break;
            case SPV_MSG_WARNING:
                cvk_warn(msgtpl, message, position.index);
                break;
            case SPV_MSG_INFO:
                cvk_info(msgtpl, message, position.index);
                break;
            case SPV_MSG_DEBUG:
                cvk_debug(msgtpl, message, position.index);
                break;
            }
#undef msgtpl
        };

    context.SetMessageConsumer(consumer);

    // Library creation
    bool create_library = false;
    if (m_build_options.find("-create-library") != std::string::npos) {
        cvk_info_fn("creating a library");
        create_library = true;
        m_binary_type = CL_PROGRAM_BINARY_TYPE_LIBRARY;
    } else {
        m_binary_type = CL_PROGRAM_BINARY_TYPE_EXECUTABLE;
    }

    // Link binaries
    cvk_debug_fn("copying input programs...");
    for (cl_uint i = 0; i < m_num_input_programs; i++) {
        cvk_debug_fn("program %u, %zu kernels...", i,
                     m_input_programs[i]->m_binary.num_kernels());
        cvk_debug_fn("about to copy code, size = %zu",
                     m_input_programs[i]->m_binary.code().size());
        binaries[i] = m_input_programs[i]->m_binary.code();
    }

    cvk_debug_fn("linking...");
    spvtools::LinkerOptions linker_options;
    linker_options.SetCreateLibrary(create_library);
    spv_result_t res =
        spvtools::Link(context, binaries, &linked, linker_options);

    if (res != SPV_SUCCESS) {
        return CL_BUILD_ERROR;
    }

    // Optimise linked binary
    spvtools::Optimizer opt(SPV_ENV_VULKAN_1_0);
    opt.SetMessageConsumer(consumer);
    opt.RegisterPass(spvtools::CreateInlineExhaustivePass());

    std::vector<uint32_t> linked_opt;

    if (!opt.Run(linked.data(), linked.size(), &linked_opt)) {
        cvk_error_fn("couldn't optimise linked SPIR-V module");
        return CL_BUILD_ERROR;
    }

    m_binary.use(std::move(linked_opt));

    // Load descriptor map
    if (!m_binary.load_descriptor_map()) {
        cvk_error("Could not load descriptor map for SPIR-V binary.");
        return CL_BUILD_ERROR;
    }

    cvk_debug_fn("linked binary has %zu kernels",
                 m_binary.kernels_arguments().size());

    return CL_BUILD_SUCCESS;
}

void cvk_program::prepare_push_constant_range() {
    auto& pcs = m_binary.push_constants();

    uint32_t min_offset = UINT32_MAX;
    uint32_t max_offset = 0, max_offset_size = 0;

    for (auto& pc_pcd : pcs) {
        auto pcd = pc_pcd.second;
        min_offset = std::min(min_offset, pcd.offset);
        if (pcd.offset >= max_offset) {
            max_offset = pcd.offset;
            max_offset_size = pcd.size;
        }
    }

    m_push_constant_range = {VK_SHADER_STAGE_COMPUTE_BIT, min_offset,
                             max_offset + max_offset_size};
}

bool cvk_program::check_capabilities(const cvk_device* device) const {
    // Get list of required SPIR-V capabilities.
    std::vector<spv::Capability> capabilities;
    if (!m_binary.get_capabilities(capabilities)) {
        cvk_error("Failed to get required SPIR-V capabilities.");
        return false;
    }

    // Check that each required capability is supported by the device.
    for (auto c : capabilities) {
        cvk_info_fn("Program requires SPIR-V capability %d.", c);
        if (!device->supports_capability(c)) {
            // TODO: propagate this message to the build log
            cvk_error_fn("Device does not support SPIR-V capability %d.", c);
            return false;
        }
    }
    return true;
}

void cvk_program::do_build() {
    cl_build_status status = CL_BUILD_SUCCESS;

    // Destroy entry points from previous build
    m_entry_points.clear();

    auto device = m_context->device();

    switch (m_operation) {
    case build_operation::compile:
    case build_operation::build:
        // Compile source and load binary
        if (!m_binary.loaded_from_binary()) {
            status = compile_source(device);
        }
        break;
    case build_operation::link:
        status = link();
        break;
    }

    prepare_push_constant_range();

    if ((m_operation == build_operation::compile) ||
        (status != CL_BUILD_SUCCESS)) {
        complete_operation(device, status);
        return;
    }

    bool cache_hit =
        device->get_pipeline_cache(m_binary.code(), m_pipeline_cache);
    if (m_pipeline_cache == VK_NULL_HANDLE) {
        complete_operation(device, CL_BUILD_ERROR);
        return;
    }

    if (!cache_hit) {
        // Validate
        // TODO validate with different rules depending on the binary type
        if ((m_binary_type == CL_PROGRAM_BINARY_TYPE_EXECUTABLE) &&
            !m_binary.validate()) {
            cvk_error("Could not validate SPIR-V binary.");
            complete_operation(device, CL_BUILD_ERROR);
            return;
        }
    }

    // Check capabilities against the device.
    char* skip_capability_check_env =
        getenv("CLVK_SKIP_SPIRV_CAPABILITY_CHECK");
    bool skip_capability_check = false;
    if (skip_capability_check_env &&
        strcmp(skip_capability_check_env, "1") == 0)
        skip_capability_check = true;
    if ((m_binary_type == CL_PROGRAM_BINARY_TYPE_EXECUTABLE) &&
        !skip_capability_check && !check_capabilities(device)) {
        cvk_error("Missing support for required SPIR-V capabilities.");
        complete_operation(device, CL_BUILD_ERROR);
        return;
    }

    // Create literal samplers
    for (auto const& desc : literal_sampler_descs()) {
        auto sampler =
            cvk_sampler::create(context(), desc.normalized_coords,
                                desc.addressing_mode, desc.filter_mode);
        if (sampler == nullptr) {
            complete_operation(device, CL_BUILD_ERROR);
            return;
        }
        m_literal_samplers.emplace_back(sampler);
    }

    // Strip the reflection information if non-semantic info is not supported
    // by the Vulkan implementation. This stripped binary is stored separately
    // from |m_binary| because clvk needs to be able to provide the binary with
    // reflection information for clGetProgramInfo.
    const uint32_t* spir_data = m_binary.spir_data();
    size_t spir_size = m_binary.spir_size();
    if (!device->is_vulkan_extension_enabled(
            VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME)) {
        if (!m_binary.strip_reflection(&m_stripped_binary)) {
            cvk_error_fn("couldn't strip reflection from SPIR-V module");
            complete_operation(device, CL_BUILD_ERROR);
            return;
        }
        spir_data = m_stripped_binary.data();
        spir_size = m_stripped_binary.size() * sizeof(uint32_t);
    }

    // Create a shader module
    VkDevice dev = device->vulkan_device();

    VkShaderModuleCreateInfo moduleCreateInfo = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, // sType
        nullptr,                                     // pNext
        0,                                           // flags
        spir_size,                                   // codeSize
        spir_data                                    // pCode
    };

    VkResult res =
        vkCreateShaderModule(dev, &moduleCreateInfo, nullptr, &m_shader_module);

    if (res != VK_SUCCESS) {
        cvk_error("vkCreateShaderModule returned %d", res);
        complete_operation(device, CL_BUILD_ERROR);
        return;
    }

    complete_operation(device, CL_BUILD_SUCCESS);
}

bool cvk_program::build(build_operation operation, cl_uint num_devices,
                        const cl_device_id* device_list, const char* options,
                        cl_uint num_input_programs,
                        const cl_program* input_programs,
                        const char** header_include_names,
                        cvk_program_callback cb, void* data) {
    std::lock_guard<std::mutex> lock(m_lock);

    // Check if there is already a build in progress
    // TODO: Allow concurrent builds targeting different devices
    if (std::count_if(m_dev_status.begin(), m_dev_status.end(),
                      [](auto& status) {
                          return status.second == CL_BUILD_IN_PROGRESS;
                      })) {
        return false;
    }

    retain();

    for (cl_uint i = 0; i < num_input_programs; i++) {
        cvk_program* iprog =
            const_cast<cvk_program*>(icd_downcast(input_programs[i]));
        iprog->retain();
        m_input_programs.push_back(iprog);
        if (header_include_names != nullptr) {
            m_header_include_names.push_back(header_include_names[i]);
        }
    }

    // Mark build in-progress and save devices
    if (num_devices == 0) {
        m_num_devices = 1u;
        m_dev_status[m_context->device()] = CL_BUILD_IN_PROGRESS;
    } else {
        m_num_devices = num_devices;
        for (cl_uint i = 0; i < num_devices; i++) {
            m_dev_status[icd_downcast(device_list[i])] = CL_BUILD_IN_PROGRESS;
        }
    }

    if (options != nullptr) {
        m_build_options = options;
    }
    m_num_input_programs = num_input_programs;
    m_operation = operation;
    m_operation_callback = cb;
    m_operation_callback_data = data;

    // Kick off build
    m_thread = std::make_unique<std::thread>(&cvk_program::do_build, this);
    if (cb) {
        m_thread->detach();
    }

    return true;
}

cvk_entry_point::cvk_entry_point(VkDevice dev, cvk_program* program,
                                 const std::string& name)
    : m_device(dev), m_context(program->context()), m_program(program),
      m_name(name), m_pod_descriptor_type(VK_DESCRIPTOR_TYPE_MAX_ENUM),
      m_pod_buffer_size(0u), m_has_pod_arguments(false),
      m_has_pod_buffer_arguments(false), m_descriptor_pool(VK_NULL_HANDLE),
      m_pipeline_layout(VK_NULL_HANDLE) {}

cvk_entry_point* cvk_program::get_entry_point(std::string& name,
                                              cl_int* errcode_ret) {
    std::lock_guard<std::mutex> lock(m_lock);

    // Check for existing entry point in cache
    if (m_entry_points.count(name)) {
        *errcode_ret = CL_SUCCESS;
        return m_entry_points.at(name).get();
    }

    // Create and initialize entry point
    cvk_entry_point* entry_point =
        new cvk_entry_point(m_context->device()->vulkan_device(), this, name);
    *errcode_ret = entry_point->init();
    if (*errcode_ret != CL_SUCCESS) {
        delete entry_point;
        return nullptr;
    }

    // Add to cache for reuse by other kernels
    m_entry_points.insert(
        {name, std::unique_ptr<cvk_entry_point>(entry_point)});

    return entry_point;
}

bool cvk_entry_point::build_descriptor_set_layout(
    const std::vector<VkDescriptorSetLayoutBinding>& bindings) {
    VkDescriptorSetLayoutCreateInfo createInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr,
        0,                                      // flags
        static_cast<uint32_t>(bindings.size()), // bindingCount
        bindings.data()                         // pBindings
    };

    VkResult res;
    if (bindings.size() > 0) {
        VkDescriptorSetLayout setLayout;
        res = vkCreateDescriptorSetLayout(m_device, &createInfo, 0, &setLayout);
        if (res != VK_SUCCESS) {
            cvk_error("Could not create descriptor set layout");
            return false;
        }
        m_descriptor_set_layouts.push_back(setLayout);
    }

    return true;
}

bool cvk_entry_point::build_descriptor_sets_layout_bindings_for_arguments(
    binding_stat_map& smap, uint32_t& num_resource_slots) {
    bool pod_found = false;

    uint32_t highest_binding = 0;

    std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
    for (auto& arg : m_args) {
        VkDescriptorType dt = VK_DESCRIPTOR_TYPE_MAX_ENUM;

        switch (arg.kind) {
        case kernel_argument_kind::buffer:
            dt = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            break;
        case kernel_argument_kind::buffer_ubo:
            dt = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            break;
        case kernel_argument_kind::ro_image:
            dt = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            break;
        case kernel_argument_kind::wo_image:
            dt = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            break;
        case kernel_argument_kind::sampler:
            dt = VK_DESCRIPTOR_TYPE_SAMPLER;
            break;
        case kernel_argument_kind::local:
            continue;
        case kernel_argument_kind::pod:
        case kernel_argument_kind::pod_ubo:
            if (!pod_found) {
                if (arg.kind == kernel_argument_kind::pod) {
                    dt = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                } else if (arg.kind == kernel_argument_kind::pod_ubo) {
                    dt = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                }

                m_pod_descriptor_type = dt;

                pod_found = true;
            } else {
                continue;
            }
            break;
        case kernel_argument_kind::pod_pushconstant:
            continue;
        }

        VkDescriptorSetLayoutBinding binding = {
            arg.binding,                 // binding
            dt,                          // descriptorType
            1,                           // decriptorCount
            VK_SHADER_STAGE_COMPUTE_BIT, // stageFlags
            nullptr                      // pImmutableSamplers
        };

        highest_binding = std::max(arg.binding, highest_binding);

        layoutBindings.push_back(binding);
        smap[binding.descriptorType]++;
    }

    num_resource_slots = highest_binding + 1;

    if (!build_descriptor_set_layout(layoutBindings)) {
        return false;
    }

    return true;
}

bool cvk_entry_point::
    build_descriptor_sets_layout_bindings_for_literal_samplers(
        binding_stat_map& smap) {

    std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
    for (auto& desc : m_program->literal_sampler_descs()) {
        VkDescriptorSetLayoutBinding binding = {
            desc.binding,                // binding
            VK_DESCRIPTOR_TYPE_SAMPLER,  // descriptorType
            1,                           // decriptorCount
            VK_SHADER_STAGE_COMPUTE_BIT, // stageFlags
            nullptr                      // pImmutableSamplers
        };
        layoutBindings.push_back(binding);
        smap[binding.descriptorType]++;
    }

    if (!build_descriptor_set_layout(layoutBindings)) {
        return false;
    }

    return true;
}

cl_int cvk_entry_point::init() {
    VkResult res;

    // Get a pointer to the arguments from the program
    auto args = m_program->args_for_kernel(m_name);

    if (args == nullptr) {
        cvk_error("Kernel %s doesn't exist in program", m_name.c_str());
        return CL_INVALID_KERNEL_NAME;
    }

    // Store a sorted copy of the arguments
    m_args = *args;
    std::sort(
        m_args.begin(), m_args.end(),
        [](kernel_argument a, kernel_argument b) { return a.pos < b.pos; });

    // Create Descriptor Sets Layout
    std::unordered_map<VkDescriptorType, uint32_t> bindingTypes;
    if (!build_descriptor_sets_layout_bindings_for_literal_samplers(
            bindingTypes)) {
        return CL_INVALID_VALUE;
    }
    if (!build_descriptor_sets_layout_bindings_for_arguments(
            bindingTypes, m_num_resource_slots)) {
        return CL_INVALID_VALUE;
    }

    // Do we have POD arguments?
    for (auto& arg : m_args) {
        if (arg.is_pod()) {
            m_has_pod_arguments = true;
            if (arg.is_pod_buffer()) {
                m_has_pod_buffer_arguments = true;
            }
        }
    }

    // Calculate POD buffer size and update the push constant range.
    VkPushConstantRange push_constant_range = m_program->push_constant_range();
    if (m_has_pod_arguments) {
        // Check we know the POD buffer's descriptor type
        if (m_has_pod_buffer_arguments &&
            m_pod_descriptor_type == VK_DESCRIPTOR_TYPE_MAX_ENUM) {
            return CL_INVALID_PROGRAM;
        }

        // Find how big the POD buffer should be
        uint32_t max_offset = 0;
        uint32_t max_offset_arg_size = 0;

        for (auto& arg : m_args) {
            if (arg.is_pod()) {
                if (arg.offset >= max_offset) {
                    max_offset = arg.offset;
                    max_offset_arg_size = arg.size;
                }
                if (!arg.is_pod_buffer()) {
                    if (arg.offset < push_constant_range.offset) {
                        push_constant_range.offset = arg.offset;
                    }

                    if (arg.offset + arg.size >
                        push_constant_range.offset + push_constant_range.size) {
                        push_constant_range.size =
                            arg.offset + arg.size - push_constant_range.offset;
                    }
                }
            }
        }

        m_pod_buffer_size = max_offset + max_offset_arg_size;
        m_pod_buffer_size = round_up(m_pod_buffer_size, 4);
    }

    // Don't pass the range at pipeline layout creation time if no push
    // constants are used
    uint32_t num_push_constant_ranges = 1;
    if (push_constant_range.offset == UINT32_MAX) {
        num_push_constant_ranges = 0;
    }

    // The size of the range must be a multiple of 4, round up to guarantee this
    push_constant_range.size = round_up(push_constant_range.size, 4);

    // Its offset must be a multiple of 4, round down to guarantee this
    push_constant_range.offset &= ~0x3U;

    // Create pipeline layout
    cvk_debug("about to create pipeline layout, number of descriptor set "
              "layouts: %zu",
              m_descriptor_set_layouts.size());

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        0,
        0,
        static_cast<uint32_t>(m_descriptor_set_layouts.size()),
        m_descriptor_set_layouts.data(),
        num_push_constant_ranges,
        &push_constant_range};

    res = vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, 0,
                                 &m_pipeline_layout);
    if (res != VK_SUCCESS) {
        cvk_error("Could not create pipeline layout.");
        return CL_INVALID_VALUE;
    }

    // Determine number and types of bindings
    std::vector<VkDescriptorPoolSize> poolSizes(bindingTypes.size());

    int bidx = 0;
    for (auto& bt : bindingTypes) {
        poolSizes[bidx].type = bt.first;
        poolSizes[bidx].descriptorCount = bt.second * MAX_INSTANCES;
        bidx++;
    }

    // Create descriptor pool
    if (poolSizes.size() > 0) {
        VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            nullptr,
            VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT, // flags
            MAX_INSTANCES * spir_binary::MAX_DESCRIPTOR_SETS,  // maxSets
            static_cast<uint32_t>(poolSizes.size()),           // poolSizeCount
            poolSizes.data(),                                  // pPoolSizes
        };

        res = vkCreateDescriptorPool(m_device, &descriptorPoolCreateInfo, 0,
                                     &m_descriptor_pool);

        if (res != VK_SUCCESS) {
            cvk_error("Could not create descriptor pool.");
            return CL_INVALID_VALUE;
        }
    }

    return CL_SUCCESS;
}

VkPipeline
cvk_entry_point::create_pipeline(const cvk_spec_constant_map& spec_constants) {
    std::lock_guard<std::mutex> lock(m_pipeline_cache_lock);

    // Check for a cached pipeline using the same specialization constants
    if (m_pipelines.count(spec_constants)) {
        VkPipeline pipeline = m_pipelines.at(spec_constants);
        cvk_info("reusing pipeline %p for kernel %s", pipeline, m_name.c_str());
        return pipeline;
    }

    std::vector<VkSpecializationMapEntry> mapEntries;
    std::vector<uint32_t> specConstantData;
    uint32_t constantDataOffset = 0;
    for (auto& spec_const : spec_constants) {
        VkSpecializationMapEntry entry = {spec_const.first, constantDataOffset,
                                          sizeof(uint32_t)};
        mapEntries.push_back(entry);
        specConstantData.push_back(spec_const.second);
        constantDataOffset += sizeof(uint32_t);
    }

    VkSpecializationInfo specializationInfo = {
        static_cast<uint32_t>(mapEntries.size()),
        mapEntries.data(),
        specConstantData.size() * sizeof(uint32_t),
        specConstantData.data(),
    };

    const VkComputePipelineCreateInfo createInfo = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, // sType
        nullptr,                                        // pNext
        0,                                              // flags
        {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, // sType
            nullptr,                                             // pNext
            0,                                                   // flags
            VK_SHADER_STAGE_COMPUTE_BIT,                         // stage
            m_program->shader_module(),                          // module
            m_name.c_str(),
            &specializationInfo // pSpecializationInfo
        },                      // stage
        m_pipeline_layout,      // layout
        VK_NULL_HANDLE,         // basePipelineHandle
        0                       // basePipelineIndex
    };

    VkPipeline pipeline;
    VkResult res =
        vkCreateComputePipelines(m_device, m_program->pipeline_cache(), 1,
                                 &createInfo, nullptr, &pipeline);

    if (res != VK_SUCCESS) {
        cvk_error_fn("Could not create compute pipeline: %s",
                     vulkan_error_string(res));
        return VK_NULL_HANDLE;
    }

    // Add to pipeline cache
    m_pipelines[spec_constants] = pipeline;

    cvk_info("created pipeline %p for kernel %s", pipeline, m_name.c_str());

    return pipeline;
}

bool cvk_entry_point::allocate_descriptor_sets(VkDescriptorSet* ds) {

    if (m_descriptor_set_layouts.size() == 0) {
        return true;
    }

    std::lock_guard<std::mutex> lock(m_descriptor_pool_lock);

    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
        m_descriptor_pool,
        static_cast<uint32_t>(
            m_descriptor_set_layouts.size()), // descriptorSetCount
        m_descriptor_set_layouts.data()};

    VkResult res =
        vkAllocateDescriptorSets(m_device, &descriptorSetAllocateInfo, ds);

    if (res != VK_SUCCESS) {
        cvk_error_fn("could not allocate descriptor sets: %s",
                     vulkan_error_string(res));
        return false;
    }

    return true;
}

std::unique_ptr<cvk_buffer> cvk_entry_point::allocate_pod_buffer() {
    cl_int err;
    auto buffer =
        cvk_buffer::create(m_context, 0, m_pod_buffer_size, nullptr, &err);
    if (err != CL_SUCCESS) {
        return nullptr;
    }

    return buffer;
}
