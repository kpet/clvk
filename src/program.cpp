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
#include <filesystem>
#include <iostream>
#include <map>
#include <sstream>
#include <system_error>
#include <utility>
#include <vector>

#include <vulkan/vulkan.h>

#include "clspv/Sampler.h"
#include "utils.hpp"

#ifdef CLSPV_ONLINE_COMPILER
#ifdef ENABLE_SPIRV_IL
#include "LLVMSPIRVLib.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Support/CommandLine.h" // FIXME(#380) remove
#include "llvm/Support/raw_ostream.h"
#endif
#include "clspv/Compiler.h"
#endif

#include "spirv-tools/linker.hpp"
#include "spirv-tools/optimizer.hpp"
#include "spirv/unified1/NonSemanticClspvReflection.h"
#include "spirv/unified1/spirv.hpp"

#include "config.hpp"
#include "init.hpp"
#include "log.hpp"
#include "program.hpp"
#include "tracing.hpp"

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
    pos_type seekoff(off_type off, std::ios_base::seekdir dir,
                     std::ios_base::openmode which) override {
        UNUSED(which);
        char* whence = eback();
        if (dir == std::ios_base::cur) {
            whence = gptr();
        } else if (dir == std::ios_base::end) {
            whence = egptr();
        }
        char* to = whence + off;
        if (to >= eback() && to <= egptr()) {
            setg(eback(), to, egptr());
            return gptr() - eback();
        }
        return -1;
    }

    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override {
        UNUSED(which);
        char* to = eback() + pos;
        if (to >= eback() && to <= egptr()) {
            setg(eback(), to, egptr());
            return gptr() - eback();
        }
        return -1;
    }
};

struct reflection_parse_data {
    uint32_t uint_id = 0;
    std::unordered_map<uint32_t, uint32_t> constants;
    std::unordered_map<uint32_t, std::string> strings;
    spir_binary* binary;
    std::unordered_map<uint32_t, kernel_argument_info> arg_infos;
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
        case NonSemanticClspvReflectionArgumentPointerUniform:
            return kernel_argument_kind::pointer_ubo;
        case NonSemanticClspvReflectionArgumentPointerPushConstant:
            return kernel_argument_kind::pointer_pushconstant;
        case NonSemanticClspvReflectionArgumentSampledImage:
            return kernel_argument_kind::sampled_image;
        case NonSemanticClspvReflectionArgumentStorageImage:
            return kernel_argument_kind::storage_image;
        case NonSemanticClspvReflectionArgumentStorageTexelBuffer:
            return kernel_argument_kind::storage_texel_buffer;
        case NonSemanticClspvReflectionArgumentUniformTexelBuffer:
            return kernel_argument_kind::uniform_texel_buffer;
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
        case NonSemanticClspvReflectionImageArgumentInfoChannelOrderPushConstant:
            return pushconstant::image_metadata;
        case NonSemanticClspvReflectionImageArgumentInfoChannelDataTypePushConstant:
            return pushconstant::image_metadata;
        case NonSemanticClspvReflectionConstantDataPointerPushConstant:
            return pushconstant::module_constants_pointer;
        case NonSemanticClspvReflectionPrintfBufferPointerPushConstant:
            return pushconstant::printf_buffer_pointer;
        case NonSemanticClspvReflectionNormalizedSamplerMaskPushConstant:
            return pushconstant::normalized_sampler_mask;
        default:
            cvk_error_fn("Unhandled reflection instruction for push constant");
            break;
        }
        return pushconstant::global_offset;
    };

    auto* parse_data = reinterpret_cast<reflection_parse_data*>(user_data);
    switch (inst->opcode) {
    case spv::OpTypeInt:
        if (inst->words[2] == 32 && inst->words[3] == 0) {
            parse_data->uint_id = inst->result_id;
        }
        break;
    case spv::OpConstant:
        if (inst->words[1] == parse_data->uint_id) {
            parse_data->constants[inst->result_id] = inst->words[3];
        }
        break;
    case spv::OpString:
        parse_data->strings[inst->result_id] =
            std::string(reinterpret_cast<const char*>(&inst->words[2]));
        break;
    case spv::OpExtInst:
        if (inst->ext_inst_type ==
            SPV_EXT_INST_TYPE_NONSEMANTIC_CLSPVREFLECTION) {
            auto ext_inst = inst->words[4];
            switch (ext_inst) {
            case NonSemanticClspvReflectionKernel: {
                // Record the kernel name.
                const auto& name = parse_data->strings[inst->words[6]];

                const auto& num_args = parse_data->constants[inst->words[7]];
                const auto& flags = parse_data->constants[inst->words[8]];
                const auto& attributes = parse_data->strings[inst->words[9]];

                parse_data->strings[inst->result_id] = name;
                parse_data->binary->add_kernel(name, num_args, attributes,
                                               flags);
                break;
            }
            case NonSemanticClspvReflectionArgumentInfo: {
                // Record the argument info.
                kernel_argument_info info;
                info.name = parse_data->strings[inst->words[5]];
                if (inst->num_operands > 6) {
                    info.type_name = parse_data->strings[inst->words[6]];
                    info.address_qualifier =
                        parse_data->constants[inst->words[7]];
                    info.access_qualifier =
                        parse_data->constants[inst->words[8]];
                    info.type_qualifier = parse_data->constants[inst->words[9]];
                    info.extended_valid = true;
                }
                parse_data->arg_infos[inst->result_id] = info;
                break;
            }
            case NonSemanticClspvReflectionNormalizedSamplerMaskPushConstant: {
                auto kernel = parse_data->strings[inst->words[5]];
                auto ordinal = parse_data->constants[inst->words[6]];
                auto offset = parse_data->constants[inst->words[7]];
                auto size = parse_data->constants[inst->words[8]];
                parse_data->binary->add_sampler_metadata(kernel, ordinal,
                                                         offset);
                auto pc = inst_to_push_constant(ext_inst);
                parse_data->binary->add_push_constant(pc, {offset, size});
                break;
            }
            case NonSemanticClspvReflectionImageArgumentInfoChannelOrderPushConstant: {
                auto kernel = parse_data->strings[inst->words[5]];
                auto ordinal = parse_data->constants[inst->words[6]];
                auto offset = parse_data->constants[inst->words[7]];
                auto size = parse_data->constants[inst->words[8]];
                parse_data->binary->add_image_channel_order_metadata(
                    kernel, ordinal, offset);
                auto pc = inst_to_push_constant(ext_inst);
                parse_data->binary->add_push_constant(pc, {offset, size});
                break;
            }
            case NonSemanticClspvReflectionImageArgumentInfoChannelDataTypePushConstant: {
                auto kernel = parse_data->strings[inst->words[5]];
                auto ordinal = parse_data->constants[inst->words[6]];
                auto offset = parse_data->constants[inst->words[7]];
                auto size = parse_data->constants[inst->words[8]];
                parse_data->binary->add_image_channel_data_type_metadata(
                    kernel, ordinal, offset);
                auto pc = inst_to_push_constant(ext_inst);
                parse_data->binary->add_push_constant(pc, {offset, size});
                break;
            }
            case NonSemanticClspvReflectionArgumentStorageBuffer:
            case NonSemanticClspvReflectionArgumentUniform:
            case NonSemanticClspvReflectionArgumentSampledImage:
            case NonSemanticClspvReflectionArgumentStorageImage:
            case NonSemanticClspvReflectionArgumentStorageTexelBuffer:
            case NonSemanticClspvReflectionArgumentUniformTexelBuffer:
            case NonSemanticClspvReflectionArgumentSampler: {
                // These arguments have descriptor set, binding and an optional
                // arg info.
                auto kernel = parse_data->strings[inst->words[5]];
                auto ordinal = parse_data->constants[inst->words[6]];
                auto descriptor_set = parse_data->constants[inst->words[7]];
                if (descriptor_set >= spir_binary::MAX_DESCRIPTOR_SETS)
                    return SPV_ERROR_INVALID_DATA;
                auto binding = parse_data->constants[inst->words[8]];
                kernel_argument_info arg_info;
                if (inst->num_operands == 9) {
                    arg_info = parse_data->arg_infos[inst->words[9]];
                }
                auto kind = inst_to_arg_kind(ext_inst);
                kernel_argument arg = {arg_info, ordinal, descriptor_set,
                                       binding,  0,       0,
                                       kind,     0,       0};
                parse_data->binary->add_kernel_argument(kernel, std::move(arg));
                break;
            }
            case NonSemanticClspvReflectionArgumentPodStorageBuffer:
            case NonSemanticClspvReflectionArgumentPodUniform:
            case NonSemanticClspvReflectionArgumentPointerUniform: {
                // These arguments have descriptor set, binding, offset, size
                // and an optional arg info.
                auto kernel = parse_data->strings[inst->words[5]];
                auto ordinal = parse_data->constants[inst->words[6]];
                auto descriptor_set = parse_data->constants[inst->words[7]];
                if (descriptor_set >= spir_binary::MAX_DESCRIPTOR_SETS)
                    return SPV_ERROR_INVALID_DATA;
                auto binding = parse_data->constants[inst->words[8]];
                auto offset = parse_data->constants[inst->words[9]];
                auto size = parse_data->constants[inst->words[10]];
                kernel_argument_info arg_info;
                if (inst->num_operands == 11) {
                    arg_info = parse_data->arg_infos[inst->words[11]];
                }
                auto kind = inst_to_arg_kind(ext_inst);
                kernel_argument arg = {arg_info, ordinal, descriptor_set,
                                       binding,  offset,  size,
                                       kind,     0,       0};
                parse_data->binary->add_kernel_argument(kernel, std::move(arg));
                break;
            }
            case NonSemanticClspvReflectionArgumentPodPushConstant:
            case NonSemanticClspvReflectionArgumentPointerPushConstant: {
                // These arguments have offset, size and an optional arg info.
                auto kernel = parse_data->strings[inst->words[5]];
                auto ordinal = parse_data->constants[inst->words[6]];
                auto offset = parse_data->constants[inst->words[7]];
                auto size = parse_data->constants[inst->words[8]];
                kernel_argument_info arg_info;
                if (inst->num_operands == 9) {
                    arg_info = parse_data->arg_infos[inst->words[9]];
                }
                auto kind = inst_to_arg_kind(ext_inst);
                kernel_argument arg = {arg_info, ordinal, 0, 0, offset,
                                       size,     kind,    0, 0};
                parse_data->binary->add_kernel_argument(kernel, std::move(arg));
                break;
            }
            case NonSemanticClspvReflectionArgumentWorkgroup: {
                // These arguments have spec id, elem size and an optional arg
                // info.
                auto kernel = parse_data->strings[inst->words[5]];
                auto ordinal = parse_data->constants[inst->words[6]];
                auto spec_id = parse_data->constants[inst->words[7]];
                auto size = parse_data->constants[inst->words[8]];
                kernel_argument_info arg_info;
                if (inst->num_operands == 9) {
                    arg_info = parse_data->arg_infos[inst->words[9]];
                }
                auto kind = inst_to_arg_kind(ext_inst);
                kernel_argument arg = {arg_info, ordinal, 0,       0,   0,
                                       0,        kind,    spec_id, size};
                parse_data->binary->add_kernel_argument(kernel, std::move(arg));
                break;
            }
            case NonSemanticClspvReflectionSpecConstantWorkgroupSize: {
                // Reflection encodes all three spec ids in a single
                // instruction.
                auto x_id = parse_data->constants[inst->words[5]];
                auto y_id = parse_data->constants[inst->words[6]];
                auto z_id = parse_data->constants[inst->words[7]];
                parse_data->binary->add_spec_constant(
                    spec_constant::workgroup_size_x, x_id);
                parse_data->binary->add_spec_constant(
                    spec_constant::workgroup_size_y, y_id);
                parse_data->binary->add_spec_constant(
                    spec_constant::workgroup_size_z, z_id);
                break;
            }
            case NonSemanticClspvReflectionSpecConstantGlobalOffset: {
                // Reflection encodes all three spec ids in a single
                // instruction.
                auto x_id = parse_data->constants[inst->words[5]];
                auto y_id = parse_data->constants[inst->words[6]];
                auto z_id = parse_data->constants[inst->words[7]];
                parse_data->binary->add_spec_constant(
                    spec_constant::global_offset_x, x_id);
                parse_data->binary->add_spec_constant(
                    spec_constant::global_offset_y, y_id);
                parse_data->binary->add_spec_constant(
                    spec_constant::global_offset_z, z_id);
                break;
            }
            case NonSemanticClspvReflectionSpecConstantWorkDim: {
                auto dim_id = parse_data->constants[inst->words[5]];
                parse_data->binary->add_spec_constant(spec_constant::work_dim,
                                                      dim_id);
                break;
            }
            case NonSemanticClspvReflectionSpecConstantSubgroupMaxSize: {
                auto size_id = parse_data->constants[inst->words[5]];
                parse_data->binary->add_spec_constant(
                    spec_constant::subgroup_max_size, size_id);
                break;
            }
            case NonSemanticClspvReflectionPushConstantGlobalOffset:
            case NonSemanticClspvReflectionPushConstantEnqueuedLocalSize:
            case NonSemanticClspvReflectionPushConstantGlobalSize:
            case NonSemanticClspvReflectionPushConstantRegionOffset:
            case NonSemanticClspvReflectionPushConstantNumWorkgroups:
            case NonSemanticClspvReflectionPushConstantRegionGroupOffset: {
                auto offset = parse_data->constants[inst->words[5]];
                auto size = parse_data->constants[inst->words[6]];
                auto pc = inst_to_push_constant(ext_inst);
                parse_data->binary->add_push_constant(pc, {offset, size});
                break;
            }
            case NonSemanticClspvReflectionLiteralSampler: {
                // Track descriptor set and binding. Decode the sampler mask.
                auto descriptor_set = parse_data->constants[inst->words[5]];
                if (descriptor_set >= spir_binary::MAX_DESCRIPTOR_SETS)
                    return SPV_ERROR_INVALID_DATA;
                auto binding = parse_data->constants[inst->words[6]];
                auto mask = parse_data->constants[inst->words[7]];
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
                parse_data->binary->add_literal_sampler(
                    {descriptor_set, binding, normalized_coords, addressing,
                     filter});
                break;
            }
            case NonSemanticClspvReflectionPropertyRequiredWorkgroupSize: {
                auto kernel = parse_data->strings[inst->words[5]];
                auto x = parse_data->constants[inst->words[6]];
                auto y = parse_data->constants[inst->words[7]];
                auto z = parse_data->constants[inst->words[8]];
                parse_data->binary->set_required_work_group_size(kernel, x, y,
                                                                 z);
                break;
            }
            case NonSemanticClspvReflectionConstantDataStorageBuffer:
            case NonSemanticClspvReflectionConstantDataPointerPushConstant: {

                auto char2int = [](char c) {
                    if (c >= '0' && c <= '9')
                        return c - '0';
                    if (c >= 'A' && c <= 'F')
                        return c - 'A' + 10;
                    if (c >= 'a' && c <= 'f')
                        return c - 'a' + 10;
                    return 0;
                };

                auto hex2bin = [&char2int](const char* str, char* bin) {
                    while (str[0] && str[1]) {
                        *(bin++) = char2int(str[0]) * 16 + char2int(str[1]);
                        str += 2;
                    }
                };

                auto data = parse_data->strings[inst->words[7]];
                if (data.size() & 1) {
                    cvk_error_fn("invalid constant data buffer string (odd "
                                 "number of digits)");
                    return SPV_ERROR_INVALID_DATA;
                }
                constant_data_buffer_info binfo{};
                auto data_size = data.size() / 2;
                binfo.data.resize(data_size);
                hex2bin(data.c_str(), binfo.data.data());
                if (ext_inst ==
                    NonSemanticClspvReflectionConstantDataStorageBuffer) {
                    binfo.type = module_buffer_type::storage_buffer;
                    binfo.set = parse_data->constants[inst->words[5]];
                    if (binfo.set >= spir_binary::MAX_DESCRIPTOR_SETS)
                        return SPV_ERROR_INVALID_DATA;
                    binfo.binding = parse_data->constants[inst->words[6]];

                } else {
                    binfo.type = module_buffer_type::pointer_push_constant;
                    binfo.pc_offset = parse_data->constants[inst->words[5]];
                    parse_data->binary->add_push_constant(
                        pushconstant::module_constants_pointer,
                        {binfo.pc_offset, 8u});
                }
                parse_data->binary->set_constant_data_buffer(binfo);
                break;
            }
            case NonSemanticClspvReflectionPrintfInfo: {
                uint32_t printf_id = parse_data->constants[inst->words[5]];
                std::string printf_string = parse_data->strings[inst->words[6]];
                std::vector<uint32_t> printf_arg_sizes;
                for (int i = 6; i < inst->num_operands; i++) {
                    printf_arg_sizes.push_back(
                        parse_data
                            ->constants[inst->words[inst->operands[i].offset]]);
                }
                parse_data->binary->add_printf_descriptor(
                    {printf_id, printf_string, printf_arg_sizes});
                break;
            }
            case NonSemanticClspvReflectionPrintfBufferStorageBuffer: {
                printf_buffer_desc_info binfo;
                binfo.type = module_buffer_type::storage_buffer;
                binfo.set = parse_data->constants[inst->words[5]];
                binfo.binding = parse_data->constants[inst->words[6]];
                binfo.size = parse_data->constants[inst->words[7]];
                parse_data->binary->set_printf_buffer_info(binfo);
                break;
            }
            case NonSemanticClspvReflectionPrintfBufferPointerPushConstant: {
                printf_buffer_desc_info binfo;
                binfo.type = module_buffer_type::pointer_push_constant;
                binfo.pc_offset = parse_data->constants[inst->words[5]];
                binfo.size = parse_data->constants[inst->words[7]];
                parse_data->binary->set_printf_buffer_info(binfo);
                parse_data->binary->add_push_constant(
                    pushconstant::printf_buffer_pointer, {binfo.pc_offset, 8u});
                break;
            }
            default:
                return SPV_ERROR_INVALID_DATA;
            }
        }
        break;
    default:
        break;
    }

    return SPV_SUCCESS;
}

bool spir_binary::load(const char* fname) {
    std::ifstream ifile;

    ifile.open(fname, std::ios::in | std::ios::binary);

    if (!ifile.is_open()) {
        cvk_error("Failed to open %s", fname);
        return false;
    }

    ifile.seekg(0, std::ios::end);
    uint32_t size = ifile.tellg();
    ifile.seekg(0, std::ios::beg);

    return load(ifile, size);
}

bool spir_binary::load(std::istream& istream, uint32_t size) {
    m_code.assign(size / SPIR_WORD_SIZE, 0);
    istream.read(reinterpret_cast<char*>(m_code.data()), size);

    if (!istream.good()) {
        cvk_warn("Failed to load SPIR-V (size: %u)", size);
        return false;
    }

    return true;
}

bool spir_binary::read(const unsigned char* src, size_t size) {
    m_loaded_from_binary = true;
    membuf bufview(src, src + size);
    std::istream istream(&bufview);
    return load(istream, size);
}

bool spir_binary::save(std::ostream& ostream) const {
    ostream.write(reinterpret_cast<const char*>(m_code.data()), size());
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

size_t spir_binary::size() const { return m_code.size() * SPIR_WORD_SIZE; }

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

bool spir_binary::validate(const spirv_validation_options& val_options) const {
    spv_diagnostic diag;
    spv_validator_options options = spvValidatorOptionsCreate();
    spvValidatorOptionsSetUniformBufferStandardLayout(
        options, val_options.uniform_buffer_std_layout);
    spv_const_binary_t binary{m_code.data(), m_code.size()};
    spv_result_t res =
        spvValidateWithOptions(m_context, options, &binary, &diag);
    spvDiagnosticPrint(diag);
    spvDiagnosticDestroy(diag);
    spvValidatorOptionsDestroy(options);
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

#if COMPILER_AVAILABLE || USING_SWIFTSHADER
    spvtools::Optimizer opt(m_target_env);
    opt.SetMessageConsumer(consumer);
    opt.RegisterPass(spvtools::CreateStripReflectInfoPass());
    spvtools::OptimizerOptions options;
    options.set_run_validator(false);
    if (!opt.Run(m_code.data(), m_code.size(), stripped, options)) {
        return false;
    }
#else
    *stripped = m_code;
#endif
    return true;
}

bool spir_binary::load_descriptor_map() {
    reflection_parse_data parse_data;
    parse_data.binary = this;

    // TODO: The parser assumes a valid SPIR-V module, but validation is not
    // run until later.
    auto result =
        spvBinaryParse(m_context, &parse_data, m_code.data(), m_code.size(),
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

#if COMPILER_AVAILABLE
bool save_cstring_to_file(
    const std::string& fname, const char* data, size_t size,
    std::ios_base::openmode open_mode = std::ios_base::out) {
    std::filesystem::path ofname(fname);
    if (ofname.has_parent_path()) {
        std::error_code error;
        std::filesystem::create_directories(ofname.parent_path(), error);
        if (error) {
            cvk_error_fn("create_directories failed: '%s' (%u)",
                         error.message().c_str(), error.value());
            return false;
        }
    }
    std::ofstream ofile{fname, open_mode};

    if (!ofile.is_open()) {
        return false;
    }

    ofile.write(data, size);
    ofile.close();

    return ofile.good();
}

bool save_string_to_file(const std::string& fname, const std::string& text) {
    return save_cstring_to_file(fname, text.c_str(), text.size());
}

#ifndef CLSPV_ONLINE_COMPILER
bool save_il_to_file(const std::string& fname, const std::vector<uint8_t>& il) {
    return save_cstring_to_file(fname, reinterpret_cast<const char*>(il.data()),
                                il.size(), std::ios::binary);
}
#endif // CLSPV_ONLINE_COMPILER
#endif // COMPILER_AVAILABLE

struct temp_folder_deletion {
    temp_folder_deletion(const std::string& path) : m_path(path) {}
    ~temp_folder_deletion() {
        if (!config.keep_temporaries && !m_path.empty())
            std::filesystem::remove_all(m_path.c_str());
    }

private:
    std::string m_path;
};

enum class spirv_validation_level
{
    skip,
    warn,
    error,
};

bool validate_binary(spir_binary const& binary,
                     spirv_validation_options const& val_options) {
    spirv_validation_level level = spirv_validation_level::error;
    if (config.spirv_validation.set) {
        if (config.spirv_validation == 0) {
            level = spirv_validation_level::skip;
        } else if (config.spirv_validation == 1) {
            level = spirv_validation_level::warn;
        }
    }

    if (level == spirv_validation_level::skip) {
        cvk_info("Skipping validation of SPIR-V binary.");
        return true;
    }

    if (binary.validate(val_options)) {
        cvk_info("SPIR-V binary is valid.");
        return true;
    }

    if (level == spirv_validation_level::warn) {
        cvk_warn("SPIR-V binary is invalid.");
        return true;
    }

    cvk_error("SPIR-V binary is invalid.");
    return false;
}

const uint32_t clvk_binary_magic =
    0x6B766C63; // "clvk" in ASCII in little-endian
const uint32_t clvk_binary_version = 1;
struct clvk_binary_header {
    uint32_t magic;
    uint32_t version;
    uint32_t binary_type;
};
#define COPY_WORD(dst, src)                                                    \
    do {                                                                       \
        ((unsigned char*)dst)[0] = ((unsigned char*)(src))[0];                 \
        ((unsigned char*)dst)[1] = ((unsigned char*)(src))[1];                 \
        ((unsigned char*)dst)[2] = ((unsigned char*)(src))[2];                 \
        ((unsigned char*)dst)[3] = ((unsigned char*)(src))[3];                 \
    } while (0)

} // namespace

bool cvk_program::read_llvm_bitcode(const unsigned char* src, size_t size) {
    if (size >= 4 &&
        (src[0] == 'B' && src[1] == 'C' && src[2] == 0xc0 && src[3] == 0xde)) {
        m_ir.resize(size);
        memcpy(m_ir.data(), src, size);
        m_dev_status[m_context->device()] = CL_BUILD_SUCCESS;
        return true;
    }
    return false;
}

void cvk_program::write_binary_header(unsigned char* dst) const {
    struct clvk_binary_header* header = (struct clvk_binary_header*)dst;
    COPY_WORD(&header->magic, &clvk_binary_magic);
    COPY_WORD(&header->version, &clvk_binary_version);
    COPY_WORD(&header->binary_type, &m_binary_type);
}

cl_program_binary_type cvk_program::read_binary_header(const unsigned char* src,
                                                       size_t size) {
    struct clvk_binary_header* header = (struct clvk_binary_header*)src;
    if (size < sizeof(header)) {
        return CL_PROGRAM_BINARY_TYPE_NONE;
    }
    uint32_t magic, version, binary_type;
    COPY_WORD(&magic, &header->magic);
    COPY_WORD(&version, &header->version);
    COPY_WORD(&binary_type, &header->binary_type);
    if (magic != clvk_binary_magic) {
        cvk_info_fn("magic not found");
        return CL_PROGRAM_BINARY_TYPE_NONE;
    }
    if (version != clvk_binary_version) {
        cvk_warn_fn("wrong version");
        return CL_PROGRAM_BINARY_TYPE_NONE;
    }
    return binary_type;
}

bool cvk_program::read(const unsigned char* src, size_t size) {
    bool success = false;
    auto binary_type = read_binary_header(src, size);
    // if the binary does not have a clvk binary header, let's try to read
    // it first as a llvm ir buffer, then as a vulkan spirv buffer.
    if (binary_type == CL_PROGRAM_BINARY_TYPE_NONE) {
        cvk_info_fn("no clvk binary header found, looking for llvm bitcode");
        if (read_llvm_bitcode(src, size)) {
            m_binary_type = CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT;
            cvk_info_fn("llvm bitcode compiled object found");
            return true;
        }
        cvk_info_fn("llvm bitcode not found, looking for Vulkan SPIR-V binary");
        if (m_binary.read(src, size)) {
            m_binary_type = CL_PROGRAM_BINARY_TYPE_EXECUTABLE;
            cvk_info_fn("Vulkan SPIR-V binary executable found");
            return true;
        }
        cvk_error_fn("unable to read binary");
        return success;
    }

    auto header_size = sizeof(struct clvk_binary_header);
    src += header_size;
    size -= header_size;

    switch (binary_type) {
    case CL_PROGRAM_BINARY_TYPE_LIBRARY:
    case CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT:
        success = read_llvm_bitcode(src, size);
        break;
    case CL_PROGRAM_BINARY_TYPE_EXECUTABLE:
        success = m_binary.read(src, size);
        break;
    }
    if (success) {
        m_binary_type = binary_type;
    }
    return success;
}

bool cvk_program::write(unsigned char* dst) const {
    write_binary_header(dst);
    dst += sizeof(struct clvk_binary_header);

    switch (m_binary_type) {
    case CL_PROGRAM_BINARY_TYPE_LIBRARY:
    case CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT:
        memcpy(dst, m_ir.data(), m_ir.size());
        return true;
    case CL_PROGRAM_BINARY_TYPE_EXECUTABLE:
        return m_binary.write(dst);
    }
    return false;
}

size_t cvk_program::binary_size() const {
    auto header_size = sizeof(struct clvk_binary_header);
    switch (m_binary_type) {
    case CL_PROGRAM_BINARY_TYPE_LIBRARY:
    case CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT:
        return header_size + m_ir.size();
    case CL_PROGRAM_BINARY_TYPE_EXECUTABLE:
        return header_size + m_binary.size();
    }
    return 0;
}

std::string cvk_program::prepare_build_options(const cvk_device* device) const {
    // Strip off a few options we can't handle
    std::string options;
    if (m_build_options.size() > 0) {
        options += " ";
        options += m_build_options;
    }
    std::vector<std::pair<std::string, std::string>> option_substitutions = {
        // FIXME The 1.2 conformance tests shouldn't pass this option.
        //       It doesn't exist after OpenCL 1.0.
        {"-cl-strict-aliasing", ""},
        // clspv require entrypoint inlining for OpenCL 2.0 and OpenCL 3.0 (for
        // generic addrspace for example).
        {"-cl-std=CL2.0", "-cl-std=CL2.0 -inline-entry-points"},
        {"-cl-std=CL3.0", "-cl-std=CL3.0 -inline-entry-points"},
        {"-create-library", ""},
    };

    for (auto& subst : option_substitutions) {
        size_t loc = options.find(subst.first);
        if (loc != std::string::npos) {
            options.replace(loc, subst.first.length(), subst.second);
        }
    }

    // Check for some options we need and add them if not present.
    std::string necessary_options[] = {
        "-cl-single-precision-constant",
        "-cl-kernel-arg-info",
    };
    for (std::string option : necessary_options) {
        if (options.find(option) == std::string::npos) {
            options += " " + option + " ";
        }
    }

    // The device sets up some compiler options based on its capabilities.
    options += " " + device->get_device_specific_compile_options() + " ";

    // Features
    if (options.find("-cl-std=CL3.0") != std::string::npos) {
        auto features = device->opencl_c_features();
        if (!features.empty()) {
            options += "-enable-feature-macros=";
            for (auto& feature : features) {
                options += feature.name;
                options += ',';
            }
            options.back() = ' '; // replace the final comma
        }
    }

    options += " -enable-printf ";
    options +=
        " -printf-buffer-size=" + std::to_string(config.printf_buffer_size) +
        " ";

#if COMPILER_AVAILABLE
    options += " " + config.clspv_options() + " ";
#endif
    // split options into a vector
    std::istringstream iss(options);
    std::vector<std::string> vector_options;
    std::string token;
    while (std::getline(iss, token, ' ')) {
        vector_options.push_back(token);
    }

    // loop through the options and quote the ones that need it
    std::string quoted_options;
    for (size_t i = 0; i < vector_options.size(); i++) {
        if (vector_options[i].empty()) {
            continue;
        }
        if (vector_options[i].find("-") == 0) {
            quoted_options += vector_options[i];
        } else {
            quoted_options += "\"" + vector_options[i] + "\"";
        }
        quoted_options += " ";
    }

    return quoted_options;
}

cl_int cvk_program::parse_user_spec_constants() {
#if COMPILER_AVAILABLE && ENABLE_SPIRV_IL
#ifndef CLSPV_ONLINE_COMPILER
    // We'll need to go through the whole temp folder rigamarole to query the
    // spec constant info with the command line tool.
    std::filesystem::path tmp_prefix(config.compiler_temp_dir());
    std::filesystem::path tmp_suffix("clvk-XXXXXX");
    std::string tmp_template = (tmp_prefix / tmp_suffix).string();
    const char* tmp = cvk_mkdtemp(tmp_template);
    if (tmp == nullptr) {
        cvk_error_fn("Could not create temporary folder \"%s\"",
                     tmp_template.c_str());
        return CL_INVALID_VALUE;
    }
    std::string tmp_folder = tmp;
    cvk_info("Created temporary folder \"%s\"", tmp_folder.c_str());

    std::string llvmspirv_input_file = tmp_folder + "/source.spv";

    if (!save_il_to_file(llvmspirv_input_file, m_il)) {
        cvk_error_fn("Couldn't save IL to file!");
        return CL_INVALID_VALUE;
    }

    std::string cmd_spv{config.llvmspirv_bin()};
    cmd_spv += " --spec-const-info ";
    cmd_spv += llvmspirv_input_file;

    std::string output = "";
    cvk_exec(cmd_spv, &output);
    auto output_stream = std::istringstream(output);

    for (std::string line; std::getline(output_stream, line);) {
        if (line.find("Spec const id") == std::string::npos) {
            continue;
        }

        auto delim = line.find(",");
        auto id_string = line.substr(0, delim);
        line.erase(0, delim + 1);
        id_string = id_string.substr(id_string.find("=") + 1);
        auto id = static_cast<uint32_t>(atoi(id_string.c_str()));

        delim = line.find(",");
        auto size_string = line.substr(0, delim);
        line.erase(0, delim + 1);
        size_string = size_string.substr(size_string.find("=") + 1);
        auto size = static_cast<uint32_t>(atoi(size_string.c_str()));

        // We need to consume the "=" and the " " here since we aren't
        // converting the value from a string.
        auto type_string = line.substr(line.find("=") + 2);
        m_user_spec_constants.emplace(
            id, user_spec_constant_data{type_string, size});
    }
    std::filesystem::remove_all(tmp_folder.c_str());
    return CL_SUCCESS;
#else
    auto m_il_start = reinterpret_cast<const unsigned char*>(m_il.data());
    membuf m_il_buf(m_il_start, m_il_start + m_il.size());
    std::istream m_il_stream(&m_il_buf);
    SPIRV::TranslatorOpts translator_opts;
    // We don't need llvm-spirv to validate extensions for us.
    translator_opts.enableAllExtensions();
    static std::mutex llvmspirv_compile_mutex;
    std::lock_guard<std::mutex> llvmspirv_compile_lock(llvmspirv_compile_mutex);

    std::vector<llvm::SpecConstInfoTy> spec_const_info;

    if (!getSpecConstInfo(m_il_stream, spec_const_info)) {
        cvk_error_fn("Failed to parse spec constants");
        return CL_INVALID_VALUE;
    }

    for (const auto& spec_const : spec_const_info) {
        m_user_spec_constants.emplace(
            spec_const.ID,
            user_spec_constant_data{spec_const.Type, spec_const.Size});
    }
    return CL_SUCCESS;
#endif // CLSPV_ONLINE_COMPILER
#else
#if !COMPILER_AVAILABLE
    cvk_error_fn("Could not parse user spec constants because clvk has been "
                 "built with CLVK_COMPILER_AVAILABLE=OFF");
#elif !ENABLE_SPIRV_IL
    cvk_error_fn("Could not parse user spec constants because clvk has been "
                 "built with CLVK_ENABLE_SPIRV_IL=OFF");
#endif
    return CL_INVALID_OPERATION;
#endif // COMPILER_AVAILABLE
}

#if COMPILER_AVAILABLE
#ifndef CLSPV_ONLINE_COMPILER
cl_build_status cvk_program::do_build_inner_offline(bool build_to_ir,
                                                    bool build_from_il,
                                                    std::string& build_options,
                                                    std::string& tmp_folder) {
    TRACE_FUNCTION("build_to_ir", build_to_ir, "build_from_il", build_from_il,
                   "build_options", TRACE_STRING(build_options.c_str()));
    // Compose clspv command-line
    std::string cmd{config.clspv_path};
    cmd += " ";

    std::string clspv_input_file{tmp_folder + "/source"};
    // Save input program to a file
    if (build_from_il) {
#ifndef ENABLE_SPIRV_IL
        cvk_error_fn("Could not build from il because clvk has been built with "
                     "CLVK_ENABLE_SPIRV_IL=OFF");
        return CL_BUILD_ERROR;
#else  // ENABLE_SPIRV_IL
        std::string llvmspirv_input_file{tmp_folder + "/source.spv"};
        clspv_input_file += ".bc";
        if (!save_il_to_file(llvmspirv_input_file, m_il)) {
            cvk_error_fn("Couldn't save IL to file!");
            return CL_BUILD_ERROR;
        }
        // Compose llvm-spirv command-line
        std::string cmd_spv{config.llvmspirv_bin};

        if (!m_user_spec_constants.empty()) {
            std::string spec_constant_flag = " --spec-const=";
            for (const auto& spec_const : m_user_spec_constants) {
                if (!spec_const.second.set) {
                    continue;
                }
                const auto& type_string = spec_const.second.type;
                spec_constant_flag +=
                    std::to_string(spec_const.first) + ":" + type_string + ":";

                if (type_string.find("i32") != std::string::npos) {
                    spec_constant_flag +=
                        std::to_string(spec_const.second.data.i32);
                } else if (type_string.find("i16") != std::string::npos) {
                    spec_constant_flag +=
                        std::to_string(spec_const.second.data.i16);
                } else if (type_string.find("i8") != std::string::npos ||
                           type_string.find("i1") != std::string::npos) {
                    spec_constant_flag +=
                        std::to_string(spec_const.second.data.i8);
                } else if (type_string.find("i64") != std::string::npos) {
                    spec_constant_flag +=
                        std::to_string(spec_const.second.data.i64);
                } else if (type_string.find("f16") != std::string::npos) {
                    // At most we should need seven (0xFFFF\0)
                    char buf[7];
                    std::snprintf(buf, 7, "0x%04X", spec_const.second.data.i16);
                    spec_constant_flag += std::string(buf);
                } else if (type_string.find("f32") != std::string::npos) {
                    float spec_value;
                    std::memcpy(&spec_value, &spec_const.second.data.i32,
                                spec_const.second.size);
                    spec_constant_flag += std::to_string(spec_value);
                } else if (type_string.find("f64") != std::string::npos) {
                    double spec_value;
                    std::memcpy(&spec_value, &spec_const.second.data.i64,
                                spec_const.second.size);
                    spec_constant_flag += std::to_string(spec_value);
                }
                spec_constant_flag += " ";
            }
            cmd_spv += spec_constant_flag;
        }

        cmd_spv += " -r ";
        cmd_spv += " -o ";
        cmd_spv += clspv_input_file;
        cmd_spv += " ";
        cmd_spv += llvmspirv_input_file;

        // Call the translator
        int status = cvk_exec(cmd_spv);

        if (status != 0) {
            cvk_error_fn("failed to translate SPIR-V to LLVM IR");
            return CL_BUILD_ERROR;
        }

        cmd += clspv_input_file;
        cmd += " ";
#endif // ENABLE_SPIRV_IL
    } else if (m_operation == build_operation::link) {
        for (auto input_program : m_input_programs) {
            if (input_program->m_binary_type !=
                    CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT &&
                input_program->m_binary_type !=
                    CL_PROGRAM_BINARY_TYPE_LIBRARY) {
                return CL_BUILD_ERROR;
            }
            std::string input_file = clspv_input_file + "_" +
                                     std::to_string((uintptr_t)input_program) +
                                     ".bc";
            if (!save_il_to_file(input_file, input_program->m_ir)) {
                cvk_error_fn("Couldn't save source to file!");
                return CL_BUILD_ERROR;
            }
            cmd += input_file;
            cmd += " ";
        }
    } else {
        if (m_source.empty() && !m_ir.empty()) {
            clspv_input_file += ".bc";
            if (!save_il_to_file(clspv_input_file, m_ir)) {
                cvk_error_fn("Couldn't save source to file!");
                return CL_BUILD_ERROR;
            }
            cmd += clspv_input_file;
            cmd += " ";
        } else {
            clspv_input_file += ".cl";
            if (!save_string_to_file(clspv_input_file, m_source)) {
                cvk_error_fn("Couldn't save source to file!");
                return CL_BUILD_ERROR;
            }
            cmd += clspv_input_file;
            cmd += " ";
        }
    }

    std::string clspv_output_file{tmp_folder + "/compiled"};
    if (build_to_ir) {
        clspv_output_file += ".bc";
    } else {
        clspv_output_file += ".spv";
    }

    cmd += build_options;
    cmd += " -o ";
    cmd += clspv_output_file;

    // Call clspv
    int status = cvk_exec(cmd, &m_build_log);
    if (status != 0) {
        cvk_error_fn("failed to compile the program");
        cvk_debug_fn("%s", m_build_log.c_str());
        return CL_BUILD_ERROR;
    }

    // Load output from clspv
    if (build_to_ir) {
        std::ifstream stream(clspv_output_file,
                             std::ios::in | std::ios::binary);
        m_ir.assign((std::istreambuf_iterator<char>(stream)),
                    std::istreambuf_iterator<char>());
        if (!stream.good()) {
            return CL_BUILD_ERROR;
        }
    } else {
        const char* filename = clspv_output_file.c_str();
        if (!m_binary.load(filename)) {
            cvk_error("Could not load SPIR-V binary from \"%s\"", filename);
            return CL_BUILD_ERROR;
        }
    }

    cvk_info("Loaded %s binary from \"%s\", size = %zu %s",
             build_to_ir ? "IR" : "SPIR-V", clspv_output_file.c_str(),
             build_to_ir ? m_ir.size() : m_binary.code().size(),
             build_to_ir ? "bytes" : "words");

    return CL_BUILD_SUCCESS;
}

#else // #ifndef CLSPV_ONLINE_COMPILER

cl_build_status cvk_program::do_build_inner_online(bool build_to_ir,
                                                   bool build_from_il,
                                                   std::string& build_options) {
    TRACE_FUNCTION("build_to_ir", build_to_ir, "build_from_il", build_from_il,
                   "build_options", TRACE_STRING(build_options.c_str()));
    cvk_info_fn("build_from_il %u - build_to_ir %u", build_from_il,
                build_to_ir);
    if (build_from_il) {
#ifndef ENABLE_SPIRV_IL
        cvk_error_fn("Could not build from il because clvk has been built with "
                     "CLVK_ENABLE_SPIRV_IL=OFF");
        return CL_BUILD_ERROR;
#else  // ENABLE_SPIRV_IL
        llvm::LLVMContext llvm_context;
        llvm::Module* llvm_module;
        std::string err;
        auto m_il_start = reinterpret_cast<const unsigned char*>(m_il.data());
        membuf m_il_buf(m_il_start, m_il_start + m_il.size());
        std::istream m_il_stream(&m_il_buf);
        SPIRV::TranslatorOpts translator_opts;
        // We don't need llvm-spirv to validate extensions for us.
        translator_opts.enableAllExtensions();

        for (const auto& spec_const : m_user_spec_constants) {
            if (!spec_const.second.set) {
                continue;
            }
            uint64_t spec_const_value = 0;
            if (spec_const.second.type == "i1" ||
                spec_const.second.type == "i8") {
                spec_const_value = spec_const.second.data.i8;
            } else if (spec_const.second.type == "i16" ||
                       spec_const.second.type == "f16") {
                spec_const_value = spec_const.second.data.i16;
            } else if (spec_const.second.type == "i32" ||
                       spec_const.second.type == "f32") {
                spec_const_value = spec_const.second.data.i32;
            } else if (spec_const.second.type == "i64" ||
                       spec_const.second.type == "f64") {
                spec_const_value = spec_const.second.data.i64;
            }
            translator_opts.setSpecConst(spec_const.first, spec_const_value);
        }

        // llvm-spirv is based on LLVM. LLVM options parsing is done using
        // global variable that are not thread safe. Thus, we need to lock call
        // to llvm-spirv in order to ensure a thread safe execution.
        {
            static std::mutex llvmspirv_compile_mutex;
            std::lock_guard<std::mutex> llvmspirv_compile_lock(
                llvmspirv_compile_mutex);
            std::vector<const char*> llvmArgv{
                "llvm-spirv(online)",
            };
            llvm::cl::ResetAllOptionOccurrences();
            llvm::cl::ParseCommandLineOptions(llvmArgv.size(), llvmArgv.data());
            if (!llvm::readSpirv(llvm_context, translator_opts, m_il_stream,
                                 llvm_module, err)) {
                cvk_error_fn("Fails to load SPIR-V as LLVM Module: %s",
                             err.c_str());
                return CL_BUILD_ERROR;
            }
        }

        m_source.clear();
        llvm::raw_string_ostream spirv_stream(m_source);
        llvm::WriteBitcodeToFile(*llvm_module, spirv_stream);
#endif // ENABLE_SPIRV_IL
    }
    cvk_info("About to compile \"%s\"", build_options.c_str());
    int status;
    // clspv is based on LLVM. LLVM options parsing is done using global
    // variable that are not thread safe. Thus, we need to lock call to clspv in
    // order to ensure a thread safe execution.
    {
        static std::mutex clspv_compile_mutex;
        std::lock_guard<std::mutex> clspv_compile_lock(clspv_compile_mutex);
        std::vector<std::string> programs;
        if (m_operation == build_operation::link) {
            for (auto input_program : m_input_programs) {
                if (input_program->m_binary_type !=
                        CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT &&
                    input_program->m_binary_type !=
                        CL_PROGRAM_BINARY_TYPE_LIBRARY) {
                    return CL_BUILD_ERROR;
                }
                programs.emplace_back(std::string{input_program->m_ir.begin(),
                                                  input_program->m_ir.end()});
            }
        } else {
            if (m_source.empty() && !m_ir.empty()) {
                programs.emplace_back(std::string{m_ir.begin(), m_ir.end()});
            } else {
                programs.emplace_back(m_source);
            }
        }
        m_build_log.clear();
        if (build_to_ir) {
            std::vector<uint32_t> ir;
            status = clspv::CompileFromSourcesString(programs, build_options,
                                                     &ir, &m_build_log);
            m_ir.clear();
            auto size = ir.size() * sizeof(uint32_t);
            m_ir.resize(size);
            memcpy(m_ir.data(), ir.data(), size);
        } else {
            status = clspv::CompileFromSourcesString(
                programs, build_options, m_binary.raw_binary(), &m_build_log);
        }
    }
    if (status != 0) {
        cvk_error_fn("failed to compile the program");
        cvk_debug_fn("%s", m_build_log.c_str());
        return CL_BUILD_ERROR;
    }
    return CL_BUILD_SUCCESS;
}

#endif // #ifndef CLSPV_ONLINE_COMPILER
#endif // #if COMPILER_AVAILABLE

cl_build_status cvk_program::do_build_inner(const cvk_device* device) {
#if !COMPILER_AVAILABLE
    UNUSED(device);
#else

    bool build_from_il =
        m_il.size() > 0 && m_operation != build_operation::link;
    bool use_tmp_folder = true;
#ifdef CLSPV_ONLINE_COMPILER
    use_tmp_folder =
        m_operation == build_operation::compile && m_num_input_programs > 0;
#endif

    std::string tmp_folder;
    if (use_tmp_folder) {
        // Create temporary folder
        std::filesystem::path tmp_prefix(config.compiler_temp_dir());
        std::filesystem::path tmp_suffix("clvk-XXXXXX");
        std::string tmp_template = (tmp_prefix / tmp_suffix).string();
        const char* tmp = cvk_mkdtemp(tmp_template);
        if (tmp == nullptr) {
            cvk_error_fn("Could not create temporary folder \"%s\"",
                         tmp_template.c_str());
            return CL_BUILD_ERROR;
        }
        tmp_folder = tmp;
        cvk_info("Created temporary folder \"%s\"", tmp_folder.c_str());
    }
    temp_folder_deletion temp(tmp_folder);

    // Prepare build options
    bool create_library =
        m_build_options.find("create-library") != std::string::npos;
    auto build_options = prepare_build_options(device);

    // Add options to specify input/output types
    if (build_from_il ||
        m_binary_type == CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT ||
        m_binary_type == CL_PROGRAM_BINARY_TYPE_LIBRARY ||
        m_operation == build_operation::link) {
        build_options += " -x ir ";
    }
    bool build_to_ir =
        m_operation == build_operation::compile || create_library;
    if (build_to_ir) {
        build_options += " --output-format=bc ";
    }

    // Save headers
    if (use_tmp_folder && m_operation == build_operation::compile) {
        build_options += "-I" + tmp_folder;
        for (cl_uint i = 0; i < m_num_input_programs; i++) {
            std::string fname{tmp_folder + "/" + m_header_include_names[i]};
            if (!save_string_to_file(fname, m_input_programs[i]->source())) {
                cvk_error_fn("Couldn't save header to file!");
                return CL_BUILD_ERROR;
            }
        }
    }

    cl_build_status build_status;
#ifdef CLSPV_ONLINE_COMPILER
    build_status =
        do_build_inner_online(build_to_ir, build_from_il, build_options);
#else
    build_status = do_build_inner_offline(build_to_ir, build_from_il,
                                          build_options, tmp_folder);
#endif // CLSPV_ONLINE_COMPILER
    if (build_status != CL_BUILD_SUCCESS) {
        return build_status;
    }

    // Select operation
    if (m_operation == build_operation::compile) {
        m_binary_type = CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT;
    } else if (m_operation == build_operation::link && create_library) {
        m_binary_type = CL_PROGRAM_BINARY_TYPE_LIBRARY;
    } else {
        m_binary_type = CL_PROGRAM_BINARY_TYPE_EXECUTABLE;
    }

#endif // #if !COMPILER_AVAILABLE
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
        cvk_info_fn("Program requires SPIR-V capability %d (%s).", c,
                    spirv_capability_to_string(c));
        if (!device->supports_capability(c)) {
            // TODO: propagate this message to the build log
            cvk_error_fn("Device does not support SPIR-V capability %d (%s).",
                         c, spirv_capability_to_string(c));
            return false;
        }
    }
    return true;
}

void cvk_program::do_build() {
    cvk_set_current_thread_name_if_supported("clvk-build");

    // Destroy entry points from previous build
    m_entry_points.clear();

    auto device = m_context->device();

    if (m_operation != build_operation::build_binary) {
        cl_build_status status = do_build_inner(device);

        if ((m_binary_type != CL_PROGRAM_BINARY_TYPE_EXECUTABLE) ||
            (status != CL_BUILD_SUCCESS)) {
            complete_operation(device, status);
            return;
        }
    }

    // Load descriptor map
    if (!m_binary.load_descriptor_map()) {
        cvk_error("Could not load descriptor map for SPIR-V binary.");
        complete_operation(device, CL_BUILD_ERROR);
        return;
    }

    if (!create_module_constant_data_buffer()) {
        complete_operation(device, CL_BUILD_ERROR);
        return;
    }

    prepare_push_constant_range();

    bool cache_hit =
        device->get_pipeline_cache(m_binary.code(), m_pipeline_cache);
    if (m_pipeline_cache == VK_NULL_HANDLE) {
        complete_operation(device, CL_BUILD_ERROR);
        return;
    }

    if (!cache_hit) {
        // Validate
        // TODO validate with different rules depending on the binary type
        if (m_binary_type == CL_PROGRAM_BINARY_TYPE_EXECUTABLE) {
            spirv_validation_options validation_options{};
            validation_options.uniform_buffer_std_layout =
                m_context->device()->supports_ubo_stdlayout();
            if (!validate_binary(m_binary, validation_options)) {
                complete_operation(device, CL_BUILD_ERROR);
                return;
            }
        }
    }

    // Check capabilities against the device.
    if ((m_binary_type == CL_PROGRAM_BINARY_TYPE_EXECUTABLE) &&
        !config.skip_spirv_capability_check && !check_capabilities(device)) {
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
    const bool should_strip_reflection =
        !device->is_vulkan_extension_enabled(
            VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME)
#ifdef USING_SWIFTSHADER
        || true
#endif
        ;
    if (should_strip_reflection) {
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

cvk_entry_point::cvk_entry_point(cvk_device* dev, cvk_program* program,
                                 const std::string& name)
    : m_device(dev), m_context(program->context()), m_program(program),
      m_name(name), m_pod_descriptor_type(VK_DESCRIPTOR_TYPE_MAX_ENUM),
      m_pod_buffer_size(0u), m_has_pod_arguments(false),
      m_has_pod_buffer_arguments(false), m_sampler_metadata(nullptr),
      m_image_metadata(nullptr), m_descriptor_pool(VK_NULL_HANDLE),
      m_pipeline_layout(VK_NULL_HANDLE), m_nb_descriptor_set_allocated(0),
      m_first_allocation_failure(true) {
    TRACE_CNT_VAR_INIT(descriptor_set_allocated_counter,
                       "clvk-entry_point_" + std::to_string((uintptr_t)this));
    TRACE_CNT(descriptor_set_allocated_counter, 0);
}

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
        new cvk_entry_point(m_context->device(), this, name);
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
        res = vkCreateDescriptorSetLayout(m_device->vulkan_device(),
                                          &createInfo, 0, &setLayout);
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
        case kernel_argument_kind::sampled_image:
            dt = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            break;
        case kernel_argument_kind::storage_image:
            dt = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            break;
        case kernel_argument_kind::storage_texel_buffer:
            dt = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
            break;
        case kernel_argument_kind::uniform_texel_buffer:
            dt = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
            break;
        case kernel_argument_kind::sampler:
            dt = VK_DESCRIPTOR_TYPE_SAMPLER;
            break;
        case kernel_argument_kind::local:
            continue;
        case kernel_argument_kind::pod:
        case kernel_argument_kind::pod_ubo:
        case kernel_argument_kind::pointer_ubo:
            if (!pod_found) {
                if (arg.kind == kernel_argument_kind::pod) {
                    dt = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                } else if (arg.kind == kernel_argument_kind::pod_ubo ||
                           arg.kind == kernel_argument_kind::pointer_ubo) {
                    dt = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                }

                m_pod_descriptor_type = dt;

                pod_found = true;
            } else {
                continue;
            }
            break;
        case kernel_argument_kind::pod_pushconstant:
        case kernel_argument_kind::pointer_pushconstant:
        case kernel_argument_kind::unused:
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

bool cvk_entry_point::
    build_descriptor_sets_layout_bindings_for_program_scope_buffers(
        binding_stat_map& smap) {

    std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
    if (m_program->module_constant_data_buffer() != nullptr) {
        auto info = m_program->module_constant_data_buffer_info();
        // If the program scope buffer isn't passed as a storage buffer (i.e.
        // it is passed a pointer push constant), there is nothing to bind here
        if (info->type != module_buffer_type::storage_buffer) {
            return true;
        }
        VkDescriptorSetLayoutBinding binding = {
            info->binding,                     // binding
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // descriptorType
            1,                                 // decriptorCount
            VK_SHADER_STAGE_COMPUTE_BIT,       // stageFlags
            nullptr                            // pImmutableSamplers
        };
        layoutBindings.push_back(binding);
        smap[binding.descriptorType]++;
    }

    if (!build_descriptor_set_layout(layoutBindings)) {
        return false;
    }

    return true;
}

bool cvk_entry_point::build_descriptor_sets_layout_bindings_for_printf_buffer(
    binding_stat_map& smap) {
    std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
    if (m_program->printf_buffer_info().size > 0 && (uses_printf())) {
        auto info = m_program->printf_buffer_info();
        if (info.type != module_buffer_type::storage_buffer) {
            return true;
        }
        VkDescriptorSetLayoutBinding binding = {
            info.binding,                      // binding
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // descriptorType
            1,                                 // decriptorCount
            VK_SHADER_STAGE_COMPUTE_BIT,       // stageFlags
            nullptr                            // pImmutableSamplers
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

    // Get the image metadata for the this entry point
    if (auto* md = m_program->image_metadata(m_name)) {
        m_image_metadata = md;
    }

    // Get the sampler metadata for this entry point
    if (auto* md = m_program->sampler_metadata(m_name)) {
        m_sampler_metadata = md;
    }

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
    if (!build_descriptor_sets_layout_bindings_for_program_scope_buffers(
            bindingTypes)) {
        return CL_INVALID_VALUE;
    }
    if (!build_descriptor_sets_layout_bindings_for_printf_buffer(
            bindingTypes)) {
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

    // Take the size of image & sampler metadata into account for the pod buffer
    // size
    {
        uint32_t max_offset = 0;
        if (m_image_metadata) {
            // Find how big the POD buffer should be
            for (const auto& md : *m_image_metadata) {
                auto order_offset = md.second.order_offset;
                auto data_type_offset = md.second.data_type_offset;
                if (md.second.has_valid_order()) {
                    max_offset = std::max(order_offset, max_offset);
                    push_constant_range.offset =
                        std::min(order_offset, push_constant_range.offset);
                    if (order_offset + sizeof(uint32_t) >
                        push_constant_range.offset + push_constant_range.size) {
                        push_constant_range.size = order_offset +
                                                   sizeof(uint32_t) -
                                                   push_constant_range.offset;
                    }
                }
                if (md.second.has_valid_data_type()) {
                    max_offset = std::max(data_type_offset, max_offset);
                    push_constant_range.offset =
                        std::min(data_type_offset, push_constant_range.offset);
                    if (data_type_offset + sizeof(uint32_t) >
                        push_constant_range.offset + push_constant_range.size) {
                        push_constant_range.size = data_type_offset +
                                                   sizeof(uint32_t) -
                                                   push_constant_range.offset;
                    }
                }
            }
        }
        if (m_sampler_metadata) {
            for (const auto& md : *m_sampler_metadata) {
                auto offset = md.second;
                max_offset = std::max(offset, max_offset);
                push_constant_range.offset =
                    std::min(offset, push_constant_range.offset);
                if (offset + sizeof(uint32_t) >
                    push_constant_range.offset + push_constant_range.size) {
                    push_constant_range.size =
                        offset + sizeof(uint32_t) - push_constant_range.offset;
                }
            }
        }
        if (max_offset + sizeof(uint32_t) > m_pod_buffer_size) {
            m_pod_buffer_size = round_up(max_offset + sizeof(uint32_t), 4);
        }
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

    if (push_constant_range.size >
        m_context->device()->vulkan_max_push_constants_size()) {
        cvk_error("Not enough space for push constants");
        return CL_INVALID_VALUE;
    }

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

    res = vkCreatePipelineLayout(m_device->vulkan_device(),
                                 &pipelineLayoutCreateInfo, 0,
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

        res = vkCreateDescriptorPool(m_device->vulkan_device(),
                                     &descriptorPoolCreateInfo, 0,
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

    void* pipelineShaderStageCreateInfoPNext = nullptr;
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo
        reqdSubgroupSizeCreateInfo;
    if (m_device->supports_subgroup_size_selection()) {
        auto reqdSubgroupSize = m_program->required_sub_group_size(m_name);
        if (reqdSubgroupSize > m_device->max_sub_group_size() ||
            reqdSubgroupSize < m_device->min_sub_group_size()) {
            if (reqdSubgroupSize != 0) {
                cvk_error_fn("required subgroup size '%u' for '%s' is out of "
                             "the supported range [%u, %u]",
                             reqdSubgroupSize, m_name.c_str(),
                             m_device->min_sub_group_size(),
                             m_device->max_sub_group_size());
                return VK_NULL_HANDLE;
            }
            reqdSubgroupSize = m_device->sub_group_size();
        }
        reqdSubgroupSizeCreateInfo.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
        reqdSubgroupSizeCreateInfo.pNext = nullptr;
        reqdSubgroupSizeCreateInfo.requiredSubgroupSize = reqdSubgroupSize;
        pipelineShaderStageCreateInfoPNext = &reqdSubgroupSizeCreateInfo;
    }

    const VkComputePipelineCreateInfo createInfo = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, // sType
        nullptr,                                        // pNext
        0,                                              // flags
        {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, // sType
            pipelineShaderStageCreateInfoPNext,                  // pNext
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
    VkResult res = vkCreateComputePipelines(m_device->vulkan_device(),
                                            m_program->pipeline_cache(), 1,
                                            &createInfo, nullptr, &pipeline);

    if (res != VK_SUCCESS) {
        cvk_error_fn("Could not create compute pipeline for kernel %s: %s",
                     vulkan_error_string(res), m_name.c_str());
        return VK_NULL_HANDLE;
    }

    // Add to pipeline cache
    m_pipelines[spec_constants] = pipeline;

    cvk_info("created pipeline %p for kernel %s", pipeline, m_name.c_str());

    return pipeline;
}

bool cvk_entry_point::allocate_descriptor_sets(VkDescriptorSet* ds) {
    TRACE_FUNCTION();

    if (m_descriptor_set_layouts.size() == 0) {
        return true;
    }

    std::lock_guard<std::mutex> lock(m_descriptor_pool_lock);

#if CLVK_UNIT_TESTING_ENABLED
    if (config.force_descriptor_set_allocation_failure() &&
        m_nb_descriptor_set_allocated + m_descriptor_set_layouts.size() >
            config.max_entry_points_instances()) {
        return false;
    }
#endif

    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
        m_descriptor_pool,
        static_cast<uint32_t>(
            m_descriptor_set_layouts.size()), // descriptorSetCount
        m_descriptor_set_layouts.data()};

    VkResult res = vkAllocateDescriptorSets(m_device->vulkan_device(),
                                            &descriptorSetAllocateInfo, ds);

    if (res != VK_SUCCESS) {
        if (config.enqueue_command_retry_sleep_us == UINT32_MAX) {
            cvk_error_fn("could not allocate descriptor sets: %s",
                         vulkan_error_string(res));
        } else if (m_first_allocation_failure) {
            cvk_warn_fn(
                "could not allocate descriptor sets: %s, retry in %u us",
                vulkan_error_string(res),
                config.enqueue_command_retry_sleep_us());
        } else {
            cvk_info_fn("could not allocate descriptor sets: %s",
                        vulkan_error_string(res));
        }
        m_first_allocation_failure = false;
        return false;
    }

    m_nb_descriptor_set_allocated += m_descriptor_set_layouts.size();
    TRACE_CNT(descriptor_set_allocated_counter, m_nb_descriptor_set_allocated);

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

bool cvk_entry_point::uses_printf() const {
    return m_program->kernel_flags(m_name) &
           NonSemanticClspvReflectionMayUsePrintf;
}
