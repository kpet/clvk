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

#ifdef CLSPV_ONLINE_COMPILER
#include "clspv/Compiler.h"
#endif
#include "spirv-tools/linker.hpp"
#include "spirv-tools/optimizer.hpp"
#include "spirv/1.0/spirv.hpp"

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

/*
 * BINARY FILE FORMAT
 * +---------+-----------------------+
 * | U32     | Version               |
 * | U32     | SPIR Size             |
 * | N * U32 | SPIR                  |
 * | U32     | Descriptor map size   |
 * | N * U32 | Descriptor map text   |
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

    // Load descriptor map
    uint32_t size_dmap;
    istream.read(reinterpret_cast<char*>(&size_dmap), sizeof(uint32_t));

    return load_descriptor_map(istream);
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

    // Write DMAP
    uint32_t dmap_size = m_dmaps_text.size();
    ostream.write(reinterpret_cast<const char*>(&dmap_size), sizeof(dmap_size));
    ostream.write(reinterpret_cast<const char*>(m_dmaps_text.c_str()),
                  dmap_size);

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
    return sizeof(MAGIC) + sizeof(uint32_t) + (m_code.size() * SPIR_WORD_SIZE) +
           sizeof(uint32_t) + m_dmaps_text.size();
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

bool parse_local_arg(kernel_argument& arg,
                     const std::vector<std::string>& tokens, int toknum) {
    if (tokens[toknum++] != "argKind") {
        return false;
    }

    if (tokens[toknum++] != "local") {
        return false;
    }

    arg.kind = kernel_argument_kind::local;

    if (tokens[toknum++] != "arrayElemSize") {
        return false;
    }

    arg.local_elem_size = atoi(tokens[toknum++].c_str());

    if (tokens[toknum++] != "arrayNumElemSpecId") {
        return false;
    }

    arg.local_spec_id = atoi(tokens[toknum++].c_str());

    return true;
}

bool parse_arg(kernel_argument& arg, const std::vector<std::string>& tokens,
               int toknum) {
    if (tokens[toknum++] != "descriptorSet") {
        return false;
    }

    arg.descriptorSet = atoi(tokens[toknum++].c_str());

    if (arg.descriptorSet >= spir_binary::MAX_DESCRIPTOR_SETS) {
        return false;
    }

    if (tokens[toknum++] != "binding") {
        return false;
    }

    arg.binding = atoi(tokens[toknum++].c_str());

    if (tokens[toknum++] != "offset") {
        return false;
    }

    arg.offset = atoi(tokens[toknum++].c_str());

    if (tokens[toknum++] != "argKind") {
        return false;
    }

    std::string akind{tokens[toknum++]};

    if (akind == "buffer") {
        arg.kind = kernel_argument_kind::buffer;
    } else if (akind == "buffer_ubo") {
        arg.kind = kernel_argument_kind::buffer_ubo;
    } else if (akind == "pod") {
        arg.kind = kernel_argument_kind::pod;
    } else if (akind == "pod_ubo") {
        arg.kind = kernel_argument_kind::pod_ubo;
    } else if (akind == "ro_image") {
        arg.kind = kernel_argument_kind::ro_image;
    } else if (akind == "wo_image") {
        arg.kind = kernel_argument_kind::wo_image;
    } else if (akind == "sampler") {
        arg.kind = kernel_argument_kind::sampler;
    } else {
        return false;
    }

    if (arg.is_pod()) {
        if (tokens[toknum++] != "argSize") {
            return false;
        }

        arg.size = atoi(tokens[toknum++].c_str());
    }

    return true;
}

bool parse_kernel_pushconstant(kernel_argument& arg,
                               const std::vector<std::string>& tokens,
                               int toknum) {
    if (tokens[toknum++] != "offset") {
        return false;
    }

    arg.offset = atoi(tokens[toknum++].c_str());

    if (tokens[toknum++] != "argKind") {
        return false;
    }

    std::string akind{tokens[toknum++]};

    if (akind == "pod_pushconstant") {
        arg.kind = kernel_argument_kind::pod;
    } else {
        return false;
    }

    if (tokens[toknum++] != "argSize") {
        return false;
    }

    arg.size = atoi(tokens[toknum++].c_str());

    return true;
}

static std::vector<std::string> tokenize(const std::string& str,
                                         const char* delim) {
    size_t start = str.find_first_not_of(delim), end;
    std::vector<std::string> tokens;

    while (start != std::string::npos) {
        end = str.find(delim, start);
        tokens.push_back(str.substr(start, end - start));
        start = str.find_first_not_of(delim, end);
    }
    // for (auto& tok : tokens) {
    //    cvk_debug("TOK: %s", tok.c_str());
    //}

    return tokens;
}

bool spir_binary::parse_sampler(const std::vector<std::string>& tokens,
                                int toknum) {

    sampler_desc desc;

    int samplerVal = atoi(tokens[toknum++].c_str());
    UNUSED(samplerVal);

    if (tokens[toknum++] != "samplerExpr") {
        return false;
    }

    std::string samplerExpr = tokens[toknum++];
    auto exprTokens = tokenize(samplerExpr, "|");

    if (tokens[toknum++] != "descriptorSet") {
        return false;
    }

    desc.descriptorSet = atoi(tokens[toknum++].c_str());

    if (desc.descriptorSet >= spir_binary::MAX_DESCRIPTOR_SETS) {
        return false;
    }

    if (tokens[toknum++] != "binding") {
        return false;
    }

    desc.binding = atoi(tokens[toknum++].c_str());

    m_literal_samplers.push_back(desc);

    return true;
}

bool spir_binary::parse_kernel(const std::vector<std::string>& tokens,
                               int toknum) {
    kernel_argument arg;
    std::string kname{tokens[toknum++]};

    if (tokens[toknum++] != "arg") {
        return false;
    }

    arg.name = tokens[toknum++];

    if (tokens[toknum++] != "argOrdinal") {
        return false;
    }

    arg.pos = atoi(tokens[toknum++].c_str());

    if (tokens[toknum] == "descriptorSet") {
        if (!parse_arg(arg, tokens, toknum)) {
            return false;
        }
    } else if (tokens[toknum] == "argKind") {
        if (!parse_local_arg(arg, tokens, toknum)) {
            return false;
        }
    } else if (tokens[toknum] == "offset") {
        if (!parse_kernel_pushconstant(arg, tokens, toknum)) {
            return false;
        }
    } else {
        return false;
    }

    m_dmaps[kname].push_back(arg);

    return true;
}

bool spir_binary::parse_kernel_decl(const std::vector<std::string>& tokens,
                                    int toknum) {
    std::string kname{tokens[toknum++]};

    if (m_dmaps.count(kname) == 0) {
        m_dmaps[kname] = {};
    }

    return true;
}

bool spir_binary::parse_pushconstant(const std::vector<std::string>& tokens,
                                     int toknum) {
    pushconstant_desc pcd;

    if (tokens[toknum++] != "name") {
        return false;
    }

    auto& name = tokens[toknum++];
    pushconstant pc;

    if (name == "global_offset") {
        pc = pushconstant::global_offset;
    } else if (name == "enqueued_local_size") {
        pc = pushconstant::enqueued_local_size;
    } else if (name == "global_size") {
        pc = pushconstant::global_size;
    } else if (name == "region_offset") {
        pc = pushconstant::region_offset;
    } else if (name == "num_workgroups") {
        pc = pushconstant::num_workgroups;
    } else if (name == "region_group_offset") {
        pc = pushconstant::region_group_offset;
    } else {
        return false;
    }

    if (tokens[toknum++] != "offset") {
        return false;
    }

    pcd.offset = atoi(tokens[toknum++].c_str());

    if (tokens[toknum++] != "size") {
        return false;
    }

    pcd.size = atoi(tokens[toknum++].c_str());

    m_push_constants[pc] = pcd;

    return true;
}

bool spir_binary::parse_specconstant(const std::vector<std::string>& tokens,
                                     int toknum) {
    auto name = tokens[toknum++];
    spec_constant constant;
    if (name == "workgroup_size_x") {
        constant = spec_constant::workgroup_size_x;
    } else if (name == "workgroup_size_y") {
        constant = spec_constant::workgroup_size_y;
    } else if (name == "workgroup_size_z") {
        constant = spec_constant::workgroup_size_z;
    } else if (name == "work_dim") {
        constant = spec_constant::work_dim;
    } else {
        return false;
    }

    if (tokens[toknum++] != "spec_id") {
        return false;
    }

    uint32_t id = atoi(tokens[toknum++].c_str());
    m_spec_constants[constant] = id;

    return true;
}

bool spir_binary::load_descriptor_map(std::istream& istream) {
    m_dmaps.clear();

    std::string line;
    while (std::getline(istream, line)) {
        m_dmaps_text += line + "\n";
        cvk_debug("DMAP line: %s", line.c_str());
        std::vector<std::string> tokens = tokenize(line, ",");

        int toknum = 0;

        if (tokens[toknum] == "kernel") {
            if (!parse_kernel(tokens, toknum + 1)) {
                return false;
            }
        } else if (tokens[toknum] == "sampler") {
            if (!parse_sampler(tokens, toknum + 1)) {
                return false;
            }
        } else if (tokens[toknum] == "pushconstant") {
            if (!parse_pushconstant(tokens, toknum + 1)) {
                return false;
            }
        } else if (tokens[toknum] == "kernel_decl") {
            if (!parse_kernel_decl(tokens, toknum + 1)) {
                return false;
            }
        } else if (tokens[toknum] == "spec_constant") {
            if (!parse_specconstant(tokens, toknum + 1)) {
                return false;
            }
        } else {
            return false;
        }
    }

    cvk_debug_fn("num_kernels = %zu", num_kernels());

    return true;
}

bool spir_binary::load_descriptor_map(const char* fname) {
    std::ifstream ifile;

    ifile.open(fname, std::ios::in);

    if (!ifile.is_open()) {
        cvk_error("Failed to open %s", fname);
        return false;
    }

    return load_descriptor_map(ifile);
}

#ifdef CLSPV_ONLINE_COMPILER
bool spir_binary::load_descriptor_map(
    const std::vector<clspv::version0::DescriptorMapEntry>& entries) {
    m_dmaps.clear();
    for (const auto& entry : entries) {
        if (cvk_log_level_enabled(loglevel::debug)) {
            std::string s;
            std::ostringstream str(s);
            str << entry;
            cvk_debug("DMAP line: %s", str.str().c_str());
        }

        if (entry.kind == clspv::version0::DescriptorMapEntry::Kind::Sampler) {
            sampler_desc desc;
            desc.descriptorSet = entry.descriptor_set;
            desc.binding = entry.binding;

            desc.normalized_coords =
                (entry.sampler_data.mask &
                 clspv::version0::kSamplerNormalizedCoordsMask) ==
                clspv::version0::CLK_NORMALIZED_COORDS_TRUE;

            switch (entry.sampler_data.mask &
                    clspv::version0::kSamplerAddressMask) {
            case clspv::version0::CLK_ADDRESS_NONE:
                desc.addressing_mode = CL_ADDRESS_NONE;
                break;
            case clspv::version0::CLK_ADDRESS_CLAMP_TO_EDGE:
                desc.addressing_mode = CL_ADDRESS_CLAMP_TO_EDGE;
                break;
            case clspv::version0::CLK_ADDRESS_CLAMP:
                desc.addressing_mode = CL_ADDRESS_CLAMP;
                break;
            case clspv::version0::CLK_ADDRESS_MIRRORED_REPEAT:
                desc.addressing_mode = CL_ADDRESS_MIRRORED_REPEAT;
                break;
            case clspv::version0::CLK_ADDRESS_REPEAT:
                desc.addressing_mode = CL_ADDRESS_REPEAT;
                break;
            default:
                cvk_error("Invalid sampler addressing mode: %d",
                          entry.sampler_data.mask &
                              clspv::version0::kSamplerAddressMask);
                return false;
            }

            switch (entry.sampler_data.mask &
                    clspv::version0::kSamplerFilterMask) {
            case clspv::version0::CLK_FILTER_NEAREST:
                desc.filter_mode = CL_FILTER_NEAREST;
                break;
            case clspv::version0::CLK_FILTER_LINEAR:
                desc.filter_mode = CL_FILTER_LINEAR;
                break;
            default:
                cvk_error("Invalid sampler filter mode: %d",
                          entry.sampler_data.mask &
                              clspv::version0::kSamplerFilterMask);
                return false;
            }

            m_literal_samplers.push_back(desc);

            continue;
        }

        if (entry.kind ==
            clspv::version0::DescriptorMapEntry::Kind::PushConstant) {
            pushconstant pc;
            pushconstant_desc pcd;
            switch (entry.push_constant_data.pc) {
            case clspv::PushConstant::GlobalOffset:
                pc = pushconstant::global_offset;
                break;
            case clspv::PushConstant::EnqueuedLocalSize:
                pc = pushconstant::enqueued_local_size;
                break;
            default:
                cvk_error("Invalid push constant: %d",
                          static_cast<int>(entry.push_constant_data.pc));
                return false;
            }

            pcd.offset = entry.push_constant_data.offset;
            pcd.size = entry.push_constant_data.size;

            m_push_constants[pc] = pcd;

            continue;
        }

        if (entry.kind ==
            clspv::version0::DescriptorMapEntry::Kind::KernelDecl) {

            if (m_dmaps.count(entry.kernel_decl_data.kernel_name) == 0) {
                m_dmaps[entry.kernel_decl_data.kernel_name] = {};
            }

            continue;
        }

        if (entry.kind ==
            clspv::version0::DescriptorMapEntry::Kind::SpecConstant) {
            spec_constant constant;
            uint32_t id = entry.spec_constant_data.spec_id;
            switch (entry.spec_constant_data.spec_constant) {
            case clspv::SpecConstant::kWorkgroupSizeX:
                constant = spec_constant::workgroup_size_x;
                break;
            case clspv::SpecConstant::kWorkgroupSizeY:
                constant = spec_constant::workgroup_size_y;
                break;
            case clspv::SpecConstant::kWorkgroupSizeZ:
                constant = spec_constant::workgroup_size_z;
                break;
            case clspv::SpecConstant::kWorkDim:
                constant = spec_constant::work_dim;
                break;
            default:
                cvk_error(
                    "Unhandled spec constant: %d",
                    static_cast<int>(entry.spec_constant_data.spec_constant));
                return false;
            }
            m_spec_constants[constant] = id;

            continue;
        }

        if (entry.kind != clspv::version0::DescriptorMapEntry::Kind::KernelArg)
            return false;

        kernel_argument arg;
        arg.name = entry.kernel_arg_data.arg_name;
        arg.pos = entry.kernel_arg_data.arg_ordinal;
        if (entry.kernel_arg_data.arg_kind == clspv::ArgKind::Local) {
            arg.kind = kernel_argument_kind::local;
            arg.local_elem_size = entry.kernel_arg_data.local_element_size;
            arg.local_spec_id = entry.kernel_arg_data.local_spec_id;
        } else {
            arg.descriptorSet = entry.descriptor_set;
            arg.binding = entry.binding;
            arg.offset = entry.kernel_arg_data.pod_offset;
            switch (entry.kernel_arg_data.arg_kind) {
            case clspv::ArgKind::Buffer:
                arg.kind = kernel_argument_kind::buffer;
                break;
            case clspv::ArgKind::BufferUBO:
                arg.kind = kernel_argument_kind::buffer_ubo;
                break;
            case clspv::ArgKind::Pod:
                arg.kind = kernel_argument_kind::pod;
                break;
            case clspv::ArgKind::PodUBO:
                arg.kind = kernel_argument_kind::pod_ubo;
                break;
            case clspv::ArgKind::PodPushConstant:
                arg.kind = kernel_argument_kind::pod_pushconstant;
                break;
            case clspv::ArgKind::ReadOnlyImage:
                arg.kind = kernel_argument_kind::ro_image;
                break;
            case clspv::ArgKind::WriteOnlyImage:
                arg.kind = kernel_argument_kind::wo_image;
                break;
            case clspv::ArgKind::Sampler:
                arg.kind = kernel_argument_kind::sampler;
                break;
            default:
                return false;
            }
            if (arg.is_pod()) {
                arg.size = entry.kernel_arg_data.pod_arg_size;
            }
        }

        m_dmaps[entry.kernel_arg_data.kernel_name].push_back(arg);
    }

    cvk_debug_fn("num_kernels = %zu", num_kernels());

    return true;
}
#endif

void spir_binary::insert_descriptor_map(const spir_binary& other) {
    for (auto& args : other.kernels_arguments()) {
        m_dmaps[args.first] = args.second;
    }
    m_dmaps_text += other.m_dmaps_text;
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

cl_build_status cvk_program::compile_source(const cvk_device* device) {
    bool use_tmp_folder = true;
    bool save_headers = true;
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
    }

#ifndef CLSPV_ONLINE_COMPILER
    bool build_from_il = m_il.size() > 0;
    std::string clspv_input_file{tmp_folder + "/source"};
    std::string llvmspirv_input_file{tmp_folder + "/source.spv"};
    // Save input program to a file
    if (build_from_il) {
        clspv_input_file += ".bc";
        if (!save_il_to_file(llvmspirv_input_file, m_il)) {
            cvk_error_fn("Couldn't save IL to file!");
            return CL_BUILD_ERROR;
        }
    } else {
        clspv_input_file += ".cl";
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
        // Some applications pass this even when using uniform NDRanges
        // Swallow the flag to enable these use cases
        {"-cl-arm-non-uniform-work-group-size", ""},
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
    options += " -cluster-pod-kernel-args ";

    std::string single_precision_option = "-cl-single-precision-constant";
    if (processed_options.find(single_precision_option) == std::string::npos) {
        options += " " + single_precision_option + " ";
    }
    if (!devices_support_images()) {
        options += " -images=0 ";
    }
    options += " -max-pushconstant-size=" +
               std::to_string(device->vulkan_max_push_constants_size()) + " ";
    options += " -pod-ubo ";
    options += " -int8 ";
    if (device->supports_ubo_stdlayout()) {
        options += " -std430-ubo-layout ";
    }
    options += " -global-offset ";
    options += " " + gCLSPVOptions + " ";

#ifdef CLSPV_ONLINE_COMPILER
    cvk_info("About to compile \"%s\"", options.c_str());
    std::vector<clspv::version0::DescriptorMapEntry> entries;
    auto result = clspv::CompileFromSourceString(
        m_source, "", options, m_binary.raw_binary(), &entries);
    cvk_info("Return code was: %d", result);
    if (result != 0) {
        cvk_error_fn("failed to compile the program");
        return CL_BUILD_ERROR;
    }

    // Load descriptor map
    if (!m_binary.load_descriptor_map(entries)) {
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

        // Call the translator
        // TODO Sanity check the command / move away from system()
        int status = std::system(cmd.c_str());
        cvk_info("Return code was: %d", status);

        if (status != 0) {
            cvk_error_fn("failed to translate SPIR-V to LLVM IR");
            return CL_BUILD_ERROR;
        }
    }

    // Compose clspv command-line
    std::string cmd{gCLSPVPath};
    std::string descriptor_map_file{tmp_folder + "/descriptors.map"};
    std::string spirv_file{tmp_folder + "/compiled.spv"};

    if (build_from_il) {
        cmd += " -x ir ";
    }
    cmd += " -descriptormap=";
    cmd += descriptor_map_file;
    cmd += " ";
    cmd += clspv_input_file;
    cmd += " ";
    cmd += options;
    cmd += " -o ";
    cmd += spirv_file;
    cvk_info("About to run \"%s\"", cmd.c_str());

    // Call clspv
    // TODO Sanity check the command / move away from system()
    int status = std::system(cmd.c_str());
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
    if (!m_binary.load_descriptor_map(descriptor_map_file.c_str())) {
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

    // Merge descriptor maps
    for (cl_uint i = 0; i < m_num_input_programs; i++) {
        m_binary.insert_descriptor_map(m_input_programs[i]->m_binary);
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
        prepare_push_constant_range();
        break;
    case build_operation::link:
        status = link();
        break;
    }

    if ((m_operation == build_operation::compile) ||
        (status != CL_BUILD_SUCCESS)) {
        complete_operation(device, status);
        return;
    }

    // Validate
    // TODO validate with different rules depending on the binary type
    if ((m_binary_type == CL_PROGRAM_BINARY_TYPE_EXECUTABLE) &&
        !m_binary.validate()) {
        cvk_error("Could not validate SPIR-V binary.");
        complete_operation(device, CL_BUILD_ERROR);
        return;
    }

    // Check capabilities against the device.
    if ((m_binary_type == CL_PROGRAM_BINARY_TYPE_EXECUTABLE) &&
        !check_capabilities(device)) {
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

    // Create a shader module
    VkDevice dev = device->vulkan_device();

    VkShaderModuleCreateInfo moduleCreateInfo = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, // sType
        nullptr,                                     // pNext
        0,                                           // flags
        m_binary.spir_size(),                        // codeSize
        m_binary.spir_data()                         // pCode
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
    m_thread = new std::thread(&cvk_program::do_build, this);

    return true;
}

cvk_entry_point::cvk_entry_point(VkDevice dev, cvk_program* program,
                                 const std::string& name)
    : m_device(dev), m_context(program->context()), m_program(program),
      m_name(name), m_pod_descriptor_type(VK_DESCRIPTOR_TYPE_MAX_ENUM),
      m_pod_buffer_size(0u), m_has_pod_arguments(false),
      m_has_pod_buffer_arguments(false), m_descriptor_pool(VK_NULL_HANDLE),
      m_pipeline_layout(VK_NULL_HANDLE), m_pipeline_cache(VK_NULL_HANDLE) {}

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
    binding_stat_map& smap, uint32_t& num_resources) {
    bool pod_found = false;

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

        layoutBindings.push_back(binding);
        smap[binding.descriptorType]++;
    }

    num_resources = layoutBindings.size();

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
    if (!build_descriptor_sets_layout_bindings_for_arguments(bindingTypes,
                                                             m_num_resources)) {
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
    }

    // Don't pass the range at pipeline layout creation time if no push
    // constants are used
    uint32_t num_push_constant_ranges = 1;
    if (push_constant_range.offset == UINT32_MAX) {
        num_push_constant_ranges = 0;
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

    // Create pipeline cache
    VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
        nullptr, // pNext
        0,       // flags
        0,       // initialDataSize
        nullptr, // pInitialData
    };

    res = vkCreatePipelineCache(m_device, &pipelineCacheCreateInfo, nullptr,
                                &m_pipeline_cache);
    if (res != VK_SUCCESS) {
        cvk_error("Could not create pipeline cache.");
        return CL_OUT_OF_RESOURCES;
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
    VkResult res = vkCreateComputePipelines(m_device, m_pipeline_cache, 1,
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

std::unique_ptr<std::vector<uint8_t>>
cvk_entry_point::allocate_pod_pushconstant_buffer() {
    return std::make_unique<std::vector<uint8_t>>(m_pod_buffer_size);
}
