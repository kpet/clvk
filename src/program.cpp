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
#include "spirv/1.0/spirv.hpp"
#include "spirv-tools/linker.hpp"
#include "spirv-tools/optimizer.hpp"

#include "program.hpp"
#include "utils.hpp"

struct membuf : public std::streambuf {
    membuf(const unsigned char *begin, const unsigned char *end) {
        auto sbegin = reinterpret_cast<char*>(const_cast<unsigned char*>(begin));
        auto send = reinterpret_cast<char*>(const_cast<unsigned char*>(end));
        setg(sbegin, sbegin, send);
    }
    membuf(unsigned char *begin, unsigned char *end) {
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

bool spir_binary::load_spir(std::istream &istream, uint32_t size)
{
    m_code.assign(size / SPIR_WORD_SIZE, 0);
    istream.read(reinterpret_cast<char*>(m_code.data()), size);
    return istream.good();
}

bool spir_binary::load_spir(const char *fname)
{
    std::ifstream ifile;

    ifile.open(fname, std::ios::in | std::ios::binary);

    if (!ifile.is_open())
    {
        cvk_error("Failed to open %s", fname);
        return false;
    }

    ifile.seekg(0, std::ios::end);
    uint32_t size = ifile.tellg();
    ifile.seekg(0, std::ios::beg);

    return load_spir(ifile, size);
}

bool spir_binary::load(std::istream &istream)
{
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

bool spir_binary::read(const unsigned char *src, size_t size)
{
    membuf bufview(src, src + size);
    std::istream istream(&bufview);
    return load(istream);
}

bool spir_binary::save_spir(const char *fname) const
{
    std::ofstream ofile;

    ofile.open(fname, std::ios::out | std::ios::binary);

    if (!ofile.is_open()) {
        return false;
    }

    ofile.write(reinterpret_cast<const char*>(m_code.data()), m_code.size() * sizeof(SPIR_WORD_SIZE));

    return ofile.good();
}

bool spir_binary::save(std::ostream &ostream) const
{
    // Write magic
    ostream.write(reinterpret_cast<const char*>(&MAGIC), sizeof(MAGIC));

    // Write SPIR
    uint32_t spir_size = m_code.size() * SPIR_WORD_SIZE;
    ostream.write(reinterpret_cast<const char*>(&spir_size), sizeof(spir_size));
    ostream.write(reinterpret_cast<const char*>(m_code.data()), spir_size);

    // Write DMAP
    uint32_t dmap_size = m_dmaps_text.size();
    ostream.write(reinterpret_cast<const char*>(&dmap_size), sizeof(dmap_size));
    ostream.write(reinterpret_cast<const char*>(m_dmaps_text.c_str()), dmap_size);

    return ostream.good();
}

bool spir_binary::save(const char *fname) const
{
    std::ofstream ofile;

    ofile.open(fname, std::ios::out | std::ios::binary);

    if (!ofile.is_open()) {
        return false;
    }

    return save(ofile);
}

size_t spir_binary::size() const
{
    return sizeof(MAGIC) +
           sizeof(uint32_t) +
           (m_code.size() * SPIR_WORD_SIZE) +
           sizeof(uint32_t) +
           m_dmaps_text.size();
}

bool spir_binary::write(unsigned char *dst) const
{
    membuf bufview(dst, dst + size());
    std::ostream ostream(&bufview);
    return save(ostream);
}

void spir_binary::use(std::vector<uint32_t> &&src)
{
    m_code = std::move(src);
}
    
void spir_binary::set_target_env(spv_target_env env)
{
    spvContextDestroy(m_context);
    m_context = spvContextCreate(env);
}

bool spir_binary::validate() const
{
    spv_diagnostic diag;
    spv_result_t res = spvValidateBinary(m_context, m_code.data(), m_code.size(), &diag);
    spvDiagnosticPrint(diag);
    spvDiagnosticDestroy(diag);
    return res == SPV_SUCCESS;
}

bool parse_local_arg(kernel_argument &arg, const std::vector<std::string> &tokens, int toknum)
{
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

bool parse_arg(kernel_argument &arg, const std::vector<std::string> &tokens, int toknum)
{
    if (tokens[toknum++] != "descriptorSet") {
        return false;
    }

    arg.descriptorSet = atoi(tokens[toknum++].c_str());

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

bool spir_binary::load_descriptor_map(std::istream &istream)
{
    m_dmaps.clear();

    std::string line;
    while (std::getline(istream, line)) {
        m_dmaps_text += line + "\n";
        cvk_debug("DMAP line: %s", line.c_str());
        const char *delim = ",";
        size_t start = line.find_first_not_of(delim), end;
        std::vector<std::string> tokens;

        while (start != std::string::npos) {
            end = line.find(delim, start);
            tokens.push_back(line.substr(start, end-start));
            start = line.find_first_not_of(delim, end);
        }

        //for (auto& tok : tokens) {
        //    cvk_debug("TOK: %s", tok.c_str());
        //}

        int toknum = 0;
        kernel_argument arg;

        if (tokens[toknum++] != "kernel") {
            return false;
        }

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
        } else {
            return false;
        }

        m_dmaps[kname].push_back(arg);
    }

    cvk_debug_fn("num_kernels = %zu", num_kernels());

    return true;
}

bool spir_binary::load_descriptor_map(const char *fname)
{
    std::ifstream ifile;

    ifile.open(fname, std::ios::in);

    if (!ifile.is_open())
    {
        cvk_error("Failed to open %s", fname);
        return false;
    }

    return load_descriptor_map(ifile);
}

#ifdef CLSPV_ONLINE_COMPILER
bool spir_binary::load_descriptor_map(const std::vector<clspv::version0::DescriptorMapEntry> &entries)
{
  m_dmaps.clear();
  for (const auto &entry : entries) {
    if (gLoggingLevel == loglevel::debug) {
      std::string s;
      std::ostringstream str(s);
      str << entry;
      cvk_debug("DMAP line: %s", str.str().c_str());
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
        case clspv::ArgKind::ReadOnlyImage:
          arg.kind = kernel_argument_kind::ro_image;
          break;
        case clspv::ArgKind::WriteOnlyImage:
          arg.kind = kernel_argument_kind::wo_image;
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

void spir_binary::insert_descriptor_map(const spir_binary &other)
{
    for (auto &args: other.kernels_arguments()) {
        m_dmaps[args.first] = args.second;
    }
    m_dmaps_text += other.m_dmaps_text;
}

bool save_string_to_file(const std::string &fname, const std::string &text)
{
    std::ofstream ofile{fname};

    if (!ofile.is_open()) {
        return false;
    }

    ofile.write(text.c_str(), text.size());
    ofile.close();

    return ofile.good();
}

cl_build_status cvk_program::compile_source()
{
    bool use_tmp_folder = true;
    bool save_headers = true;
#ifdef CLSPV_ONLINE_COMPILER
    use_tmp_folder = m_operation == build_operation::compile && m_num_input_programs > 0;
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
    // Save source to file
    std::string src_file{tmp_folder + "/source.cl"};
    if (!save_string_to_file(src_file, m_source)) {
        cvk_error_fn("Couldn't save source to file!");
        return CL_BUILD_ERROR;
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
    };

    for (auto &subst : option_substitutions) {
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

    options += " -pod-ubo ";
    options += " -int8 ";
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
    // Compose clspv command-line
    std::string cmd{gCLSPVPath};
    std::string descriptor_map_file{tmp_folder + "/descriptors.map"};
    std::string spirv_file{tmp_folder + "/compiled.spv"};

    cmd += " -descriptormap=";
    cmd += descriptor_map_file;
    cmd += " ";
    cmd += src_file;
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
    const char *filename = spirv_file.c_str();
    if (!m_binary.load_spir(filename)) {
        cvk_error("Could not load SPIR-V binary from \"%s\"", filename);
        return CL_BUILD_ERROR;
    } else {
        cvk_info("Loaded SPIR-V binary from \"%s\", size = %zu words", filename, m_binary.code().size());
    }

    // Load descriptor map
    if (!m_binary.load_descriptor_map(descriptor_map_file.c_str())) {
        cvk_error("Could not load descriptor map for SPIR-V binary.");
        return CL_BUILD_ERROR;
    }
#endif

    return CL_BUILD_SUCCESS;
}

cl_build_status cvk_program::link()
{
    spvtools::Context context(SPV_ENV_VULKAN_1_0);
    std::vector<uint32_t> linked;
    std::vector<std::vector<uint32_t>> binaries(m_num_input_programs);

    const spvtools::MessageConsumer consumer = [](spv_message_level_t level,
                                                  const char*,
                                                  const spv_position_t &position,
                                                  const char* message) {

        #define msgtpl "spvtools says '%s' at position %zu"
        switch(level) {
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
        cvk_debug_fn("program %u, %zu kernels...", i, m_input_programs[i]->m_binary.num_kernels());
        cvk_debug_fn("about to copy code, size = %zu", m_input_programs[i]->m_binary.code().size());
        binaries[i] = m_input_programs[i]->m_binary.code();
    }

    cvk_debug_fn("linking...");
    spvtools::LinkerOptions linker_options;
    linker_options.SetCreateLibrary(create_library);
    spv_result_t res = spvtools::Link(context, binaries, &linked, linker_options);

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

    cvk_debug_fn("linked binary has %zu kernels", m_binary.kernels_arguments().size());

    return CL_BUILD_SUCCESS;
}

void cvk_program::do_build()
{
    cl_build_status status = CL_BUILD_SUCCESS;

    auto device = m_context->device();

    switch (m_operation) {
    case build_operation::compile:
    case build_operation::build:
        // Compile source and load binary
        if (!m_binary.loaded_from_binary()) {
            status = compile_source();
        }
        break;
    case build_operation::link:
        status = link();
        break;
    }

    if ((m_operation == build_operation::compile) || (status != CL_BUILD_SUCCESS)) {
        complete_operation(device, status);
        return;
    }

    // Validate
    // TODO validate with different rules depending on the binary type
    if ((m_binary_type == CL_PROGRAM_BINARY_TYPE_EXECUTABLE) && !m_binary.validate()) {
        cvk_error("Could not validate SPIR-V binary.");
        complete_operation(device, CL_BUILD_ERROR);
        return;
    }

    // Create a shader module
    VkDevice dev = device->vulkan_device();

    VkShaderModuleCreateInfo moduleCreateInfo = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, // sType
        nullptr, // pNext
        0, // flags
        m_binary.spir_size(), // codeSize
        m_binary.spir_data() // pCode
    };

    VkResult res = vkCreateShaderModule(dev, &moduleCreateInfo, nullptr, &m_shader_module);

    if (res != VK_SUCCESS) {
        cvk_error("vkCreateShaderModule returned %d", res);
        complete_operation(device, CL_BUILD_ERROR);
        return;
    }

    complete_operation(device, CL_BUILD_SUCCESS);
}

bool cvk_program::build(build_operation operation, cl_uint num_devices, const cvk_device *const*device_list, const char *options, cl_uint num_input_programs, const cvk_program *const*input_programs, const char **header_include_names, cvk_program_callback cb, void *data)
{
    if (!m_lock.try_lock()) {
        return false;
    }

    retain();

    for (cl_uint i = 0; i < num_input_programs; i++) {
        cvk_program *iprog = const_cast<cvk_program*>(input_programs[i]);
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
            m_dev_status[device_list[i]] = CL_BUILD_IN_PROGRESS;
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
