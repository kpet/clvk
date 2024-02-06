// Copyright 2022 The clvk authors.
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

#include "config.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

const config_struct config;

namespace {

template <typename T> constexpr config_option_type option_type() = delete;

#define DEFINE_OPTION_TYPE_GETTER(ctype, type)                                 \
    template <> constexpr config_option_type option_type<ctype>() {            \
        return type;                                                           \
    }

DEFINE_OPTION_TYPE_GETTER(std::string, config_option_type::string)
DEFINE_OPTION_TYPE_GETTER(uint32_t, config_option_type::uint32)
DEFINE_OPTION_TYPE_GETTER(bool, config_option_type::boolean)

config_option gConfigOptions[] = {
#define OPTION(type, name, valdef) {option_type<type>(), #name, &config.name},
#include "config.def"
#undef OPTION
};

void parse_string(void* value_ptr, const char* txt) {
    auto cfgval = static_cast<config_value<std::string>*>(value_ptr);
    cfgval->value.assign(txt);
    cfgval->set = true;
}

void parse_boolean(void* value_ptr, const char* txt) {
    auto cfgval = static_cast<config_value<bool>*>(value_ptr);
    cfgval->value = atoi(txt);
    cfgval->set = true;
}

void parse_uint32(void* value_ptr, const char* txt) {
    auto cfgval = static_cast<config_value<uint32_t>*>(value_ptr);
    int base = 10;
    if (strlen(txt) >= 2 && txt[0] == '0' && txt[1] == 'x') {
        base = 16;
    }
    cfgval->value = std::stoul(txt, nullptr, base);
    cfgval->set = true;
}

// Helper function to trim whitespace and remove quotations
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\""); // Also include quotes
    size_t last = str.find_last_not_of(" \t\n\"");

    // Check for valid range
    if (first == std::string::npos || last == std::string::npos) {
        return str; // Empty or only whitespace/quotes
    }

    // Extract the trimmed and unquoted substring
    std::string trimmed = str.substr(first, (last - first + 1));

    // Check if the trimmed string still starts and ends with quotes
    if (trimmed.front() == '"' && trimmed.back() == '"') {
        return trimmed.substr(1, trimmed.size() - 2); // Remove the quotes
    } else {
        return trimmed;
    }
}

void read_config_file(std::unordered_map<std::string, std::string>& umap, std::ifstream& config_stream) {
    std::string line;
    while (std::getline(config_stream, line)) {
        // Ignore comments and empty lines
        if (line.empty() || line[0] == '#') {
            continue;
        }
        line = trim(line);
        // Check for section headers
        if (line[0] == '[' && line[line.size() - 1] == ']') {
            continue;
        } else {
            // Parse key-value pairs
            size_t pos = line.find('=');
            if (pos != std::string::npos) {
                std::string key = trim(line.substr(0, pos));
                std::string value = trim(line.substr(pos + 1));

                // Store values (if any)
                if (value != "") {
                    umap[key] = value;
                }
            }
        }
    }
    config_stream.close();
}

void parse_config_file() {
    std::unordered_map<std::string, std::string> file_config_values;
    std::string conf_file = "clvk.conf";
    std::ifstream config_stream;

    // First check if env var has file
    std::string conv_file_env_var = "CLVK_CONFIG_FILE";
    const char* conf_file_env_path = getenv(conv_file_env_var.c_str());
    std::vector<std::string> config_file_paths;
    if (conf_file_env_path != nullptr) {
        config_file_paths.push_back(conf_file_env_path);
    }
    config_file_paths.push_back("/etc/clvk.conf");
    config_file_paths.push_back("~/config/clvk.conf");
    config_file_paths.push_back((std::filesystem::current_path() / conf_file).string());

    bool config_found = false;
    for (auto& curr_path : config_file_paths) {
        if (!std::filesystem::exists(curr_path))
            continue;

        config_stream.open(curr_path);
        if (!config_stream.is_open()) {
            std::cerr << "Error opening config file -" << curr_path << std::endl;
        }
        config_found = true;
        break;
    }

    if (!config_found) {
        std::cerr << "Error: No valid configuration file found. "
                  << "Please check the following locations:\n";
        for (auto& path : config_file_paths) {
            std::cerr << " - " << path << std::endl;
        }
        return;
    }

    read_config_file(file_config_values, config_stream);
    for (auto& opt : gConfigOptions) {
        if (file_config_values.find(opt.name) == file_config_values.end() ||
            file_config_values[opt.name].length() == 0) {
                    continue;
        }
        auto curr_conf = (file_config_values[opt.name]).c_str();
        void* optval = const_cast<void*>(opt.value);
        switch (opt.type) {
        case config_option_type::string:
            parse_string(optval, curr_conf);
            break;
        case config_option_type::boolean:
            parse_boolean(optval, curr_conf);
            break;
        case config_option_type::uint32:
            parse_uint32(optval, curr_conf);
            break;
        }
    }
}

void parse_env() {
    for (auto& opt : gConfigOptions) {
        std::string var_name = "CLVK_";
        std::string optname_upper(opt.name);
        std::transform(optname_upper.begin(), optname_upper.end(),
                       optname_upper.begin(), ::toupper);
        var_name += optname_upper;
        // printf("var_name = '%s' ", var_name.c_str());
        const char* txt = getenv(var_name.c_str());
        if (txt == nullptr) {
            //    printf("is not set\n");
            continue;
        }
        // printf("is set\n");
        void* optval = const_cast<void*>(opt.value);
        switch (opt.type) {
        case config_option_type::string:
            parse_string(optval, txt);
            break;
        case config_option_type::boolean:
            parse_boolean(optval, txt);
            break;
        case config_option_type::uint32:
            parse_uint32(optval, txt);
            break;
        }
    }
}

void gen_config_file() {
    std::unordered_map<std::string, std::string> file_config_values;
    std::ifstream config_def("config.def");
    std::ofstream config_file("clvk.conf");
    if (!config_def.is_open()) {
        std::cerr << "Error opening config.def\n";
        return;
    }
    if (!config_file.is_open()) {
        std::cerr << "Error opening clvk.conf\n";
        return;
    }
    read_config_file(file_config_values, config_def);
    for (auto& key_val : file_config_values ) {
        config_file << key_val.first << "=" << key_val.second << "\n" ;
    }
}

} // namespace

void init_config() {
    // make a default config file from config.def
    gen_config_file();
    parse_config_file();
    parse_env();
}
