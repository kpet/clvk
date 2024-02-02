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

config_option_type infer_type(const std::string& valueStr) {
    // 1. Check for boolean values
    if (valueStr == "true" || valueStr == "false") {
        return config_option_type::boolean;
    }

    // 2. Attempt integer conversion
    try {
        std::stoul(valueStr); // Test if conversion is possible
        return config_option_type::uint32;
    } catch (const std::invalid_argument&) {
        // Fall through to string if integer conversion fails
    }

    // 3. If all else fails, assume it's a string
    return config_option_type::string;
}

void read_config_file(std::unordered_map<std::string, std::string>& umap) {
    std::string line;
    std::ifstream config_stream;
    // Get current working directory
    auto current_path = std::filesystem::current_path();
    // Create the full path to the config file
    std::string full_config_path = (current_path / config_file).string();
    config_stream.open(full_config_path);
    if (!config_stream.is_open()) {
        std::cerr << "Error opening config.toml" << std::endl;
        return;
    }

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

void parse_env() {
    std::unordered_map<std::string, std::string> fileConfigValues;
    read_config_file(fileConfigValues);
    for (auto& opt : gConfigOptions) {
        std::string var_name = "CLVK_";
        std::string optname_upper(opt.name);
        std::transform(optname_upper.begin(), optname_upper.end(),
                       optname_upper.begin(), ::toupper);
        var_name += optname_upper;
        const char* txt = getenv(var_name.c_str());
        if (txt == nullptr) {
            if (fileConfigValues.find(opt.name) == fileConfigValues.end() ||
                fileConfigValues[opt.name].length() == 0) {
                continue;
            }
            auto curr_val = fileConfigValues[opt.name];
            config_option_type optType = infer_type(fileConfigValues[opt.name]);
            void* optval = const_cast<void*>(opt.value);
            switch (optType) {
            case config_option_type::string:
                parse_string(optval, curr_val.c_str());
                continue;
            case config_option_type::boolean:
                parse_boolean(optval, curr_val.c_str());
                continue;
            case config_option_type::uint32:
                parse_uint32(optval, curr_val.c_str());
                continue;
            }
        } else {
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
}

} // namespace

void init_config() {
    // TODO Parse config file
    // Parse environment
    parse_env();
}
