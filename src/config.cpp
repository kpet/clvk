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
#include "log.hpp"
#include "unit.hpp"
#include "utils.hpp"

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <vector>

const config_struct config;

namespace {

std::vector<config_option> gConfigOptions = {
#define OPTION(type, name, valdef)                                             \
    {option_type<type>(), #name, &config.name, false, false},
#define EARLY_OPTION(type, name, valdef)                                       \
    {option_type<type>(), #name, &config.name, true, false},
#define PROPERTY OPTION
#include "config.def"
#undef PROPERTY
#undef EARLY_OPTION
#undef OPTION
};

// Helper function to trim whitespace.
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n");
    size_t last = str.find_last_not_of(" \t\n");

    // Check for valid range
    if (first == std::string::npos || last == std::string::npos) {
        return str; // Empty or only whitespace
    }

    // Extract the trimmed string
    std::string trimmed = str.substr(first, (last - first + 1));
    return trimmed;
}

void parse_string_set(void* value_ptr, const char* txt) {
    auto cfgval = static_cast<config_value<std::set<std::string>>*>(value_ptr);

    size_t pos = 0;
    std::string s(txt);
    size_t comma = s.find(',', pos);
    while (comma != std::string::npos) {
        std::string s_substr = trim(s.substr(pos, comma - pos));
        if (!s_substr.empty()) {
            cfgval->value.insert(s_substr);
        }
        pos = comma + 1;
        comma = s.find(',', pos);
    }
    std::string s_substr = trim(s.substr(pos));
    if (!s_substr.empty()) {
        cfgval->value.insert(s_substr);
    }

    cfgval->set = true;
}

#define CASE(X)                                                                \
    if (s == std::string(#X)) {                                                \
        val = X;                                                               \
        return true;                                                           \
    }
bool channel_order_from_string(std::string s, cl_channel_order& val) {
    CASE(CL_R)
    CASE(CL_A)
    CASE(CL_DEPTH)
    CASE(CL_LUMINANCE)
    CASE(CL_INTENSITY)
    CASE(CL_RG)
    CASE(CL_RA)
    CASE(CL_Rx)
    CASE(CL_RGB)
    CASE(CL_RGx)
    CASE(CL_RGBA)
    CASE(CL_ARGB)
    CASE(CL_BGRA)
    CASE(CL_ABGR)
    CASE(CL_RGBx)
    CASE(CL_sRGB)
    CASE(CL_sRGBA)
    CASE(CL_sBGRA)
    CASE(CL_sRGBx)
    return false;
}
bool channel_type_from_string(std::string s, cl_channel_type& val) {
    CASE(CL_SNORM_INT8)
    CASE(CL_SNORM_INT16)
    CASE(CL_UNORM_INT8)
    CASE(CL_UNORM_INT16)
    CASE(CL_UNORM_SHORT_565)
    CASE(CL_UNORM_SHORT_555)
    CASE(CL_UNORM_INT_101010)
    CASE(CL_UNORM_INT_101010_2)
    CASE(CL_SIGNED_INT8)
    CASE(CL_SIGNED_INT16)
    CASE(CL_SIGNED_INT32)
    CASE(CL_UNSIGNED_INT8)
    CASE(CL_UNSIGNED_INT16)
    CASE(CL_UNSIGNED_INT32)
    CASE(CL_HALF_FLOAT)
    CASE(CL_FLOAT)
    return false;
}
#undef CASE

void parse_image_format_set(void* value_ptr, const char* txt) {
    auto cfgval = static_cast<config_value<image_format_set>*>(value_ptr);

    std::string s(txt);
    size_t bracket = s.find('{', 0);
    while (bracket != std::string::npos) {
        size_t comma = s.find(',', bracket);
        if (comma == std::string::npos) {
            cvk_warn_fn("Could not find expected ',' after '{' in '%s'",
                        s.c_str());
            return;
        }
        cl_image_format format;
        auto order = trim(s.substr(bracket + 1, comma - bracket - 1));
        if (!channel_order_from_string(order, format.image_channel_order)) {
            cvk_warn_fn("Could not parse order '%s' from '%s'", order.c_str(),
                        s.c_str());
            return;
        }
        bracket = s.find('}', comma);
        if (bracket == std::string::npos) {
            cvk_warn_fn("Could not find expected '}' after ',' in '%s'",
                        s.c_str());
            return;
        }
        auto data_type = trim(s.substr(comma + 1, bracket - comma - 1));
        if (!channel_type_from_string(data_type,
                                      format.image_channel_data_type)) {
            cvk_warn_fn("Could not parse data_type '%s' from '%s'",
                        data_type.c_str(), s.c_str());
            return;
        }
        cfgval->value.insert(format);

        bracket = s.find("{", bracket);
    }

    cfgval->set = true;
}

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

static constexpr size_t txt_size = 4096;
char gTxt[txt_size];

char* print_string_set(void* value_ptr) {
    auto cfgval = static_cast<config_value<std::set<std::string>>*>(value_ptr);
    size_t pos = 1;
    gTxt[0] = '\'';
    for (auto& s : cfgval->value) {
        pos += snprintf(&gTxt[pos], txt_size - pos, "%s, ", s.c_str());
    }
    gTxt[pos - 2] = '\'';
    gTxt[pos - 1] = '\0';
    return gTxt;
}

char* print_image_format_set(void* value_ptr) {
    auto cfgval = static_cast<config_value<image_format_set>*>(value_ptr);
    size_t pos = 1;
    gTxt[0] = '\'';
    for (auto& format : cfgval->value) {
        pos += snprintf(
            &gTxt[pos], txt_size - pos, "{%s, %s}, ",
            cl_channel_order_to_string(format.image_channel_order).c_str(),
            cl_channel_type_to_string(format.image_channel_data_type).c_str());
    }
    gTxt[pos - 2] = '\'';
    gTxt[pos - 1] = '\0';
    return gTxt;
}

char* print_string(void* value_ptr) {
    auto cfgval = static_cast<config_value<std::string>*>(value_ptr);
    snprintf(gTxt, txt_size, "'%s'", cfgval->value.c_str());
    return gTxt;
}

char* print_boolean(void* value_ptr) {
    auto cfgval = static_cast<config_value<bool>*>(value_ptr);
    snprintf(gTxt, txt_size, "%s", cfgval->value ? "true" : "false");
    return gTxt;
}

char* print_uint32(void* value_ptr) {
    auto cfgval = static_cast<config_value<uint32_t>*>(value_ptr);
    snprintf(gTxt, txt_size, "%u (0x%x)", cfgval->value, cfgval->value);
    return gTxt;
}

std::string get_clvk_env_name(const std::string& name) {
    std::string var_name = "CLVK_";
    std::string optname_upper(name);
    std::transform(optname_upper.begin(), optname_upper.end(),
                   optname_upper.begin(), ::toupper);
    var_name += optname_upper;
    return var_name;
}

void read_config_file(std::unordered_map<std::string, std::string>& umap,
                      std::ifstream& config_stream) {

    std::string line;
    while (std::getline(config_stream, line)) {
        // Ignore comments and empty lines
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Parse key-value pairs
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = trim(line.substr(0, pos));
            std::string value = trim(line.substr(pos + 1));
            // Store values (if any)
            if (value != "") {
                umap[key] = value;
                cvk_debug_group_fn(loggroup::cfg, "'%s' = '%s'", key.c_str(),
                                   value.c_str());
            }
        } else {
            cvk_warn_group_fn(loggroup::cfg, "%s , %s",
                              "The following line is malformed", line.c_str());
        }
    }
    config_stream.close();
}

void parse_option(void* val, config_option& opt, const char* txt) {
    switch (opt.type) {
    case config_option_type::string:
        parse_string(val, txt);
        break;
    case config_option_type::boolean:
        parse_boolean(val, txt);
        break;
    case config_option_type::uint32:
        parse_uint32(val, txt);
        break;
    case config_option_type::string_set:
        parse_string_set(val, txt);
        break;
    case config_option_type::image_format_set:
        parse_image_format_set(val, txt);
        break;
    }
    opt.set = true;
}

void parse_config_file(bool early_option) {
    std::unordered_map<std::string, std::string> file_config_values;
    std::string conf_file = "clvk.conf";
    std::ifstream config_stream;

    std::vector<std::string> config_file_paths;
    config_file_paths.push_back("/etc/clvk.conf");
#ifdef __ANDROID__
    config_file_paths.push_back("/system/etc/clvk.conf");
    config_file_paths.push_back("/vendor/etc/clvk.conf");
#endif
    config_file_paths.push_back("/usr/local/etc/clvk.conf");
    config_file_paths.push_back("~/.config/clvk.conf");
    config_file_paths.push_back(
        (std::filesystem::current_path() / conf_file).string());
    if (!config.config_file().empty()) {
        config_file_paths.push_back(config.config_file());
    }

    for (auto& curr_path : config_file_paths) {
        if (!std::filesystem::exists(curr_path)) {
            continue;
        }
        config_stream.open(curr_path);
        if (!config_stream.is_open()) {
            cvk_error("Error opening config file - %s", curr_path.c_str());
        }
        cvk_info_group_fn(loggroup::cfg, "Parsing config file '%s'",
                          curr_path.c_str());
        read_config_file(file_config_values, config_stream);
    }

    for (auto& opt : gConfigOptions) {
        if (early_option != opt.early_option) {
            continue;
        }
        if (file_config_values.find(opt.name) == file_config_values.end()) {
            continue;
        }
        if (opt.set) { // means already set by 'parse_env'
            continue;
        }
        CVK_ASSERT(file_config_values[opt.name].length() > 0);
        auto curr_conf = (file_config_values[opt.name]).c_str();
        void* optval = const_cast<void*>(opt.value);
        parse_option(optval, opt, curr_conf);
    }
    return;
}

void parse_env(bool early_option) {
    for (auto& opt : gConfigOptions) {
        if (early_option != opt.early_option) {
            continue;
        }
        auto var_name = get_clvk_env_name(opt.name);
        const char* txt = getenv(var_name.c_str());
        if (txt == nullptr) {
            continue;
        }
        cvk_debug_group_fn(loggroup::cfg, "'%s' = '%s'", var_name.c_str(), txt);
        void* optval = const_cast<void*>(opt.value);
        parse_option(optval, opt, txt);
    }
}

void print_config() {
    cvk_info_group_fn(loggroup::cfg, "");
    std::vector<config_option> options(gConfigOptions.size());
    std::partial_sort_copy(gConfigOptions.begin(), gConfigOptions.end(),
                           options.begin(), options.end(),
                           [](const config_option& a, const config_option& b) {
                               return a.name < b.name;
                           });
    for (const auto& opt : options) {
        void* optval = const_cast<void*>(opt.value);
        char* txt = print_option(opt.type, optval);
        if (opt.set) {
            cvk_info_group(loggroup::cfg, "  *%s: %s", opt.name.c_str(), txt);
        } else {
            cvk_debug_group(loggroup::cfg, "  %s: %s", opt.name.c_str(), txt);
        }
    }
}

} // namespace

bool operator==(const image_format_set& a, const image_format_set& b) {
    for (auto& elema : a) {
        if (b.count(elema) == 0) {
            return false;
        }
    }
    for (auto& elemb : b) {
        if (a.count(elemb) == 0) {
            return false;
        }
    }
    return true;
}

char* print_option(config_option_type type, void* val) {
    switch (type) {
    case config_option_type::string:
        return print_string(val);
    case config_option_type::boolean:
        return print_boolean(val);
    case config_option_type::uint32:
        return print_uint32(val);
    case config_option_type::string_set:
        return print_string_set(val);
    case config_option_type::image_format_set:
        return print_image_format_set(val);
    }
}

void init_config() {
    parse_env(false);
    parse_config_file(false);
    print_config();
}

void init_early_config() {
    parse_env(true);
    parse_config_file(true);
}
