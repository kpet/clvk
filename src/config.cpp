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
    cfgval->value = atoi(txt);
    cfgval->set = true;
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

} // namespace

void init_config() {
    // TODO Parse config file
    // Parse environment
    parse_env();
}
