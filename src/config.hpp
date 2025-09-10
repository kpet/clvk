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

#pragma once

#include "cl_headers.hpp"
#include "image_format.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <unordered_set>

enum class config_option_type
{
    string,
    uint32,
    boolean,
    string_set,
    image_format_set,
};

struct config_option {
    config_option_type type;
    std::string name;
    const void* value;
    bool early_option;
    bool set;
};

using image_format_set =
    std::unordered_set<cl_image_format, ClFormatHash, ClFormatEqual>;
bool operator==(const image_format_set& a, const image_format_set& b);

template <typename T> struct config_value {
    explicit config_value(const char* val) : value(val) {}
    explicit config_value(uint32_t val) : value(val) {}
    explicit config_value(bool val) : value(val) {}
    explicit config_value(std::string val) : value(val) {}
    explicit config_value(std::set<std::string> val) : value(val) {}
    explicit config_value(image_format_set val) : value(val) {}
    bool set;
    T value;
    operator T() const { return value; }
    T& operator()() { return value; }
    const T& operator()() const { return value; }
};

struct config_struct {
#define OPTION(type, name, valdef) const config_value<type> name{valdef};
#define EARLY_OPTION OPTION
#define PROPERTY OPTION
#include "config.def"
#undef PROPERTY
#undef EARLY_OPTION
#undef OPTION
};

extern const config_struct config;

extern void init_config();

extern void init_early_config();

template <typename T> constexpr config_option_type option_type() = delete;

#define DEFINE_OPTION_TYPE_GETTER(ctype, type)                                 \
    template <> constexpr config_option_type option_type<ctype>() {            \
        return type;                                                           \
    }

DEFINE_OPTION_TYPE_GETTER(std::string, config_option_type::string)
DEFINE_OPTION_TYPE_GETTER(uint32_t, config_option_type::uint32)
DEFINE_OPTION_TYPE_GETTER(bool, config_option_type::boolean)
DEFINE_OPTION_TYPE_GETTER(std::set<std::string>, config_option_type::string_set)
DEFINE_OPTION_TYPE_GETTER(image_format_set,
                          config_option_type::image_format_set)

char* print_option(config_option_type type, void* val);
