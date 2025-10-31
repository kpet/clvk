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

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

enum class config_option_type
{
    string,
    uint32,
    boolean,
};

struct config_option {
    config_option_type type;
    std::string name;
    const void* value;
    bool early_option;
};

template <typename T> struct config_value {
    explicit config_value(const char* val) : value(val) {}
    explicit config_value(uint32_t val) : value(val) {}
    explicit config_value(bool val) : value(val) {}
    bool set;
    T value;
    operator T() const { return value; }
    T& operator()() { return value; }
    const T& operator()() const { return value; }
};

struct config_struct {
#define OPTION(type, name, valdef) const config_value<type> name{valdef};
#define EARLY_OPTION OPTION
#include "config.def"
#undef EARLY_OPTION
#undef OPTION
};

extern const config_struct config;

extern void init_config();

extern void init_early_config();
