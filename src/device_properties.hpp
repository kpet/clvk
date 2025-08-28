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

#include <memory>
#include <set>
#include <string>

#include "cl_headers.hpp"
#include "config.hpp"
#include "utils.hpp"

#define OPTION(type, name, valdef)
#define EARLY_OPTION OPTION
struct cvk_device_properties_virtual {
#define PROPERTY(type, name, valdef)                                           \
    virtual type name() const { return init(config.name(), type()); }
#include "config.def"
#undef PROPERTY

    static std::set<std::string> init(std::set<std::string> config,
                                      std::set<std::string> property) {
        std::set<std::string> set;
        set.insert(config.begin(), config.end());
        set.insert(property.begin(), property.end());
        return set;
    }
    static image_format_set init(image_format_set config,
                                 image_format_set property) {
        image_format_set set;
        set.insert(config.begin(), config.end());
        set.insert(property.begin(), property.end());
        return set;
    }
    static std::string init(std::string config, std::string property) {
        UNUSED(property);
        return config;
    }
    static uint32_t init(uint32_t config, uint32_t property) {
        UNUSED(property);
        return config;
    }
    static bool init(bool config, uint32_t property) {
        UNUSED(property);
        return config;
    }

    virtual ~cvk_device_properties_virtual() {}
};

struct cvk_device_properties {
    static constexpr uint32_t txt_size = FILENAME_MAX;

    cvk_device_properties(
        std::unique_ptr<cvk_device_properties_virtual> properties) {
        cvk_info_group_fn(loggroup::cfg, "");
#define PROPERTY(type, name, valdef)                                           \
    if (config.name.set) {                                                     \
        m_##name = cvk_device_properties_virtual::init(config.name(),          \
                                                       properties->name());    \
    } else {                                                                   \
        m_##name = properties->name();                                         \
    };                                                                         \
    {                                                                          \
        config_value<type> val(m_##name);                                      \
        char* txt = print_option(option_type<type>(), &val);                   \
        if (m_##name != config.name()) {                                       \
            cvk_info_group(loggroup::cfg, "  *" #name ": %s", txt);            \
        } else {                                                               \
            cvk_debug_group(loggroup::cfg, "  " #name ": %s", txt);            \
        }                                                                      \
    }
#include "config.def"
#undef PROPERTY
    }
#define PROPERTY(type, name, valdef)                                           \
    type name() const { return m_##name; }
#include "config.def"
#undef PROPERTY
private:
#define PROPERTY(type, name, valdef) type m_##name;
#include "config.def"
#undef PROPERTY
};
#undef EARLY_OPTION
#undef OPTION

std::unique_ptr<cvk_device_properties> create_cvk_device_properties(
    const char* name, const uint32_t vendorID, const uint32_t deviceID,
    const uint32_t driverVersion, const VkDriverId driverId);
