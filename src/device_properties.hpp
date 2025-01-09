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
#include <unordered_set>

#include "cl_headers.hpp"
#include "config.hpp"
#include "image_format.hpp"

struct cvk_device_properties {
    virtual std::string vendor() const { return "Unknown vendor"; }
    virtual cl_ulong get_global_mem_cache_size() const { return 0; }
    virtual cl_ulong get_num_compute_units() const { return 1; }

    virtual cl_uint get_max_cmd_batch_size() const {
        return config.max_cmd_batch_size();
    }
    virtual cl_uint get_max_first_cmd_batch_size() const {
        return config.max_first_cmd_batch_size();
    }
    virtual cl_uint get_max_cmd_group_size() const {
        return config.max_cmd_group_size();
    }
    virtual cl_uint get_max_first_cmd_group_size() const {
        return config.max_first_cmd_group_size();
    }

    virtual std::string get_spirv_arch() const { return config.spirv_arch(); }
    virtual bool get_physical_addressing() const {
        return config.physical_addressing();
    }

    virtual std::string get_compile_options() const { return ""; }

    virtual const std::set<std::string> get_native_builtins() const {
        return std::set<std::string>();
    }

    virtual uint32_t get_preferred_subgroup_size() const {
        return config.preferred_subgroup_size();
    }

    virtual bool is_non_uniform_decoration_broken() const { return false; }

    virtual bool is_bgra_format_not_supported_for_image1d_buffer() const {
        return false;
    }

    using image_format_set =
        std::unordered_set<cl_image_format, ClFormatHash, ClFormatEqual>;
    virtual const image_format_set& get_disabled_image_formats() const {
        static image_format_set no_disabled_formats{};
        return no_disabled_formats;
    }

    virtual ~cvk_device_properties() {}
};

std::unique_ptr<cvk_device_properties> create_cvk_device_properties(
    const char* name, const uint32_t vendorID, const uint32_t deviceID,
    const uint32_t driverVersion, const VkDriverId driverId);
