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

#pragma once

#include "cl_headers.hpp"

#include <unordered_map>

#include <vulkan/vulkan.h>

struct cvk_device;

struct ClFormatHash {
    size_t operator()(const cl_image_format& format) const {
        return format.image_channel_order << 16 |
               format.image_channel_data_type;
    }
};

struct ClFormatEqual {
    bool operator()(const cl_image_format& lhs,
                    const cl_image_format& rhs) const {
        return lhs.image_channel_order == rhs.image_channel_order &&
               lhs.image_channel_data_type == rhs.image_channel_data_type;
    }
};

struct image_format_support {
    static constexpr cl_mem_flags RO = CL_MEM_READ_ONLY;
    static constexpr cl_mem_flags WO = CL_MEM_WRITE_ONLY;
    static constexpr cl_mem_flags RW = CL_MEM_KERNEL_READ_AND_WRITE;
    static constexpr cl_mem_flags ROWO = RO | WO | CL_MEM_READ_WRITE;
    static constexpr cl_mem_flags ALL = ROWO | RW;

    image_format_support(cl_mem_flags flags, VkFormat fmt)
        : flags(flags), vkfmt(fmt) {}
    image_format_support(VkFormat fmt) : flags(ALL), vkfmt(fmt) {}
    image_format_support() {}

    cl_mem_flags flags;
    VkFormat vkfmt;
};

using format_mapping_map =
    std::unordered_map<cl_image_format, image_format_support, ClFormatHash,
                       ClFormatEqual>;

const format_mapping_map& get_format_maps();

bool cl_image_format_to_vulkan_format(cl_image_format clformat,
                                      cl_mem_object_type image_type,
                                      cvk_device* device,
                                      image_format_support* fmt_support,
                                      VkComponentMapping* components_sampled,
                                      VkComponentMapping* components_storage);
