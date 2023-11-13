// Copyright 2023 The clvk authors.
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

#include "image_format.hpp"
#include "CL/cl.h"
#include <unordered_map>

format_mapping_map FormatMaps = {
    // R formats
    {{CL_R, CL_UNORM_INT8}, VK_FORMAT_R8_UNORM},
    {{CL_R, CL_SNORM_INT8}, VK_FORMAT_R8_SNORM},
    {{CL_R, CL_UNSIGNED_INT8}, VK_FORMAT_R8_UINT},
    {{CL_R, CL_SIGNED_INT8}, VK_FORMAT_R8_SINT},
    {{CL_R, CL_UNORM_INT16}, VK_FORMAT_R16_UNORM},
    {{CL_R, CL_SNORM_INT16}, VK_FORMAT_R16_SNORM},
    {{CL_R, CL_UNSIGNED_INT16}, VK_FORMAT_R16_UINT},
    {{CL_R, CL_SIGNED_INT16}, VK_FORMAT_R16_SINT},
    {{CL_R, CL_HALF_FLOAT}, VK_FORMAT_R16_SFLOAT},
    {{CL_R, CL_UNSIGNED_INT32}, VK_FORMAT_R32_UINT},
    {{CL_R, CL_SIGNED_INT32}, VK_FORMAT_R32_SINT},
    {{CL_R, CL_FLOAT}, VK_FORMAT_R32_SFLOAT},

    // LUMINANCE formats
    {{CL_LUMINANCE, CL_UNORM_INT8},
     {image_format_support::ROWO, VK_FORMAT_R8_UNORM}},
    {{CL_LUMINANCE, CL_SNORM_INT8},
     {image_format_support::ROWO, VK_FORMAT_R8_SNORM}},
    {{CL_LUMINANCE, CL_UNSIGNED_INT8},
     {image_format_support::ROWO, VK_FORMAT_R8_UINT}},
    {{CL_LUMINANCE, CL_SIGNED_INT8},
     {image_format_support::ROWO, VK_FORMAT_R8_SINT}},
    {{CL_LUMINANCE, CL_UNORM_INT16},
     {image_format_support::ROWO, VK_FORMAT_R16_UNORM}},
    {{CL_LUMINANCE, CL_SNORM_INT16},
     {image_format_support::ROWO, VK_FORMAT_R16_SNORM}},
    {{CL_LUMINANCE, CL_UNSIGNED_INT16},
     {image_format_support::ROWO, VK_FORMAT_R16_UINT}},
    {{CL_LUMINANCE, CL_SIGNED_INT16},
     {image_format_support::ROWO, VK_FORMAT_R16_SINT}},
    {{CL_LUMINANCE, CL_HALF_FLOAT},
     {image_format_support::ROWO, VK_FORMAT_R16_SFLOAT}},
    {{CL_LUMINANCE, CL_UNSIGNED_INT32},
     {image_format_support::ROWO, VK_FORMAT_R32_UINT}},
    {{CL_LUMINANCE, CL_SIGNED_INT32},
     {image_format_support::ROWO, VK_FORMAT_R32_SINT}},
    {{CL_LUMINANCE, CL_FLOAT},
     {image_format_support::ROWO, VK_FORMAT_R32_SFLOAT}},

    // INTENSITY formats
    {{CL_INTENSITY, CL_UNORM_INT8},
     {image_format_support::ROWO, VK_FORMAT_R8_UNORM}},
    {{CL_INTENSITY, CL_SNORM_INT8},
     {image_format_support::ROWO, VK_FORMAT_R8_SNORM}},
    {{CL_INTENSITY, CL_UNSIGNED_INT8},
     {image_format_support::ROWO, VK_FORMAT_R8_UINT}},
    {{CL_INTENSITY, CL_SIGNED_INT8},
     {image_format_support::ROWO, VK_FORMAT_R8_SINT}},
    {{CL_INTENSITY, CL_UNORM_INT16},
     {image_format_support::ROWO, VK_FORMAT_R16_UNORM}},
    {{CL_INTENSITY, CL_SNORM_INT16},
     {image_format_support::ROWO, VK_FORMAT_R16_SNORM}},
    {{CL_INTENSITY, CL_UNSIGNED_INT16},
     {image_format_support::ROWO, VK_FORMAT_R16_UINT}},
    {{CL_INTENSITY, CL_SIGNED_INT16},
     {image_format_support::ROWO, VK_FORMAT_R16_SINT}},
    {{CL_INTENSITY, CL_HALF_FLOAT},
     {image_format_support::ROWO, VK_FORMAT_R16_SFLOAT}},
    {{CL_INTENSITY, CL_UNSIGNED_INT32},
     {image_format_support::ROWO, VK_FORMAT_R32_UINT}},
    {{CL_INTENSITY, CL_SIGNED_INT32},
     {image_format_support::ROWO, VK_FORMAT_R32_SINT}},
    {{CL_INTENSITY, CL_FLOAT},
     {image_format_support::ROWO, VK_FORMAT_R32_SFLOAT}},

    // RG formats
    {{CL_RG, CL_UNORM_INT8}, VK_FORMAT_R8G8_UNORM},
    {{CL_RG, CL_SNORM_INT8}, VK_FORMAT_R8G8_SNORM},
    {{CL_RG, CL_UNSIGNED_INT8}, VK_FORMAT_R8G8_UINT},
    {{CL_RG, CL_SIGNED_INT8}, VK_FORMAT_R8G8_SINT},
    {{CL_RG, CL_UNORM_INT16}, VK_FORMAT_R16G16_UNORM},
    {{CL_RG, CL_SNORM_INT16}, VK_FORMAT_R16G16_SNORM},
    {{CL_RG, CL_UNSIGNED_INT16}, VK_FORMAT_R16G16_UINT},
    {{CL_RG, CL_SIGNED_INT16}, VK_FORMAT_R16G16_SINT},
    {{CL_RG, CL_HALF_FLOAT}, VK_FORMAT_R16G16_SFLOAT},
    {{CL_RG, CL_UNSIGNED_INT32}, VK_FORMAT_R32G32_UINT},
    {{CL_RG, CL_SIGNED_INT32}, VK_FORMAT_R32G32_SINT},
    {{CL_RG, CL_FLOAT}, VK_FORMAT_R32G32_SFLOAT},

    // RGB formats
    {{CL_RGB, CL_UNORM_INT8}, VK_FORMAT_R8G8B8_UNORM},
    {{CL_RGB, CL_SNORM_INT8}, VK_FORMAT_R8G8B8_SNORM},
    {{CL_RGB, CL_UNSIGNED_INT8}, VK_FORMAT_R8G8B8_UINT},
    {{CL_RGB, CL_SIGNED_INT8}, VK_FORMAT_R8G8B8_SINT},
    {{CL_RGB, CL_UNORM_INT16}, VK_FORMAT_R16G16B16_UNORM},
    {{CL_RGB, CL_SNORM_INT16}, VK_FORMAT_R16G16B16_SNORM},
    {{CL_RGB, CL_UNSIGNED_INT16}, VK_FORMAT_R16G16B16_UINT},
    {{CL_RGB, CL_SIGNED_INT16}, VK_FORMAT_R16G16B16_SINT},
    {{CL_RGB, CL_HALF_FLOAT}, VK_FORMAT_R16G16B16_SFLOAT},
    {{CL_RGB, CL_UNSIGNED_INT32}, VK_FORMAT_R32G32B32_UINT},
    {{CL_RGB, CL_SIGNED_INT32}, VK_FORMAT_R32G32B32_SINT},
    {{CL_RGB, CL_FLOAT}, VK_FORMAT_R32G32B32_SFLOAT},
    {{CL_RGB, CL_UNORM_SHORT_565}, VK_FORMAT_R5G6B5_UNORM_PACK16},

    // RGBA formats
    {{CL_RGBA, CL_UNORM_INT8}, VK_FORMAT_R8G8B8A8_UNORM},
    {{CL_RGBA, CL_SNORM_INT8}, VK_FORMAT_R8G8B8A8_SNORM},
    {{CL_RGBA, CL_UNSIGNED_INT8}, VK_FORMAT_R8G8B8A8_UINT},
    {{CL_RGBA, CL_SIGNED_INT8}, VK_FORMAT_R8G8B8A8_SINT},
    {{CL_RGBA, CL_UNORM_INT16}, VK_FORMAT_R16G16B16A16_UNORM},
    {{CL_RGBA, CL_SNORM_INT16}, VK_FORMAT_R16G16B16A16_SNORM},
    {{CL_RGBA, CL_UNSIGNED_INT16}, VK_FORMAT_R16G16B16A16_UINT},
    {{CL_RGBA, CL_SIGNED_INT16}, VK_FORMAT_R16G16B16A16_SINT},
    {{CL_RGBA, CL_HALF_FLOAT}, VK_FORMAT_R16G16B16A16_SFLOAT},
    {{CL_RGBA, CL_UNSIGNED_INT32}, VK_FORMAT_R32G32B32A32_UINT},
    {{CL_RGBA, CL_SIGNED_INT32}, VK_FORMAT_R32G32B32A32_SINT},
    {{CL_RGBA, CL_FLOAT}, VK_FORMAT_R32G32B32A32_SFLOAT},

    // BGRA formats
    {{CL_BGRA, CL_UNORM_INT8}, VK_FORMAT_B8G8R8A8_UNORM},
    {{CL_BGRA, CL_SNORM_INT8}, VK_FORMAT_B8G8R8A8_SNORM},
    {{CL_BGRA, CL_UNSIGNED_INT8}, VK_FORMAT_B8G8R8A8_UINT},
    {{CL_BGRA, CL_SIGNED_INT8}, VK_FORMAT_B8G8R8A8_SINT},
};

const format_mapping_map& get_format_maps() { return FormatMaps; }

static bool get_equivalent_bgra_format_for_image1d_buffer(
    VkFormat& fmt, VkComponentMapping* components_sampled,
    VkComponentMapping* components_storage) {
    const std::unordered_map<VkFormat, VkFormat> map = {
        {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM},
    };
    const VkComponentMapping BGRA_mapping = {
        VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_IDENTITY};

    auto it = map.find(fmt);
    if (it != map.end()) {
        fmt = it->second;
        *components_sampled = *components_storage = BGRA_mapping;
        return true;
    }
    return false;
}

static void get_component_mappings_for_channel_order(
    cl_channel_order order, VkComponentMapping* components_sampled,
    VkComponentMapping* components_storage) {
    if (order == CL_LUMINANCE) {
        *components_sampled = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R,
                               VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_A};
    } else if (order == CL_INTENSITY) {
        *components_sampled = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R,
                               VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R};
    } else {
        *components_sampled = {
            VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
    }

    *components_storage = {
        VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
}

bool cl_image_format_to_vulkan_format(cl_image_format clformat,
                                      cl_mem_object_type image_type,
                                      cvk_device* device,
                                      image_format_support* fmt_support,
                                      VkComponentMapping* components_sampled,
                                      VkComponentMapping* components_storage) {
    auto m = FormatMaps.find(clformat);
    bool success = false;

    if (m != FormatMaps.end()) {
        *fmt_support = (*m).second;
        success = true;
    }

    get_component_mappings_for_channel_order(
        clformat.image_channel_order, components_sampled, components_storage);

    if (image_type == CL_MEM_OBJECT_IMAGE1D_BUFFER &&
        clformat.image_channel_order == CL_BGRA &&
        device->is_bgra_format_not_supported_for_image1d_buffer()) {
        get_equivalent_bgra_format_for_image1d_buffer(
            fmt_support->vkfmt, components_sampled, components_storage);
    }

    return success;
}
