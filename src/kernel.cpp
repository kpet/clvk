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

#include <algorithm>

#include "clspv/Sampler.h"

#include "kernel.hpp"
#include "memory.hpp"

cl_ulong cvk_kernel::local_mem_size() const {
    cl_ulong ret = 0; // FIXME take the compile-time allocations into account

    for (uint32_t i = 0; i < m_args.size(); i++) {
        auto const& arg = m_args[i];
        if (arg.kind == kernel_argument_kind::local) {
            ret += m_argument_values->local_arg_size(i);
        }
    }

    return ret;
}

cl_int cvk_kernel::init() {
    cl_int errcode;
    m_entry_point = m_program->get_entry_point(m_name, &errcode);
    if (!m_entry_point) {
        return errcode;
    }

    // Store a copy of the arguments
    m_args = m_entry_point->args();

    if (const auto* md = m_entry_point->image_metadata()) {
        m_image_metadata = md;
    }

    if (const auto* md = m_entry_point->sampler_metadata()) {
        m_sampler_metadata = md;
    }

    // Init argument values
    m_argument_values = cvk_kernel_argument_values::create(m_entry_point);
    if (m_argument_values == nullptr) {
        return CL_OUT_OF_RESOURCES;
    }

    return CL_SUCCESS;
}

VkPipeline
cvk_kernel::create_pipeline(const cvk_spec_constant_map& spec_constants) {
    return m_entry_point->create_pipeline(spec_constants);
}

std::unique_ptr<cvk_kernel> cvk_kernel::clone(cl_int* errcode_ret) const {

    auto kernel = std::make_unique<cvk_kernel>(m_program, m_name.c_str());

    *errcode_ret = kernel->init();

    if (*errcode_ret != CL_SUCCESS) {
        return nullptr;
    }

    kernel->m_argument_values =
        cvk_kernel_argument_values::create(*m_argument_values.get());

    return kernel;
}

void cvk_kernel::set_image_metadata(cl_uint index, const void* image) {
    if (!m_image_metadata) {
        return;
    }
    auto md = m_image_metadata->find(index);
    if (md != m_image_metadata->end()) {

        auto mem = icd_downcast(*reinterpret_cast<const cl_mem*>(image));
        assert(mem->is_image_type());
        auto format = static_cast<cvk_image*>(mem)->format();

        if (md->second.has_valid_order()) {
            auto order_offset = md->second.order_offset;
            auto order = format.image_channel_order;
            m_argument_values->set_pod_data(order_offset, sizeof(order),
                                            &order);
        }
        if (md->second.has_valid_data_type()) {
            auto data_type_offset = md->second.data_type_offset;
            auto data_type = format.image_channel_data_type;
            m_argument_values->set_pod_data(data_type_offset, sizeof(data_type),
                                            &data_type);
        }
    }
}

void cvk_kernel::set_sampler_metadata(cl_uint index, const void* sampler) {
    if (!m_sampler_metadata) {
        return;
    }
    auto md = m_sampler_metadata->find(index);
    if (md != m_sampler_metadata->end()) {
        auto apisampler = *reinterpret_cast<const cl_sampler*>(sampler);
        auto offset = md->second;
        auto sampler = icd_downcast(apisampler);
        uint32_t sampler_mask = (sampler->normalized_coords()
                                     ? clspv::CLK_NORMALIZED_COORDS_TRUE
                                     : clspv::CLK_NORMALIZED_COORDS_FALSE) |
                                (sampler->filter_mode() == CL_FILTER_NEAREST
                                     ? clspv::CLK_FILTER_NEAREST
                                     : clspv::CLK_FILTER_LINEAR);
        switch (sampler->addressing_mode()) {
        case CL_ADDRESS_NONE:
            sampler_mask |= clspv::CLK_ADDRESS_NONE;
            break;
        case CL_ADDRESS_CLAMP:
            sampler_mask |= clspv::CLK_ADDRESS_CLAMP;
            break;
        case CL_ADDRESS_REPEAT:
            sampler_mask |= clspv::CLK_ADDRESS_REPEAT;
            break;
        case CL_ADDRESS_CLAMP_TO_EDGE:
            sampler_mask |= clspv::CLK_ADDRESS_CLAMP_TO_EDGE;
            break;
        case CL_ADDRESS_MIRRORED_REPEAT:
            sampler_mask |= clspv::CLK_ADDRESS_MIRRORED_REPEAT;
            break;
        default:
            break;
        }
        m_argument_values->set_pod_data(offset, sizeof(sampler_mask),
                                        &sampler_mask);
    }
}

cl_int cvk_kernel::set_arg(cl_uint index, size_t size, const void* value) {
    std::lock_guard<std::mutex> lock(m_lock);

    // Clone argument values if they have been used in an enqueue
    if (m_argument_values->is_enqueued()) {
        m_argument_values =
            cvk_kernel_argument_values::create(*m_argument_values);
        if (m_argument_values == nullptr) {
            return CL_OUT_OF_RESOURCES;
        }
    }

    auto const& arg = m_args[index];

    cl_int ret = m_argument_values->set_arg(arg, size, value);

    // if the argument is an image, we need to set its metadata
    // (channel_order/channel_data_type).
    if (arg.kind == kernel_argument_kind::sampled_image ||
        arg.kind == kernel_argument_kind::storage_image ||
        arg.kind == kernel_argument_kind::storage_texel_buffer ||
        arg.kind == kernel_argument_kind::uniform_texel_buffer) {
        set_image_metadata(index, value);
    }

    if (arg.kind == kernel_argument_kind::sampler) {
        set_sampler_metadata(index, value);
    }

    return ret;
}

bool cvk_kernel::args_valid() const { return m_argument_values->args_valid(); }

bool cvk_kernel_argument_values::setup_descriptor_sets() {
    std::lock_guard<std::mutex> lock(m_lock);

    auto program = m_entry_point->program();
    auto dev = program->context()->device()->vulkan_device();

    // Do nothing if these argument values have already been used in an enqueue
    if (m_is_enqueued) {
        return true;
    }

    // Allocate descriptor sets
    if (!m_entry_point->allocate_descriptor_sets(descriptor_sets())) {
        return false;
    }
    VkDescriptorSet* ds = descriptor_sets();

    // Make enough space to store all descriptor write structures
    size_t max_descriptor_writes =
        m_args.size() // upper bound that includes POD buffers
        + program->literal_sampler_descs().size() +
        1; // module constant data buffer
    std::vector<VkWriteDescriptorSet> descriptor_writes;
    std::vector<VkDescriptorBufferInfo> buffer_info;
    std::vector<VkDescriptorImageInfo> image_info;
    std::vector<VkBufferView> buffer_views;
    descriptor_writes.reserve(max_descriptor_writes);
    buffer_info.reserve(max_descriptor_writes);
    image_info.reserve(max_descriptor_writes);
    buffer_views.reserve(max_descriptor_writes);

    // Setup module-scope variables
    if (program->module_constant_data_buffer() != nullptr &&
        program->module_constant_data_buffer_info()->type ==
            module_buffer_type::storage_buffer) {
        auto buffer = program->module_constant_data_buffer();
        auto info = program->module_constant_data_buffer_info();
        cvk_debug_fn(
            "constant data buffer %p, size = %zu @ set = %u, binding = %u",
            buffer->vulkan_buffer(), buffer->size(), info->set, info->binding);
        // Update descriptors
        VkDescriptorBufferInfo bufferInfo = {buffer->vulkan_buffer(),
                                             0, // offset
                                             VK_WHOLE_SIZE};
        buffer_info.push_back(bufferInfo);

        VkWriteDescriptorSet writeDescriptorSet = {
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            ds[info->set],
            info->binding,                     // dstBinding
            0,                                 // dstArrayElement
            1,                                 // descriptorCount
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // descriptorType
            nullptr,                           // pImageInfo
            &buffer_info.back(),
            nullptr, // pTexelBufferView
        };
        descriptor_writes.push_back(writeDescriptorSet);
    }

    // Setup descriptors for POD arguments
    if (m_entry_point->has_pod_buffer_arguments()) {
        // Create POD buffer
        if (!create_pod_buffer()) {
            return false;
        }

        // Update descriptors
        cvk_debug_fn("pod buffer %p, size = %zu @ set = %u, binding = %u",
                     m_pod_buffer->vulkan_buffer(), m_pod_buffer->size(),
                     m_pod_arg->descriptorSet, m_pod_arg->binding);
        VkDescriptorBufferInfo bufferInfo = {m_pod_buffer->vulkan_buffer(),
                                             0, // offset
                                             VK_WHOLE_SIZE};
        buffer_info.push_back(bufferInfo);

        VkWriteDescriptorSet writeDescriptorSet = {
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            ds[m_pod_arg->descriptorSet],
            m_pod_arg->binding,                   // dstBinding
            0,                                    // dstArrayElement
            1,                                    // descriptorCount
            m_entry_point->pod_descriptor_type(), // descriptorType
            nullptr,                              // pImageInfo
            &buffer_info.back(),
            nullptr, // pTexelBufferView
        };
        descriptor_writes.push_back(writeDescriptorSet);
    }

    // Setup other kernel argument descriptors
    for (cl_uint i = 0; i < m_args.size(); i++) {
        auto const& arg = m_args[i];

        switch (arg.kind) {

        case kernel_argument_kind::buffer:
        case kernel_argument_kind::buffer_ubo: {
            auto buffer = static_cast<cvk_buffer*>(get_arg_value(arg));
            if (buffer == nullptr) {
                cvk_debug_fn("ignoring NULL buffer argument");
                break;
            }
            auto vkbuf = buffer->vulkan_buffer();
            cvk_debug_fn(
                "buffer %p, offset = %zu, size = %zu @ set = %u, binding = %u",
                buffer->vulkan_buffer(), buffer->vulkan_buffer_offset(),
                buffer->size(), arg.descriptorSet, arg.binding);
            VkDescriptorBufferInfo bufferInfo = {
                vkbuf,
                buffer->vulkan_buffer_offset(), // offset
                buffer->size()};
            buffer_info.push_back(bufferInfo);

            auto descriptor_type = arg.kind == kernel_argument_kind::buffer
                                       ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                                       : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            VkWriteDescriptorSet writeDescriptorSet = {
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr,
                ds[arg.descriptorSet],
                arg.binding, // dstBinding
                0,           // dstArrayElement
                1,           // descriptorCount
                descriptor_type,
                nullptr, // pImageInfo
                &buffer_info.back(),
                nullptr, // pTexelBufferView
            };
            descriptor_writes.push_back(writeDescriptorSet);
            break;
        }
        case kernel_argument_kind::sampler: {
            auto clsampler = static_cast<cvk_sampler*>(get_arg_value(arg));
            bool normalized_coord_sampler_required = false;
            if (auto md = m_entry_point->sampler_metadata()) {
                normalized_coord_sampler_required = md->find(i) != md->end();
            }
            auto sampler =
                normalized_coord_sampler_required &&
                        !clsampler->normalized_coords()
                    ? clsampler
                          ->get_or_create_vulkan_sampler_with_normalized_coords()
                    : clsampler->vulkan_sampler();
            if (sampler == VK_NULL_HANDLE) {
                cvk_error_fn("Could not set descriptor for sampler");
                return false;
            }

            cvk_debug_fn("sampler %p @ set = %u, binding = %u", sampler,
                         arg.descriptorSet, arg.binding);
            VkDescriptorImageInfo imageInfo = {
                sampler,
                VK_NULL_HANDLE,           // imageView
                VK_IMAGE_LAYOUT_UNDEFINED // imageLayout
            };
            image_info.push_back(imageInfo);

            VkWriteDescriptorSet writeDescriptorSet = {
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr,
                ds[arg.descriptorSet],
                arg.binding, // dstBinding
                0,           // dstArrayElement
                1,           // descriptorCount
                VK_DESCRIPTOR_TYPE_SAMPLER,
                &image_info.back(), // pImageInfo
                nullptr,            // pBufferInfo
                nullptr,            // pTexelBufferView
            };
            descriptor_writes.push_back(writeDescriptorSet);
            break;
        }
        case kernel_argument_kind::sampled_image:
        case kernel_argument_kind::storage_image: {
            auto image = static_cast<cvk_image*>(get_arg_value(arg));
            bool sampled = arg.kind == kernel_argument_kind::sampled_image;
            auto view = sampled ? image->vulkan_sampled_view()
                                : image->vulkan_storage_view();

            cvk_debug_fn("image view %p @ set = %u, binding = %u", view,
                         arg.descriptorSet, arg.binding);
            VkDescriptorImageInfo imageInfo = {
                VK_NULL_HANDLE,
                view,                   // imageView
                VK_IMAGE_LAYOUT_GENERAL // imageLayout
            };
            image_info.push_back(imageInfo);

            VkDescriptorType dtype = sampled ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
                                             : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;

            VkWriteDescriptorSet writeDescriptorSet = {
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr,
                ds[arg.descriptorSet],
                arg.binding, // dstBinding
                0,           // dstArrayElement
                1,           // descriptorCount
                dtype,
                &image_info.back(), // pImageInfo
                nullptr,            // pBufferInfo
                nullptr,            // pTexelBufferView
            };
            descriptor_writes.push_back(writeDescriptorSet);
            break;
        }
        case kernel_argument_kind::storage_texel_buffer:
        case kernel_argument_kind::uniform_texel_buffer: {
            auto image = static_cast<cvk_image*>(get_arg_value(arg));
            bool uniform =
                arg.kind == kernel_argument_kind::uniform_texel_buffer;
            auto view = image->vulkan_buffer_view();
            buffer_views.push_back(view);

            cvk_debug_fn("buffer view %p @ set = %u, binding = %u", view,
                         arg.descriptorSet, arg.binding);

            VkDescriptorType dtype =
                uniform ? VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER
                        : VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;

            VkWriteDescriptorSet writeDescriptorSet = {
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr,
                ds[arg.descriptorSet],
                arg.binding, // dstBinding
                0,           // dstArrayElement
                1,           // descriptorCount
                dtype,
                nullptr,              // pImageInfo
                nullptr,              // pBufferInfo
                &buffer_views.back(), // pTexelBufferView
            };
            descriptor_writes.push_back(writeDescriptorSet);
            break;
        }
        case kernel_argument_kind::pod: // skip POD arguments
        case kernel_argument_kind::pod_ubo:
        case kernel_argument_kind::pod_pushconstant:
        case kernel_argument_kind::pointer_ubo:
        case kernel_argument_kind::pointer_pushconstant:
            break;
        case kernel_argument_kind::local: // nothing to do?
            break;
        case kernel_argument_kind::unused:
            break;
        default:
            cvk_error_fn("unsupported argument type");
            return false;
        }
    }

    // Setup literal samplers
    for (size_t i = 0; i < program->literal_sampler_descs().size(); i++) {
        auto desc = program->literal_sampler_descs()[i];
        auto clsampler = icd_downcast(program->literal_samplers()[i]);
        auto sampler = clsampler->vulkan_sampler();

        VkDescriptorImageInfo imageInfo = {
            sampler,
            VK_NULL_HANDLE,           // imageView
            VK_IMAGE_LAYOUT_UNDEFINED // imageLayout
        };
        image_info.push_back(imageInfo);

        VkWriteDescriptorSet writeDescriptorSet = {
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            ds[desc.descriptorSet],
            desc.binding, // dstBinding
            0,            // dstArrayElement
            1,            // descriptorCount
            VK_DESCRIPTOR_TYPE_SAMPLER,
            &image_info.back(), // pImageInfo
            nullptr,            // pBufferInfo
            nullptr,            // pTexelBufferView
        };
        descriptor_writes.push_back(writeDescriptorSet);
    }

    m_is_enqueued = true;

    // Write descriptors to device
    vkUpdateDescriptorSets(dev, static_cast<uint32_t>(descriptor_writes.size()),
                           descriptor_writes.data(), 0, nullptr);

    return true;
}
