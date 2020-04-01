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

#include "kernel.hpp"
#include "memory.hpp"

std::unique_ptr<cvk_buffer> cvk_kernel::allocate_pod_buffer() {
    cl_int err;
    auto buffer = cvk_buffer::create(
        m_context, 0, m_entry_point->pod_buffer_size(), nullptr, &err);
    if (err != CL_SUCCESS) {
        return nullptr;
    }

    return buffer;
}

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

    // Init POD arguments
    if (has_pod_arguments()) {

        // Find out POD binding
        for (auto& arg : m_args) {
            if (arg.is_pod()) {
                m_pod_arg = &arg;
                break;
            }
        }

        if (m_pod_arg == nullptr) {
            return CL_INVALID_PROGRAM;
        }
    }

    // Init argument values
    m_argument_values = cvk_kernel_argument_values::create(
        this, m_entry_point->num_resources());
    if (m_argument_values == nullptr) {
        return CL_OUT_OF_RESOURCES;
    }

    return CL_SUCCESS;
}

bool cvk_kernel::setup_descriptor_sets(
    VkDescriptorSet* ds,
    std::unique_ptr<cvk_kernel_argument_values>& arg_values) {

    if (!m_entry_point->allocate_descriptor_sets(ds)) {
        return false;
    }

    auto dev = m_context->device()->vulkan_device();

    // Transfer ownership of the argument values to the command
    arg_values = std::move(m_argument_values);
    arg_values->retain_resources();

    // Create a new set, copy the argument values
    m_argument_values = cvk_kernel_argument_values::create(*arg_values.get());
    if (m_argument_values == nullptr) {
        return false;
    }

    // Setup descriptors for POD arguments
    if (has_pod_arguments()) {

        // Update desciptors
        VkDescriptorBufferInfo bufferInfo = {arg_values->pod_vulkan_buffer(),
                                             0, // offset
                                             VK_WHOLE_SIZE};

        VkWriteDescriptorSet writeDescriptorSet = {
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            ds[m_pod_arg->descriptorSet],
            m_pod_arg->binding,                   // dstBinding
            0,                                    // dstArrayElement
            1,                                    // descriptorCount
            m_entry_point->pod_descriptor_type(), // descriptorType
            nullptr,                              // pImageInfo
            &bufferInfo,
            nullptr, // pTexelBufferView
        };
        vkUpdateDescriptorSets(dev, 1, &writeDescriptorSet, 0, nullptr);
    }

    // Setup other kernel argument descriptors
    for (cl_uint i = 0; i < m_args.size(); i++) {
        auto const& arg = m_args[i];

        switch (arg.kind) {

        case kernel_argument_kind::buffer:
        case kernel_argument_kind::buffer_ubo: {
            auto buffer =
                static_cast<cvk_buffer*>(arg_values->get_arg_value(arg));
            auto vkbuf = buffer->vulkan_buffer();
            cvk_debug_fn("buffer = %p", buffer);
            VkDescriptorBufferInfo bufferInfo = {vkbuf,
                                                 buffer->vulkan_buffer_offset(), // offset
                                                 buffer->size()};

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
                &bufferInfo,
                nullptr, // pTexelBufferView
            };
            vkUpdateDescriptorSets(dev, 1, &writeDescriptorSet, 0, nullptr);
            break;
        }
        case kernel_argument_kind::sampler: {
            auto clsampler =
                static_cast<cvk_sampler*>(arg_values->get_arg_value(arg));
            auto sampler = clsampler->vulkan_sampler();

            VkDescriptorImageInfo imageInfo = {
                sampler,
                VK_NULL_HANDLE,           // imageView
                VK_IMAGE_LAYOUT_UNDEFINED // imageLayout
            };

            VkWriteDescriptorSet writeDescriptorSet = {
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr,
                ds[arg.descriptorSet],
                arg.binding, // dstBinding
                0,           // dstArrayElement
                1,           // descriptorCount
                VK_DESCRIPTOR_TYPE_SAMPLER,
                &imageInfo, // pImageInfo
                nullptr,    // pBufferInfo
                nullptr,    // pTexelBufferView
            };
            vkUpdateDescriptorSets(dev, 1, &writeDescriptorSet, 0, nullptr);
            break;
        }
        case kernel_argument_kind::ro_image:
        case kernel_argument_kind::wo_image: {
            auto image =
                static_cast<cvk_image*>(arg_values->get_arg_value(arg));

            VkDescriptorImageInfo imageInfo = {
                VK_NULL_HANDLE,
                image->vulkan_image_view(), // imageView
                VK_IMAGE_LAYOUT_GENERAL     // imageLayout
            };

            VkDescriptorType dtype = arg.kind == kernel_argument_kind::ro_image
                                         ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
                                         : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;

            VkWriteDescriptorSet writeDescriptorSet = {
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr,
                ds[arg.descriptorSet],
                arg.binding, // dstBinding
                0,           // dstArrayElement
                1,           // descriptorCount
                dtype,
                &imageInfo, // pImageInfo
                nullptr,    // pBufferInfo
                nullptr,    // pTexelBufferView
            };
            vkUpdateDescriptorSets(dev, 1, &writeDescriptorSet, 0, nullptr);
            break;
        }
        case kernel_argument_kind::pod: // skip POD arguments
        case kernel_argument_kind::pod_ubo:
            break;
        case kernel_argument_kind::local: // nothing to do?
            break;
        default:
            cvk_error_fn("unsupported argument type");
            return false;
        }
    }

    // Setup literal samplers
    for (size_t i = 0; i < program()->literal_sampler_descs().size(); i++) {
        auto desc = program()->literal_sampler_descs()[i];
        auto clsampler = icd_downcast(program()->literal_samplers()[i]);
        auto sampler = clsampler->vulkan_sampler();

        VkDescriptorImageInfo imageInfo = {
            sampler,
            VK_NULL_HANDLE,           // imageView
            VK_IMAGE_LAYOUT_UNDEFINED // imageLayout
        };

        VkWriteDescriptorSet writeDescriptorSet = {
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            ds[desc.descriptorSet],
            desc.binding, // dstBinding
            0,            // dstArrayElement
            1,            // descriptorCount
            VK_DESCRIPTOR_TYPE_SAMPLER,
            &imageInfo, // pImageInfo
            nullptr,    // pBufferInfo
            nullptr,    // pTexelBufferView
        };
        vkUpdateDescriptorSets(dev, 1, &writeDescriptorSet, 0, nullptr);
    }

    return true;
}

VkPipeline
cvk_kernel::create_pipeline(const cvk_spec_constant_map& spec_constants) {
    return m_entry_point->create_pipeline(spec_constants);
}

cl_int cvk_kernel::set_arg(cl_uint index, size_t size, const void* value) {
    std::lock_guard<std::mutex> lock(m_lock);

    auto const& arg = m_args[index];

    return m_argument_values->set_arg(arg, size, value);
}
