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

    // Mark all arguments as unset
    m_args_set.resize(m_args.size(), false);

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

    if (ret == CL_SUCCESS) {
        // Mark argument as set
        m_args_set[index] = true;
    }

    return ret;
}

bool cvk_kernel_argument_values::setup_descriptor_sets() {
    std::lock_guard<std::mutex> lock(m_lock);

    auto program = m_entry_point->program();
    auto dev = program->context()->device()->vulkan_device();

    // Do nothing if these argument values have already been used in an enqueue
    if (m_is_enqueued) {
        return true;
    }

    m_is_enqueued = true;

    // Allocate descriptor sets
    if (!m_entry_point->allocate_descriptor_sets(descriptor_sets())) {
        return false;
    }
    VkDescriptorSet* ds = descriptor_sets();

    // Make enough space to store all descriptor write structures
    size_t max_descriptor_writes =
        m_args.size() + program->literal_sampler_descs().size();
    std::vector<VkWriteDescriptorSet> descriptor_writes;
    std::vector<VkDescriptorBufferInfo> buffer_info;
    std::vector<VkDescriptorImageInfo> image_info;
    descriptor_writes.reserve(max_descriptor_writes);
    buffer_info.reserve(max_descriptor_writes);
    image_info.reserve(max_descriptor_writes);

    // Setup descriptors for POD arguments
    if (m_entry_point->has_pod_buffer_arguments()) {
        // Create POD buffer
        if (!create_pod_buffer()) {
            return false;
        }

        // Update descriptors
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
            auto vkbuf = buffer->vulkan_buffer();
            cvk_debug_fn("buffer = %p", buffer);
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
        case kernel_argument_kind::ro_image:
        case kernel_argument_kind::wo_image: {
            auto image = static_cast<cvk_image*>(get_arg_value(arg));

            VkDescriptorImageInfo imageInfo = {
                VK_NULL_HANDLE,
                image->vulkan_image_view(), // imageView
                VK_IMAGE_LAYOUT_GENERAL     // imageLayout
            };
            image_info.push_back(imageInfo);

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
                &image_info.back(), // pImageInfo
                nullptr,            // pBufferInfo
                nullptr,            // pTexelBufferView
            };
            descriptor_writes.push_back(writeDescriptorSet);
            break;
        }
        case kernel_argument_kind::pod: // skip POD arguments
        case kernel_argument_kind::pod_ubo:
        case kernel_argument_kind::pod_pushconstant:
            break;
        case kernel_argument_kind::local: // nothing to do?
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

    // Write descriptors to device
    vkUpdateDescriptorSets(dev, static_cast<uint32_t>(descriptor_writes.size()),
                           descriptor_writes.data(), 0, nullptr);

    return true;
}
