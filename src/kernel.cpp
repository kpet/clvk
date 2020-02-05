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

bool cvk_kernel::build_descriptor_set_layout(
    VkDevice vkdev, const std::vector<VkDescriptorSetLayoutBinding>& bindings) {
    VkDescriptorSetLayoutCreateInfo createInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr,
        0,                                      // flags
        static_cast<uint32_t>(bindings.size()), // bindingCount
        bindings.data()                         // pBindings
    };

    VkResult res;
    if (bindings.size() > 0) {
        VkDescriptorSetLayout setLayout;
        res = vkCreateDescriptorSetLayout(vkdev, &createInfo, 0, &setLayout);
        if (res != VK_SUCCESS) {
            cvk_error("Could not create descriptor set layout");
            return false;
        }
        m_descriptor_set_layouts.push_back(setLayout);
    }

    return true;
}

bool cvk_kernel::build_descriptor_sets_layout_bindings_for_arguments(
    VkDevice vkdev, binding_stat_map& smap, uint32_t& num_resources) {
    bool pod_found = false;

    std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
    for (auto& arg : m_args) {
        VkDescriptorType dt = VK_DESCRIPTOR_TYPE_MAX_ENUM;

        switch (arg.kind) {
        case kernel_argument_kind::buffer:
            dt = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            break;
        case kernel_argument_kind::buffer_ubo:
            dt = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            break;
        case kernel_argument_kind::ro_image:
            dt = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            break;
        case kernel_argument_kind::wo_image:
            dt = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            break;
        case kernel_argument_kind::sampler:
            dt = VK_DESCRIPTOR_TYPE_SAMPLER;
            break;
        case kernel_argument_kind::local:
            continue;
        case kernel_argument_kind::pod:
        case kernel_argument_kind::pod_ubo:
            if (!pod_found) {
                if (arg.kind == kernel_argument_kind::pod) {
                    dt = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                } else if (arg.kind == kernel_argument_kind::pod_ubo) {
                    dt = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                }

                m_pod_descriptor_type = dt;

                pod_found = true;
            } else {
                continue;
            }
        }

        VkDescriptorSetLayoutBinding binding = {
            arg.binding,                 // binding
            dt,                          // descriptorType
            1,                           // decriptorCount
            VK_SHADER_STAGE_COMPUTE_BIT, // stageFlags
            nullptr                      // pImmutableSamplers
        };

        layoutBindings.push_back(binding);
        smap[binding.descriptorType]++;
    }

    num_resources = layoutBindings.size();

    if (!build_descriptor_set_layout(vkdev, layoutBindings)) {
        return false;
    }

    return true;
}

bool cvk_kernel::build_descriptor_sets_layout_bindings_for_literal_samplers(
    VkDevice vkdev, binding_stat_map& smap) {

    std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
    for (auto& desc : m_program->literal_sampler_descs()) {
        VkDescriptorSetLayoutBinding binding = {
            desc.binding,                // binding
            VK_DESCRIPTOR_TYPE_SAMPLER,  // descriptorType
            1,                           // decriptorCount
            VK_SHADER_STAGE_COMPUTE_BIT, // stageFlags
            nullptr                      // pImmutableSamplers
        };
        layoutBindings.push_back(binding);
        smap[binding.descriptorType]++;
    }

    if (!build_descriptor_set_layout(vkdev, layoutBindings)) {
        return false;
    }

    return true;
}

std::unique_ptr<cvk_buffer> cvk_kernel::allocate_pod_buffer() {
    cl_int err;
    auto buffer =
        cvk_buffer::create(m_context, 0, m_pod_buffer_size, nullptr, &err);
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
    // Get a pointer to the arguments from the program
    auto args = m_program->args_for_kernel(m_name);

    if (args == nullptr) {
        cvk_error("Kernel %s doesn't exist in program", m_name.c_str());
        return CL_INVALID_KERNEL_NAME;
    }

    // Store a sorted copy of the arguments
    m_args = *args;
    std::sort(
        m_args.begin(), m_args.end(),
        [](kernel_argument a, kernel_argument b) { return a.pos < b.pos; });

    // Create Descriptor Sets Layout
    VkResult res;
    auto vkdev = m_context->device()->vulkan_device();

    std::unordered_map<VkDescriptorType, uint32_t> bindingTypes;
    if (!build_descriptor_sets_layout_bindings_for_literal_samplers(
            vkdev, bindingTypes)) {
        return CL_INVALID_VALUE;
    }

    uint32_t num_resources;
    if (!build_descriptor_sets_layout_bindings_for_arguments(
            vkdev, bindingTypes, num_resources)) {
        return CL_INVALID_VALUE;
    }

    // Do we have POD arguments?
    for (auto& arg : m_args) {
        if (arg.is_pod()) {
            m_has_pod_arguments = true;
        }
    }

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

        // Check we know the POD buffer's descriptor type
        if (m_pod_descriptor_type == VK_DESCRIPTOR_TYPE_MAX_ENUM) {
            return CL_INVALID_PROGRAM;
        }

        // Find how big the POD buffer should be
        int max_offset = 0;
        int max_offset_arg_size = 0;

        for (auto& arg : m_args) {
            if (arg.is_pod() && (arg.offset >= max_offset)) {
                max_offset = arg.offset;
                max_offset_arg_size = arg.size;
            }
        }

        m_pod_buffer_size = max_offset + max_offset_arg_size;
    }

    // Init argument values
    m_argument_values = cvk_kernel_argument_values::create(this, num_resources);
    if (m_argument_values == nullptr) {
        return CL_OUT_OF_RESOURCES;
    }

    // Create pipeline layout
    cvk_debug("about to create pipeline layout, number of descriptor set "
              "layouts: %zu",
              m_descriptor_set_layouts.size());
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        0,
        0,
        static_cast<uint32_t>(m_descriptor_set_layouts.size()),
        m_descriptor_set_layouts.data(),
        0,
        0};

    res = vkCreatePipelineLayout(vkdev, &pipelineLayoutCreateInfo, 0,
                                 &m_pipeline_layout);
    if (res != VK_SUCCESS) {
        cvk_error("Could not create pipeline layout.");
        return CL_INVALID_VALUE;
    }

    // Determine number and types of bindings
    std::vector<VkDescriptorPoolSize> poolSizes(bindingTypes.size());

    int bidx = 0;
    for (auto& bt : bindingTypes) {
        poolSizes[bidx].type = bt.first;
        poolSizes[bidx].descriptorCount = bt.second * cvk_kernel::MAX_INSTANCES;
        bidx++;
    }

    // Create descriptor pool
    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        nullptr,
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,            // flags
        cvk_kernel::MAX_INSTANCES * spir_binary::MAX_DESCRIPTOR_SETS, // maxSets
        static_cast<uint32_t>(poolSizes.size()), // poolSizeCount
        poolSizes.data(),                        // pPoolSizes
    };

    res = vkCreateDescriptorPool(vkdev, &descriptorPoolCreateInfo, 0,
                                 &m_descriptor_pool);

    if (res != VK_SUCCESS) {
        cvk_error("Could not create descriptor pool.");
        return CL_INVALID_VALUE;
    }

    // Create pipeline cache
    VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
        nullptr, // pNext
        0,       // flags
        0,       // initialDataSize
        nullptr, // pInitialData
    };

    res = vkCreatePipelineCache(vkdev, &pipelineCacheCreateInfo, nullptr,
                                &m_pipeline_cache);
    if (res != VK_SUCCESS) {
        cvk_error("Could not create pipeline cache.");
        return CL_OUT_OF_RESOURCES;
    }

    return CL_SUCCESS;
}

bool cvk_kernel::setup_descriptor_sets(
    VkDescriptorSet* ds,
    std::unique_ptr<cvk_kernel_argument_values>& arg_values) {
    std::lock_guard<std::mutex> lock(m_lock);

    // Allocate descriptor sets
    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
        m_descriptor_pool,
        static_cast<uint32_t>(
            m_descriptor_set_layouts.size()), // descriptorSetCount
        m_descriptor_set_layouts.data()};

    auto dev = m_context->device()->vulkan_device();

    VkResult res =
        vkAllocateDescriptorSets(dev, &descriptorSetAllocateInfo, ds);

    if (res != VK_SUCCESS) {
        cvk_error_fn("could not allocate descriptor sets: %s",
                     vulkan_error_string(res));
        return false;
    }

    // Transfer ownership of the argument values to the command
    arg_values = std::move(m_argument_values);

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
            m_pod_arg->binding,    // dstBinding
            0,                     // dstArrayElement
            1,                     // descriptorCount
            m_pod_descriptor_type, // descriptorType
            nullptr,               // pImageInfo
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
                                                 0, // offset
                                                 VK_WHOLE_SIZE};

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
cvk_kernel::create_pipeline(const VkSpecializationInfo& specializationInfo) {
    const VkComputePipelineCreateInfo createInfo = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, // sType
        nullptr,                                        // pNext
        0,                                              // flags
        {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, // sType
            nullptr,                                             // pNext
            0,                                                   // flags
            VK_SHADER_STAGE_COMPUTE_BIT,                         // stage
            program()->shader_module(),                          // module
            name().c_str(),
            &specializationInfo // pSpecializationInfo
        },                      // stage
        pipeline_layout(),      // layout
        VK_NULL_HANDLE,         // basePipelineHandle
        0                       // basePipelineIndex
    };

    VkPipeline pipeline;
    auto vkdev = m_context->device()->vulkan_device();
    VkResult res = vkCreateComputePipelines(vkdev, m_pipeline_cache, 1,
                                            &createInfo, nullptr, &pipeline);

    if (res != VK_SUCCESS) {
        cvk_error_fn("Could not create compute pipeline: %s",
                     vulkan_error_string(res));
        return VK_NULL_HANDLE;
    }

    return pipeline;
}

cl_int cvk_kernel::set_arg(cl_uint index, size_t size, const void* value) {
    std::lock_guard<std::mutex> lock(m_lock);

    auto const& arg = m_args[index];

    return m_argument_values->set_arg(arg, size, value);
}
