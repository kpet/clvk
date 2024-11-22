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

#include <algorithm>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "spirv-tools/libspirv.h"
#include "spirv/unified1/spirv.hpp"
#include <vulkan/vulkan.h>

#include "cl_headers.hpp"
#include "device_properties.hpp"
#include "icd.hpp"
#include "objects.hpp"
#include "sha1.hpp"
#include "vkutils.hpp"

struct cvk_vulkan_extension_functions {
    PFN_vkGetCalibratedTimestampsEXT vkGetCalibratedTimestampsEXT;
    PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR;
};

#define MAKE_NAME_VERSION(major, minor, patch, name)                           \
    cl_name_version { CL_MAKE_VERSION(major, minor, patch), name }

static cl_version gOpenCLCVersion = CL_MAKE_VERSION(1, 2, 0);

static constexpr bool devices_support_images() { return true; }

struct cvk_platform;

struct cvk_device : public _cl_device_id,
                    object_magic_header<object_magic::device> {

    cvk_device(cvk_platform* platform, VkPhysicalDevice pd)
        : m_platform(platform), m_pdev(pd) {
        vkGetPhysicalDeviceProperties(m_pdev, &m_properties);
        vkGetPhysicalDeviceMemoryProperties(m_pdev, &m_mem_properties);

        switch (m_properties.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            m_type = CL_DEVICE_TYPE_GPU;
            break;
        case VK_PHYSICAL_DEVICE_TYPE_CPU:
            m_type = CL_DEVICE_TYPE_CPU;
            break;
        case VK_PHYSICAL_DEVICE_TYPE_OTHER:
        default:
            m_type = CL_DEVICE_TYPE_ACCELERATOR;
            break;
        }

        m_clvk_properties = create_cvk_device_properties(
            m_properties.deviceName, m_properties.vendorID,
            m_properties.deviceID);
    }

    static cvk_device* create(cvk_platform* platform, VkInstance instance,
                              VkPhysicalDevice pdev);

    virtual ~cvk_device() {
        for (auto entry : m_pipeline_caches) {
            save_pipeline_cache(entry.first, entry.second);
            vkDestroyPipelineCache(m_dev, entry.second, nullptr);
        }
        vkDestroyDevice(m_dev, nullptr);
    }

#ifdef CLVK_UNIT_TESTING_ENABLED

    VkPhysicalDeviceLimits& vulkan_limits_writable() {
        return m_properties.limits;
    }

    void restore_device_properties() {
        vkGetPhysicalDeviceProperties(m_pdev, &m_properties);
    }

#endif

    const VkPhysicalDeviceLimits& vulkan_limits() const {
        return m_properties.limits;
    }
    cvk_platform* platform() const { return m_platform; }
    const char* name() const { return m_properties.deviceName; }
    uint32_t vendor_id() const { return m_properties.vendorID; }
    std::string vendor() const;

    CHECK_RETURN uint32_t memory_type_index_for_resource(
        uint32_t valid_memory_type_bits,
        VkMemoryPropertyFlags required_properties,
        VkMemoryPropertyFlags avoid_properties) const {

        for (uint32_t k = 0; k < m_mem_properties.memoryTypeCount; k++) {
            auto dev_properties = m_mem_properties.memoryTypes[k].propertyFlags;
            bool valid = (1ULL << k) & valid_memory_type_bits;
            bool satisfactory =
                (dev_properties & required_properties) == required_properties;
            bool avoid = dev_properties & avoid_properties;
            if (satisfactory && valid && !avoid) {
                return k;
            }
        }

        return VK_MAX_MEMORY_TYPES;
    }

    CHECK_RETURN uint32_t memory_type_index_for_resource(
        uint32_t valid_memory_type_bits, int num_supported,
        const VkMemoryPropertyFlags* supported_memory_types) const {

        for (int i = 0; i < num_supported; i++) {
            auto k = memory_type_index_for_resource(
                valid_memory_type_bits, supported_memory_types[i], 0);
            if (k != VK_MAX_MEMORY_TYPES) {
                cvk_debug_fn("selected %u", k);
                return k;
            }
        }

        return VK_MAX_MEMORY_TYPES;
    }

    CHECK_RETURN uint32_t memory_type_index_for_resource(
        uint32_t valid_memory_type_bits, int num_supported,
        const VkMemoryPropertyFlags* supported_memory_types,
        VkMemoryPropertyFlags avoid_memory_type) const {

        for (int i = 0; i < num_supported; i++) {
            auto k = memory_type_index_for_resource(valid_memory_type_bits,
                                                    supported_memory_types[i],
                                                    avoid_memory_type);
            if (k != VK_MAX_MEMORY_TYPES) {
                cvk_debug_fn("selected %u", k);
                return k;
            }
        }
        return memory_type_index_for_resource(
            valid_memory_type_bits, num_supported, supported_memory_types);
    }

    CHECK_RETURN bool memory_index_is_coherent(uint32_t index) const {
        return m_mem_properties.memoryTypes[index].propertyFlags &
               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    }

    struct allocation_parameters {
        VkDeviceSize size;
        uint32_t memory_type_index;
        bool memory_coherent;
    };

    static constexpr VkMemoryPropertyFlags image_supported_memory_types[] = {
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    };

    static constexpr VkMemoryPropertyFlags buffer_supported_memory_types[] = {
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,

        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT,

        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    };

    CHECK_RETURN uint32_t
    memory_type_index_for_image(uint32_t valid_memory_type_bits) const {
        return memory_type_index_for_resource(
            valid_memory_type_bits, ARRAY_SIZE(image_supported_memory_types),
            image_supported_memory_types, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }

    CHECK_RETURN allocation_parameters select_memory_for(VkImage image) const {
        VkMemoryRequirements memreqs;
        vkGetImageMemoryRequirements(m_dev, image, &memreqs);

        allocation_parameters ret;
        ret.size = memreqs.size;
        ret.memory_type_index =
            memory_type_index_for_image(memreqs.memoryTypeBits);
        ret.memory_coherent = memory_index_is_coherent(ret.memory_type_index);

        return ret;
    }

    CHECK_RETURN uint32_t
    memory_type_index_for_buffer(uint32_t valid_memory_type_bits) const {
        return memory_type_index_for_resource(
            valid_memory_type_bits, ARRAY_SIZE(buffer_supported_memory_types),
            buffer_supported_memory_types);
    }

    CHECK_RETURN allocation_parameters
    select_memory_for(VkBuffer buffer, cl_mem_flags flags) const {
        UNUSED(flags);
        VkMemoryRequirements memreqs;
        vkGetBufferMemoryRequirements(m_dev, buffer, &memreqs);

        allocation_parameters ret;
        ret.size = memreqs.size;
        ret.memory_type_index =
            memory_type_index_for_buffer(memreqs.memoryTypeBits);
        ret.memory_coherent = memory_index_is_coherent(ret.memory_type_index);

        return ret;
    }

    uint64_t global_mem_size() const {
        // Return the size of the smallest memory heap that can be used to
        // allocate images or buffers
        uint64_t size = UINT64_MAX;
        uint32_t type_index = VK_MAX_MEMORY_TYPES;
        uint32_t heap_index = VK_MAX_MEMORY_HEAPS;

        // buffers
        type_index = memory_type_index_for_buffer(0xFFFFFFFFU);
        heap_index = m_mem_properties.memoryTypes[type_index].heapIndex;
        size = std::min(size, m_mem_properties.memoryHeaps[heap_index].size);

        // images
        type_index = memory_type_index_for_image(0xFFFFFFFFU);
        heap_index = m_mem_properties.memoryTypes[type_index].heapIndex;
        size = std::min(size, m_mem_properties.memoryHeaps[heap_index].size);

        double percentage_of_available_memory_reported =
            static_cast<double>(
                config.percentage_of_available_memory_reported()) /
            100;
        cvk_info("Using %u%% of total memory size",
                 config.percentage_of_available_memory_reported());

        return size * percentage_of_available_memory_reported;
    }

    uint64_t max_mem_alloc_size() const {
        auto global_memory_size = global_mem_size();
        CVK_ASSERT(global_memory_size % 4 == 0);
        // Min memory as per the specs.
        auto specMinAllocSz =
            std::max(std::min((uint64_t)(1024 * 1024 * 1024),
                              (uint64_t)(global_memory_size / 4)),
                     (uint64_t)(32 * 1024 * 1024));
        // Max memory allocation for single buffer can be adjusted with
        // environment variable CLVK_MEM_MAX_ALLOC_SIZE_MB.
        // For multiple allocations(total memory allocations), environment var
        // CLVK_PERCENTAGE_OF_AVAILABLE_MEMORY_REPORTED can be adjusted.
        auto maxAllocSz = m_maintenance3_properties.maxMemoryAllocationSize;
        if (config.max_mem_alloc_size_mb.set) {
            maxAllocSz =
                std::min(maxAllocSz, (uint64_t)config.max_mem_alloc_size_mb() *
                                         1024 * 1024);
        }
        maxAllocSz = std::min(maxAllocSz, global_memory_size);

        if (specMinAllocSz > maxAllocSz) {
            cvk_warn("Returning value (%s) for CL_DEVICE_MAX_MEM_ALLOC_SIZE "
                     "which is\n"
                     "smaller than required by the OpenCL specification (%s). ",
                     pretty_size(maxAllocSz).c_str(),
                     pretty_size(specMinAllocSz).c_str());
        }

        return maxAllocSz;
    }

    size_t image_max_buffer_size() const {
        return std::min((uint64_t)vulkan_limits().maxTexelBufferElements,
                        max_mem_alloc_size());
    }

    cl_uint mem_base_addr_align() const {
        // The OpenCL spec requires at least 1024 bits (long16's alignment)
        uint32_t required_by_vulkan_impl =
            m_properties.limits.minStorageBufferOffsetAlignment * 8;
        return std::max(required_by_vulkan_impl, 1024U);
    }

    cl_ulong global_mem_cache_size() const {
        return m_clvk_properties->get_global_mem_cache_size();
    }

    cl_uint num_compute_units() const {
        return m_clvk_properties->get_num_compute_units();
    }

    cl_uint max_samplers() const {
        // There are only 20 different possible samplers in OpenCL 1.2, cap the
        // number of supported samplers to that to help with negative testing of
        // the limit against Vulkan implementations that report a very large
        // number for maxPerStageDescriptorSamplers.
        return std::min(20u, vulkan_limits().maxPerStageDescriptorSamplers);
    }

    cl_uint max_work_item_dimensions() const { return 3; }

    size_t max_work_group_size() const {
        return vulkan_limits().maxComputeWorkGroupInvocations;
    }

    cl_uint sub_group_size() const {
        if (supports_subgroup_size_selection()) {
            if (config.force_subgroup_size.set &&
                config.force_subgroup_size() >= min_sub_group_size() &&
                config.force_subgroup_size() <= max_sub_group_size()) {
                return config.force_subgroup_size();
            } else if (config.force_subgroup_size.set) {
                cvk_warn_fn("CLVK_FORCE_SUBGROUP_SIZE as been set to '%u', "
                            "which is out of the supported range [%u, %u], "
                            "thus it will be ignored",
                            config.force_subgroup_size(), min_sub_group_size(),
                            max_sub_group_size());
            }
            if (m_preferred_subgroup_size != 0 &&
                m_preferred_subgroup_size >= min_sub_group_size() &&
                m_preferred_subgroup_size <= max_sub_group_size()) {
                return m_preferred_subgroup_size;
            } else if (config.preferred_subgroup_size.set) {
                cvk_warn_fn("CLVK_PREFERRED_SUBGROUP_SIZE as been set to '%u', "
                            "which is out of the supported range [%u, %u], "
                            "thus it will be ignored",
                            m_preferred_subgroup_size, min_sub_group_size(),
                            max_sub_group_size());
            }
        }
        return m_subgroup_properties.subgroupSize;
    }
    cl_uint min_sub_group_size() const {
        return m_subgroup_size_control_properties.minSubgroupSize;
    }
    cl_uint max_sub_group_size() const {
        return m_subgroup_size_control_properties.maxSubgroupSize;
    }

    cl_uint max_num_sub_groups() const {
        if (!supports_subgroups()) {
            return 0;
        }
        return ceil_div(max_work_group_size(),
                        static_cast<size_t>(sub_group_size()));
    }

    bool supports_dot_product() const {
        return m_features_shader_integer_dot_product.shaderIntegerDotProduct;
    }

    cl_device_integer_dot_product_capabilities_khr
    dot_product_capabilities() const {
        if (supports_dot_product()) {
            return CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_KHR |
                   CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_PACKED_KHR;
        } else {
            return 0;
        }
    }

    cl_device_integer_dot_product_acceleration_properties_khr
    dot_product_4x8bit_packed_properties() const {
        cl_device_integer_dot_product_acceleration_properties_khr res;
        res.signed_accelerated =
            m_integer_dot_product_properties
                .integerDotProduct4x8BitPackedSignedAccelerated;
        res.unsigned_accelerated =
            m_integer_dot_product_properties
                .integerDotProduct4x8BitPackedUnsignedAccelerated;
        res.mixed_signedness_accelerated =
            m_integer_dot_product_properties
                .integerDotProduct4x8BitPackedMixedSignednessAccelerated;
        res.accumulating_saturating_signed_accelerated =
            m_integer_dot_product_properties
                .integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated;
        res.accumulating_saturating_unsigned_accelerated =
            m_integer_dot_product_properties
                .integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated;
        res.accumulating_saturating_mixed_signedness_accelerated =
            m_integer_dot_product_properties
                .integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated;
        return res;
    }

    cl_device_integer_dot_product_acceleration_properties_khr
    dot_product_8bit_properties() const {
        cl_device_integer_dot_product_acceleration_properties_khr res;
        res.signed_accelerated = m_integer_dot_product_properties
                                     .integerDotProduct8BitSignedAccelerated;
        res.unsigned_accelerated =
            m_integer_dot_product_properties
                .integerDotProduct8BitUnsignedAccelerated;
        res.mixed_signedness_accelerated =
            m_integer_dot_product_properties
                .integerDotProduct8BitMixedSignednessAccelerated;
        res.accumulating_saturating_signed_accelerated =
            m_integer_dot_product_properties
                .integerDotProductAccumulatingSaturating8BitSignedAccelerated;
        res.accumulating_saturating_unsigned_accelerated =
            m_integer_dot_product_properties
                .integerDotProductAccumulatingSaturating8BitUnsignedAccelerated;
        res.accumulating_saturating_mixed_signedness_accelerated =
            m_integer_dot_product_properties
                .integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated;
        return res;
    }

    bool supports_images() const {
        return devices_support_images() ? CL_TRUE : CL_FALSE;
    }

    bool supports_read_write_images() const {
        return supports_capability(
                   spv::CapabilityStorageImageReadWithoutFormat) &&
               supports_capability(
                   spv::CapabilityStorageImageWriteWithoutFormat);
    }

    bool supports_fp16() const { return m_has_fp16_support; }

    bool supports_fp64() const { return m_has_fp64_support; }

    bool supports_int8() const { return m_has_int8_support; }

    bool supports_subgroups() const { return m_has_subgroups_support; }

    bool supports_subgroup_size_selection() const {
        return m_has_subgroup_size_selection;
    }

    bool supports_non_uniform_decoration() const {
        return (m_properties.apiVersion >= VK_MAKE_VERSION(1, 2, 0) ||
                is_vulkan_extension_enabled(
                    VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)) &&
               !m_clvk_properties->is_non_uniform_decoration_broken();
    }

    bool supports_atomic_order_acq_rel() const {
        return m_features_vulkan_memory_model.vulkanMemoryModel;
    }

    bool supports_atomic_scope_device() const {
        return m_features_vulkan_memory_model.vulkanMemoryModelDeviceScope;
    }

    bool compiler_available() const {
#ifdef COMPILER_AVAILABLE
        return true;
#else
        return false;
#endif
    }

    CHECK_RETURN const std::string& extension_string() const {
        return m_extension_string;
    }
    CHECK_RETURN const std::vector<cl_name_version>& extensions() const {
        return m_extensions;
    }

    /// Returns true if the device supports the given SPIR-V capability.
    CHECK_RETURN bool supports_capability(spv::Capability capability) const;

    /// Returns true if std430 layout is supported for uniform buffers.
    CHECK_RETURN bool supports_ubo_stdlayout() const {
        return m_features_ubo_stdlayout.uniformBufferStandardLayout;
    }

    cl_version version() const { return config.opencl_version; }

    cl_version c_version() const { return gOpenCLCVersion; }

    std::string version_string() const {
        return "OpenCL " + std::to_string(CL_VERSION_MAJOR(version())) + "." +
               std::to_string(CL_VERSION_MINOR(version())) + " " +
               version_desc();
    }

    std::string c_version_string() const {
        return "OpenCL C " + std::to_string(CL_VERSION_MAJOR(c_version())) +
               "." + std::to_string(CL_VERSION_MINOR(c_version())) + " " +
               version_desc();
    }

    std::string profile() const { return "FULL_PROFILE"; }

    std::string driver_version() const {
        return std::to_string(CL_VERSION_MAJOR(version())) + "." +
               std::to_string(CL_VERSION_MINOR(version())) + " " +
               version_desc();
    }

    const std::string& ils_string() const { return m_ils_string; }

    const std::vector<cl_name_version>& ils() const { return m_ils; }

    const std::vector<cl_name_version>& opencl_c_versions() const {
        return m_opencl_c_versions;
    }
    const std::vector<cl_name_version>& opencl_c_features() const {
        return m_opencl_c_features;
    }

    cl_device_type type() const { return m_type; }

    cl_bool has_host_unified_memory() const {
        switch (m_properties.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_CPU:
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            return CL_TRUE;
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        case VK_PHYSICAL_DEVICE_TYPE_OTHER:
        default:
            return CL_FALSE;
        }
    }

    cvk_vulkan_queue_wrapper& vulkan_queue_allocate() {
        static std::mutex queue_allocation_lock;
        std::lock_guard lock(queue_allocation_lock);

        // Simple round-robin allocation for now
        auto& queue = m_vulkan_queues[m_vulkan_queue_alloc_index++];

        if (m_vulkan_queue_alloc_index == m_vulkan_queues.size()) {
            m_vulkan_queue_alloc_index = 0;
        }

        return queue;
    }

    cl_device_fp_config fp_config(cl_device_info fptype) const {
        if ((fptype == CL_DEVICE_HALF_FP_CONFIG) && supports_fp16()) {
            return CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN | CL_FP_FMA;
        }
        if (fptype == CL_DEVICE_SINGLE_FP_CONFIG) {
            return CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN | CL_FP_FMA;
        }

        if ((fptype == CL_DEVICE_DOUBLE_FP_CONFIG) && supports_fp64()) {
            return CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO |
                   CL_FP_ROUND_TO_INF | CL_FP_INF_NAN | CL_FP_FMA |
                   CL_FP_DENORM;
        }

        return 0;
    }

    VkPhysicalDevice vulkan_physical_device() const { return m_pdev; }

    VkDevice vulkan_device() const { return m_dev; }

    const VkPhysicalDevice8BitStorageFeaturesKHR&
    device_8bit_storage_features() const {
        return m_features_8bit_storage;
    }
    const VkPhysicalDevice16BitStorageFeaturesKHR&
    device_16bit_storage_features() const {
        return m_features_16bit_storage;
    }
    const VkPhysicalDeviceVulkanMemoryModelFeaturesKHR&
    vulkan_memory_model_features() const {
        return m_features_vulkan_memory_model;
    }

    uint32_t vulkan_max_push_constants_size() const {
        return m_properties.limits.maxPushConstantsSize;
    }

    uint32_t vulkan_max_uniform_buffer_range() const {
        return m_properties.limits.maxUniformBufferRange;
    }

    bool supports_non_uniform_workgroup() const { return true; }

    void select_work_group_size(const std::array<uint32_t, 3>& global_size,
                                std::array<uint32_t, 3>& local_size) const;

    bool is_vulkan_extension_enabled(const char* ext) const {
        return std::find(m_vulkan_device_extensions.begin(),
                         m_vulkan_device_extensions.end(),
                         ext) != m_vulkan_device_extensions.end();
    }

    // Get a previously created Vulkan pipeline cache for a given SPIR-V binary,
    // or create a new one if necessary. Returns true if an existing pipeline
    // cache was found and reused.
    bool get_pipeline_cache(const std::vector<uint32_t>& spirv,
                            VkPipelineCache& pipeline_cache);

    spv_target_env vulkan_spirv_env() const { return m_vulkan_spirv_env; }

    CHECK_RETURN bool has_timer_support() const { return m_has_timer_support; }

    CHECK_RETURN cl_int get_device_host_timer(cl_ulong* dev_ts,
                                              cl_ulong* host_ts) const;
    cl_ulong device_timer_to_host(cl_ulong dev, cl_ulong sync_dev,
                                  cl_ulong sync_host) const;

    uint64_t timestamp_to_ns(uint64_t ts) const {
        double ns_per_tick = vulkan_limits().timestampPeriod;
        // Most implementations seem to use 1 ns = 1 tick, handle this as a
        // special case to not lose precision.
        if (ns_per_tick == 1.0) {
            return ts;
        } else {
            return ts * ns_per_tick; // TODO is it good enough?
        }
    }

    size_t preferred_work_group_size_multiple() const {
        // Use a hard-coded value that ought to be better than 1 on most devices
        return 16;
    }

    // Driver-specific behaviors.
    enum cvk_driver_behavior
    {
        use_reset_command_buffer_bit = 0x00000001,
    };
    bool is_driver_behavior_enabled(cvk_driver_behavior behavior) const {
        return m_driver_behaviors & behavior;
    }

    // Device UUID
    static_assert(CL_UUID_SIZE_KHR == VK_UUID_SIZE,
                  "Vulkan and CL UUID must have the same size");
    static_assert(CL_LUID_SIZE_KHR == VK_LUID_SIZE,
                  "Vulkan and CL LUID must have the same size");

    const cl_uchar* uuid() const { return m_device_id_properties.deviceUUID; }

    const cl_uchar* driver_uuid() const {
        return m_device_id_properties.driverUUID;
    }

    cl_bool luid_valid() const {
        return m_device_id_properties.deviceLUIDValid;
    }

    const cl_uchar* luid() const { return m_device_id_properties.deviceLUID; }

    cl_uint node_mask() const { return m_device_id_properties.deviceNodeMask; }

    cl_device_pci_bus_info_khr pci_bus_info() const {
        return {
            m_pci_bus_info_properties.pciDomain,
            m_pci_bus_info_properties.pciBus,
            m_pci_bus_info_properties.pciDevice,
            m_pci_bus_info_properties.pciFunction,
        };
    }

    cl_uint get_max_cmd_batch_size() const { return m_max_cmd_batch_size; }
    cl_uint get_max_first_cmd_batch_size() const {
        return m_max_first_cmd_batch_size;
    }
    cl_uint get_max_cmd_group_size() const { return m_max_cmd_group_size; }
    cl_uint get_max_first_cmd_group_size() const {
        return m_max_first_cmd_group_size;
    }

    cl_uint address_bits() const { return m_spirv_arch == "spir64" ? 64 : 32; }
    bool uses_physical_addressing() const { return m_physical_addressing; }

    const std::string& get_device_specific_compile_options() const {
        return m_device_compiler_options;
    }

    const cvk_vulkan_extension_functions& vkfns() const { return m_vkfns; }

    bool is_bgra_format_not_supported_for_image1d_buffer() const {
        return m_clvk_properties
            ->is_bgra_format_not_supported_for_image1d_buffer();
    }

    bool is_image_format_disabled(cl_image_format format) const {
        return m_clvk_properties->get_disabled_image_formats().count(format) !=
               0;
    }

private:
    std::string version_desc() const {
        std::string ret = "CLVK on Vulkan v";
        ret += vulkan_version_string(m_properties.apiVersion);
        ret += " driver " + std::to_string(m_properties.driverVersion);
        return ret;
    }

    CHECK_RETURN bool init_queues(uint32_t* num_queues, uint32_t* queue_family);
    CHECK_RETURN bool init_extensions();
    void init_clvk_runtime_behaviors();
    void init_vulkan_properties(VkInstance instance);
    void init_driver_behaviors();
    void init_features(VkInstance instance);
    void init_command_pointers(VkInstance instance);
    void init_compiler_options();
    void build_extension_ils_list();
    CHECK_RETURN bool create_vulkan_queues_and_device(uint32_t num_queues,
                                                      uint32_t queue_family);
    CHECK_RETURN bool init_time_management(VkInstance instance);
    void init_spirv_environment();
    void log_limits_and_memory_information();
    CHECK_RETURN bool init(VkInstance instance);

    cvk_platform* m_platform;
    cl_device_type m_type{};

    cvk_vulkan_extension_functions m_vkfns{};
    VkPhysicalDevice m_pdev;
    // Properties
    VkPhysicalDeviceProperties m_properties;
    VkPhysicalDeviceMaintenance3Properties m_maintenance3_properties;
    VkPhysicalDeviceMemoryProperties m_mem_properties;
    VkPhysicalDeviceDriverPropertiesKHR m_driver_properties;
    VkPhysicalDeviceIDPropertiesKHR m_device_id_properties;
    VkPhysicalDeviceSubgroupProperties m_subgroup_properties{};
    VkPhysicalDeviceSubgroupSizeControlProperties
        m_subgroup_size_control_properties{};
    VkPhysicalDevicePCIBusInfoPropertiesEXT m_pci_bus_info_properties;
    VkPhysicalDeviceShaderIntegerDotProductProperties
        m_integer_dot_product_properties{};
    // Vulkan features
    VkPhysicalDeviceFeatures2 m_features{};
    VkPhysicalDeviceVariablePointerFeatures m_features_variable_pointer{};
    VkPhysicalDeviceShaderFloat16Int8FeaturesKHR m_features_float16_int8{};
    VkPhysicalDeviceUniformBufferStandardLayoutFeaturesKHR
        m_features_ubo_stdlayout{};
    VkPhysicalDevice8BitStorageFeaturesKHR m_features_8bit_storage{};
    VkPhysicalDevice16BitStorageFeaturesKHR m_features_16bit_storage{};
    VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures
        m_features_shader_subgroup_extended_types{};
    VkPhysicalDeviceSubgroupSizeControlFeatures
        m_features_subgroup_size_control{};
    VkPhysicalDeviceVulkanMemoryModelFeaturesKHR
        m_features_vulkan_memory_model{};
    VkPhysicalDeviceBufferDeviceAddressFeaturesKHR
        m_features_buffer_device_address{};
    VkPhysicalDeviceFloatControlsProperties m_float_controls_properties{};
    VkPhysicalDeviceShaderIntegerDotProductFeatures
        m_features_shader_integer_dot_product{};
    VkPhysicalDeviceGlobalPriorityQueryFeaturesKHR
        m_features_queue_global_priority{};

    VkDevice m_dev;
    std::vector<const char*> m_vulkan_device_extensions;

    std::vector<cvk_vulkan_queue_wrapper> m_vulkan_queues;
    uint32_t m_vulkan_queue_alloc_index;

    std::string m_extension_string;
    std::vector<cl_name_version> m_extensions;
    std::string m_ils_string;
    std::vector<cl_name_version> m_ils;
    std::vector<cl_name_version> m_opencl_c_versions;
    std::vector<cl_name_version> m_opencl_c_features;
    std::string m_device_compiler_options;

    uint32_t m_driver_behaviors;

    // Pipeline caching
    std::string get_pipeline_cache_filename(const cvk_sha1_hash& sha1) const;
    void save_pipeline_cache(const cvk_sha1_hash& sha1,
                             const VkPipelineCache& pipeline_cache) const;
    struct sha1_hasher {
        size_t operator()(const cvk_sha1_hash& sha1) const {
            size_t result = 0;
            for (unsigned i = 0; i < SHA1_DIGEST_NUM_WORDS; i++) {
                // TODO: Better hash?
                result *= 59;
                result += sha1[i];
            }
            return result;
        }
    };
    std::unordered_map<cvk_sha1_hash, VkPipelineCache, sha1_hasher>
        m_pipeline_caches;
    std::mutex m_pipeline_cache_mutex;

    bool m_has_timer_support{};
    bool m_has_fp16_support{};
    bool m_has_fp64_support{};
    bool m_has_int8_support{};
    bool m_has_subgroups_support{};
    bool m_has_subgroup_size_selection{};

    cl_uint m_max_cmd_batch_size;
    cl_uint m_max_first_cmd_batch_size;
    cl_uint m_max_cmd_group_size;
    cl_uint m_max_first_cmd_group_size;

    std::string m_spirv_arch;
    bool m_physical_addressing;

    cl_uint m_preferred_subgroup_size{};

    spv_target_env m_vulkan_spirv_env;

    std::unique_ptr<cvk_device_properties> m_clvk_properties;
};

static inline cvk_device* icd_downcast(cl_device_id device) {
    return static_cast<cvk_device*>(device);
}

struct cvk_platform : public _cl_platform_id,
                      object_magic_header<object_magic::platform> {
    cvk_platform() {
        m_extensions = {
            MAKE_NAME_VERSION(1, 0, 0, "cl_khr_icd"),
            MAKE_NAME_VERSION(1, 0, 0, "cl_khr_extended_versioning"),
        };

        for (auto& ext : m_extensions) {
            m_extension_string += ext.name;
            m_extension_string += " ";
        }
    }
    ~cvk_platform() {
        for (auto dev : m_devices) {
            delete dev;
        }
    }

    CHECK_RETURN bool create_device(VkInstance instance,
                                    VkPhysicalDevice pdev) {
        auto dev = cvk_device::create(this, instance, pdev);
        if (dev != nullptr) {
            m_devices.push_back(dev);
            return true;
        } else {
            return false;
        }
    }

    cl_version version() const { return config.opencl_version; }

    std::string version_string() const {
        std::string ret = "OpenCL ";
        auto ver = version();
        ret += std::to_string(CL_VERSION_MAJOR(ver));
        ret += ".";
        ret += std::to_string(CL_VERSION_MINOR(ver));
        ret += " clvk";
        return ret;
    }

    std::string name() const { return "clvk"; }

    std::string vendor() const { return "clvk"; }

    std::string profile() const { return "FULL_PROFILE"; }

    std::string icd_suffix() const { return "clvk"; }

    const std::vector<cvk_device*>& devices() const { return m_devices; }

    const std::vector<cl_name_version>& extensions() const {
        return m_extensions;
    }

    const std::string& extension_string() const { return m_extension_string; }

    cl_ulong host_timer_resolution() const {
        for (auto dev : m_devices) {
            if (!dev->has_timer_support()) {
                return 0;
            }
        }
        return 1;
    }

private:
    std::vector<cl_name_version> m_extensions;
    std::string m_extension_string;
    std::vector<cvk_device*> m_devices;
};

static inline cvk_platform* icd_downcast(cl_platform_id platform) {
    return static_cast<cvk_platform*>(platform);
}
