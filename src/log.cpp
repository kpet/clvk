// Copyright 2020 The clvk authors.
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

#include "log.hpp"
#include "config.hpp"
#include "queue.hpp"

#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#ifndef _MSC_VER
#include <unistd.h>
#endif

#ifdef WIN32
#include <io.h>
#endif

#define CASE(X)                                                                \
    case X:                                                                    \
        return #X;

static int gLoggingLevel;
static uint64_t gLoggingGroupMask;
static bool gLoggingColour;
static FILE* gLoggingFile;

static uint64_t init_logging_groups() {
    uint64_t mask = loggroup::all;

    static std::unordered_map<std::string, loggroup> groups{
        {"refcounting", loggroup::refcounting},
        {"api", loggroup::api},
        {"event", loggroup::event},
        {"validation", loggroup::validation},
        {"none", loggroup::none},
        {"all", loggroup::all},
    };

    if (config.log_groups.set) {
        std::stringstream sstr(config.log_groups());
        bool first_group_enable_seen = false;
        while (sstr.good()) {
            std::string group_endis;
            getline(sstr, group_endis, ',');
            bool is_disable = false;
            if (group_endis.rfind("-", 0) == 0) {
                group_endis.erase(0, 1);
                is_disable = true;
            }
            loggroup group;
            if (groups.count(group_endis) != 0) {
                group = groups.at(group_endis);
            } else {
                fprintf(stderr, "FATAL: Unknown log group '%s'.\n",
                        group_endis.c_str());
                exit(EXIT_FAILURE);
            }

            if (is_disable) {
                mask &= ~group;
            } else {
                if (!first_group_enable_seen) {
                    mask = 0;
                    first_group_enable_seen = true;
                }
                mask |= group;
            }
        }
    }

    return mask;
}

void init_logging() {
    loglevel setting = static_cast<loglevel>(config.log());
    if (config.log.set) {
        if ((config.log < loglevel::fatal) || (config.log > loglevel::debug)) {
            // FIXME handle all errors
            fprintf(stderr, "FATAL: Unknown log level '%u'.\n", config.log());
            exit(EXIT_FAILURE);
        }
        setting = static_cast<loglevel>(config.log());
    }
    gLoggingLevel = setting;

    gLoggingGroupMask = init_logging_groups();

    if (config.log_dest.set) {

        std::string val(config.log_dest);

        if (val == "stdout") {
            gLoggingFile = stdout;
        } else if (val == "stderr") {
            gLoggingFile = stderr;
        } else if (val.rfind("file:", 0) == 0) {

            val.erase(0, strlen("file:"));

            gLoggingFile = fopen(val.c_str(), "w+");

            if (gLoggingFile == nullptr) {
                fprintf(stderr, "FATAL: Could not open log file '%s': %s.\n",
                        val.c_str(), strerror(errno));
                exit(EXIT_FAILURE);
            }
        } else {
            fprintf(stderr, "FATAL: Unknown log destination '%s'.\n",
                    val.c_str());
            exit(EXIT_FAILURE);
        }
    } else {
        gLoggingFile = stderr;
    }

    bool isTTY = isatty(fileno(gLoggingFile));
    if (isTTY) {
        gLoggingColour = true;
    } else {
        gLoggingColour = false;
    }

    if (config.log_colour.set) {
        gLoggingColour = config.log_colour;
    }
}

void term_logging() {
    if ((gLoggingFile != stdout) && (gLoggingFile != stderr)) {
        fclose(gLoggingFile);
    }
}

bool cvk_log_level_enabled(loglevel level) { return gLoggingLevel >= level; }
bool cvk_log_group_enabled(uint64_t group_mask) {
    return gLoggingGroupMask & group_mask;
}

static const char colourRed[] = "\e[0;31m";
static const char colourYellow[] = "\e[0;33m";
static const char colourReset[] = "\e[0m";

void cvk_log(uint64_t group_mask, loglevel level, const char* fmt, ...) {

    if (!cvk_log_level_enabled(level)) {
        return;
    }

    if (!cvk_log_group_enabled(group_mask)) {
        return;
    }

    const char* colourCode = nullptr;

    if (gLoggingColour) {

        switch (level) {
        case loglevel::fatal:
        case loglevel::error:
            colourCode = colourRed;
            break;
        case loglevel::warn:
            colourCode = colourYellow;
            break;
        case loglevel::info:
        case loglevel::debug:
            break;
        }

        if (colourCode != nullptr) {
            fprintf(gLoggingFile, "%s", colourCode);
        }
    }

    fprintf(gLoggingFile, "[CLVK] ");

    va_list args;
    va_start(args, fmt);
    vfprintf(gLoggingFile, fmt, args);
    va_end(args);

    if ((gLoggingColour) && (colourCode != nullptr)) {
        fprintf(gLoggingFile, "%s", colourReset);
    }

    // Force to flush error and fatal logs
    if (level <= loglevel::error) {
        fflush(gLoggingFile);
    }

    if (level == loglevel::fatal) {
        exit(EXIT_FAILURE);
    }
}

std::string pretty_size(uint64_t size) {
    std::string ret;

    static const char* units[] = {" B", "kB", "MB", "GB", "TB"};
    int unit = 0;
    float fsize = size;
    while (fsize > 1024) {
        unit++;
        fsize /= 1024;
    }

    ret = std::to_string(fsize) + " " + units[unit];

    return ret;
}

const char* vulkan_error_string(VkResult result) {
    switch (result) {
        CASE(VK_SUCCESS)
        CASE(VK_NOT_READY)
        CASE(VK_TIMEOUT)
        CASE(VK_EVENT_SET)
        CASE(VK_EVENT_RESET)
        CASE(VK_INCOMPLETE)
        CASE(VK_ERROR_OUT_OF_HOST_MEMORY)
        CASE(VK_ERROR_OUT_OF_DEVICE_MEMORY)
        CASE(VK_ERROR_INITIALIZATION_FAILED)
        CASE(VK_ERROR_DEVICE_LOST)
        CASE(VK_ERROR_MEMORY_MAP_FAILED)
        CASE(VK_ERROR_LAYER_NOT_PRESENT)
        CASE(VK_ERROR_EXTENSION_NOT_PRESENT)
        CASE(VK_ERROR_FEATURE_NOT_PRESENT)
        CASE(VK_ERROR_INCOMPATIBLE_DRIVER)
        CASE(VK_ERROR_TOO_MANY_OBJECTS)
        CASE(VK_ERROR_FORMAT_NOT_SUPPORTED)
        CASE(VK_ERROR_FRAGMENTED_POOL)
        CASE(VK_ERROR_SURFACE_LOST_KHR)
        CASE(VK_ERROR_NATIVE_WINDOW_IN_USE_KHR)
        CASE(VK_SUBOPTIMAL_KHR)
        CASE(VK_ERROR_OUT_OF_DATE_KHR)
        CASE(VK_ERROR_INCOMPATIBLE_DISPLAY_KHR)
        CASE(VK_ERROR_VALIDATION_FAILED_EXT)
        CASE(VK_ERROR_INVALID_SHADER_NV)
        CASE(VK_ERROR_INVALID_EXTERNAL_HANDLE)
        CASE(VK_ERROR_FRAGMENTATION_EXT)
        CASE(VK_ERROR_NOT_PERMITTED_EXT)
        CASE(VK_ERROR_OUT_OF_POOL_MEMORY_KHR)
    default:
        return "Unknown vulkan error";
    }
}

std::string vulkan_memory_property_flags_string(VkMemoryPropertyFlags flags) {
    std::string str;

    if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        str += "HOST_VISIBLE ";
    }

    if (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
        str += "HOST_COHERENT ";
    }

    if (flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) {
        str += "HOST_CACHED ";
    }

    if (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
        str += "DEVICE_LOCAL ";
    }

    if (flags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) {
        str += "LAZILY_ALLOCATED ";
    }

    return str;
}

std::string vulkan_queue_flags_string(VkQueueFlags flags) {
    std::string str;

    if (flags & VK_QUEUE_GRAPHICS_BIT) {
        str += "GRAPHICS ";
    }

    if (flags & VK_QUEUE_COMPUTE_BIT) {
        str += "COMPUTE ";
    }

    if (flags & VK_QUEUE_TRANSFER_BIT) {
        str += "TRANSFER ";
    }

    if (flags & VK_QUEUE_SPARSE_BINDING_BIT) {
        str += "SPARSE_BINDING ";
    }

    return str;
}

std::string vulkan_physical_device_type_string(VkPhysicalDeviceType type) {
    switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_OTHER:
        return "Other";
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        return "Integrated GPU";
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        return "Discrete GPU";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        return "Virtual GPU";
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
        return "CPU";
    default:
        return "Unknown";
    }
}

std::string vulkan_calibrateable_time_domain_string(VkTimeDomainEXT td) {
    switch (td) {
    case VK_TIME_DOMAIN_DEVICE_EXT:
        return "Device";
    case VK_TIME_DOMAIN_CLOCK_MONOTONIC_EXT:
        return "Clock monotonic";
    case VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_EXT:
        return "Clock monotonic raw";
    case VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT:
        return "Query performance counter";
    default:
        return "Unknown";
    }
}

std::string vulkan_format_features_string(VkFormatFeatureFlags flags) {
    std::string str;

    if (flags & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) {
        str += "SAMPLED_IMAGE ";
    }

    if (flags & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) {
        str += "STORAGE_IMAGE ";
    }

    if (flags & VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT) {
        str += "STORAGE_IMAGE_ATOMIC ";
    }

    if (flags & VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT) {
        str += "UNIFORM_TEXEL_BUFFER ";
    }

    if (flags & VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT) {
        str += "STORAGE_TEXEL_BUFFER ";
    }

    if (flags & VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_ATOMIC_BIT) {
        str += "STORAGE_TEXEL_BUFFER_ATOMIC ";
    }

    if (flags & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) {
        str += "SAMPLED_IMAGE_FILTER_LINEAR ";
    }

    if (flags & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT) {
        str += "TRANSFER_SRC ";
    }

    if (flags & VK_FORMAT_FEATURE_TRANSFER_DST_BIT) {
        str += "TRANSFER_DST ";
    }

    return str;
}

std::string vulkan_vendor_id_string(VkVendorId vid) {
    switch (vid) {
    case VK_VENDOR_ID_VIV:
        return "Vivante";
    case VK_VENDOR_ID_VSI:
        return "VeriSilicon";
    case VK_VENDOR_ID_KAZAN:
        return "Kazan Software Renderer";
    case VK_VENDOR_ID_CODEPLAY:
        return "Codeplay Software Ltd.";
    case VK_VENDOR_ID_MESA:
        return "Mesa";
    case VK_VENDOR_ID_POCL:
        return "PoCL";
#ifdef VK_VENDOR_ID_MOBILEYE
    case VK_VENDOR_ID_MOBILEYE:
        return "Mobileye";
#endif
    default:
        return "Unknown Khronos Vendor ID";
    }
}

std::string cl_channel_order_to_string(cl_channel_order order) {
    switch (order) {
        CASE(CL_R)
        CASE(CL_A)
        CASE(CL_DEPTH)
        CASE(CL_LUMINANCE)
        CASE(CL_INTENSITY)
        CASE(CL_RG)
        CASE(CL_RA)
        CASE(CL_Rx)
        CASE(CL_RGB)
        CASE(CL_RGx)
        CASE(CL_RGBA)
        CASE(CL_ARGB)
        CASE(CL_BGRA)
        CASE(CL_ABGR)
        CASE(CL_RGBx)
        CASE(CL_sRGB)
        CASE(CL_sRGBA)
        CASE(CL_sBGRA)
        CASE(CL_sRGBx)
    }
    return "Unknown channel order";
}

std::string cl_channel_type_to_string(cl_channel_type type) {
    switch (type) {
        CASE(CL_SNORM_INT8)
        CASE(CL_SNORM_INT16)
        CASE(CL_UNORM_INT8)
        CASE(CL_UNORM_INT16)
        CASE(CL_UNORM_SHORT_565)
        CASE(CL_UNORM_SHORT_555)
        CASE(CL_UNORM_INT_101010)
        CASE(CL_UNORM_INT_101010_2)
        CASE(CL_SIGNED_INT8)
        CASE(CL_SIGNED_INT16)
        CASE(CL_SIGNED_INT32)
        CASE(CL_UNSIGNED_INT8)
        CASE(CL_UNSIGNED_INT16)
        CASE(CL_UNSIGNED_INT32)
        CASE(CL_HALF_FLOAT)
        CASE(CL_FLOAT)
    }
    return "Unknown channel type";
}

const char* cl_command_type_to_string(cl_command_type type) {
    switch (type) {
        CASE(CL_COMMAND_NDRANGE_KERNEL);
        CASE(CL_COMMAND_TASK);
        CASE(CL_COMMAND_NATIVE_KERNEL);
        CASE(CL_COMMAND_READ_BUFFER);
        CASE(CL_COMMAND_WRITE_BUFFER);
        CASE(CL_COMMAND_COPY_BUFFER);
        CASE(CL_COMMAND_READ_IMAGE);
        CASE(CL_COMMAND_WRITE_IMAGE);
        CASE(CL_COMMAND_COPY_IMAGE);
        CASE(CL_COMMAND_COPY_BUFFER_TO_IMAGE);
        CASE(CL_COMMAND_COPY_IMAGE_TO_BUFFER);
        CASE(CL_COMMAND_MAP_BUFFER);
        CASE(CL_COMMAND_MAP_IMAGE);
        CASE(CL_COMMAND_UNMAP_MEM_OBJECT);
        CASE(CL_COMMAND_MARKER);
        CASE(CL_COMMAND_ACQUIRE_GL_OBJECTS);
        CASE(CL_COMMAND_RELEASE_GL_OBJECTS);
        CASE(CL_COMMAND_SEMAPHORE_WAIT_KHR);
        CASE(CL_COMMAND_SEMAPHORE_SIGNAL_KHR);
        CASE(CLVK_COMMAND_BATCH);
        CASE(CLVK_COMMAND_IMAGE_INIT);
    default:
        return "CL_COMMAND_UNKNOWN";
    }
}

const char* cl_device_type_to_string(cl_device_type type) {
    switch (type) {
        CASE(CL_DEVICE_TYPE_CPU);
        CASE(CL_DEVICE_TYPE_GPU);
        CASE(CL_DEVICE_TYPE_ACCELERATOR);
        CASE(CL_DEVICE_TYPE_CUSTOM);
        CASE(CL_DEVICE_TYPE_DEFAULT);
        CASE(CL_DEVICE_TYPE_ALL);
    default:
        return "CL_DEVICE_TYPE_UNKNOWN";
    }
}

const char* cl_command_execution_status_to_string(cl_int status) {
    switch (status) {
        CASE(CL_COMPLETE);
        CASE(CL_RUNNING);
        CASE(CL_SUBMITTED);
        CASE(CL_QUEUED);
    default:
        return "CL_COMMAND_EXECUTION_STATUS_UNKNOWN";
    }
}

const char* spirv_capability_to_string(spv::Capability capability) {
    switch (capability) {
        CASE(spv::CapabilityMatrix);
        CASE(spv::CapabilityShader);
        CASE(spv::CapabilityGeometry);
        CASE(spv::CapabilityTessellation);
        CASE(spv::CapabilityAddresses);
        CASE(spv::CapabilityLinkage);
        CASE(spv::CapabilityKernel);
        CASE(spv::CapabilityVector16);
        CASE(spv::CapabilityFloat16Buffer);
        CASE(spv::CapabilityFloat16);
        CASE(spv::CapabilityFloat64);
        CASE(spv::CapabilityInt64);
        CASE(spv::CapabilityInt64Atomics);
        CASE(spv::CapabilityImageBasic);
        CASE(spv::CapabilityImageReadWrite);
        CASE(spv::CapabilityImageMipmap);
        CASE(spv::CapabilityPipes);
        CASE(spv::CapabilityGroups);
        CASE(spv::CapabilityDeviceEnqueue);
        CASE(spv::CapabilityLiteralSampler);
        CASE(spv::CapabilityAtomicStorage);
        CASE(spv::CapabilityInt16);
        CASE(spv::CapabilityTessellationPointSize);
        CASE(spv::CapabilityGeometryPointSize);
        CASE(spv::CapabilityImageGatherExtended);
        CASE(spv::CapabilityStorageImageMultisample);
        CASE(spv::CapabilityUniformBufferArrayDynamicIndexing);
        CASE(spv::CapabilitySampledImageArrayDynamicIndexing);
        CASE(spv::CapabilityStorageBufferArrayDynamicIndexing);
        CASE(spv::CapabilityStorageImageArrayDynamicIndexing);
        CASE(spv::CapabilityClipDistance);
        CASE(spv::CapabilityCullDistance);
        CASE(spv::CapabilityImageCubeArray);
        CASE(spv::CapabilitySampleRateShading);
        CASE(spv::CapabilityImageRect);
        CASE(spv::CapabilitySampledRect);
        CASE(spv::CapabilityGenericPointer);
        CASE(spv::CapabilityInt8);
        CASE(spv::CapabilityInputAttachment);
        CASE(spv::CapabilitySparseResidency);
        CASE(spv::CapabilityMinLod);
        CASE(spv::CapabilitySampled1D);
        CASE(spv::CapabilityImage1D);
        CASE(spv::CapabilitySampledCubeArray);
        CASE(spv::CapabilitySampledBuffer);
        CASE(spv::CapabilityImageBuffer);
        CASE(spv::CapabilityImageMSArray);
        CASE(spv::CapabilityStorageImageExtendedFormats);
        CASE(spv::CapabilityImageQuery);
        CASE(spv::CapabilityDerivativeControl);
        CASE(spv::CapabilityInterpolationFunction);
        CASE(spv::CapabilityTransformFeedback);
        CASE(spv::CapabilityGeometryStreams);
        CASE(spv::CapabilityStorageImageReadWithoutFormat);
        CASE(spv::CapabilityStorageImageWriteWithoutFormat);
        CASE(spv::CapabilityMultiViewport);
        CASE(spv::CapabilitySubgroupDispatch);
        CASE(spv::CapabilityNamedBarrier);
        CASE(spv::CapabilityPipeStorage);
        CASE(spv::CapabilityGroupNonUniform);
        CASE(spv::CapabilityGroupNonUniformVote);
        CASE(spv::CapabilityGroupNonUniformArithmetic);
        CASE(spv::CapabilityGroupNonUniformBallot);
        CASE(spv::CapabilityGroupNonUniformShuffle);
        CASE(spv::CapabilityGroupNonUniformShuffleRelative);
        CASE(spv::CapabilityGroupNonUniformClustered);
        CASE(spv::CapabilityGroupNonUniformQuad);
        CASE(spv::CapabilityShaderLayer);
        CASE(spv::CapabilityShaderViewportIndex);
        CASE(spv::CapabilityUniformDecoration);
        CASE(spv::CapabilityCoreBuiltinsARM);
        CASE(spv::CapabilityFragmentShadingRateKHR);
        CASE(spv::CapabilitySubgroupBallotKHR);
        CASE(spv::CapabilityDrawParameters);
        CASE(spv::CapabilityWorkgroupMemoryExplicitLayoutKHR);
        CASE(spv::CapabilityWorkgroupMemoryExplicitLayout8BitAccessKHR);
        CASE(spv::CapabilityWorkgroupMemoryExplicitLayout16BitAccessKHR);
        CASE(spv::CapabilitySubgroupVoteKHR);
        CASE(spv::CapabilityStorageBuffer16BitAccess);
        CASE(spv::CapabilityStorageUniform16);
        CASE(spv::CapabilityStoragePushConstant16);
        CASE(spv::CapabilityStorageInputOutput16);
        CASE(spv::CapabilityDeviceGroup);
        CASE(spv::CapabilityMultiView);
        CASE(spv::CapabilityVariablePointersStorageBuffer);
        CASE(spv::CapabilityVariablePointers);
        CASE(spv::CapabilityAtomicStorageOps);
        CASE(spv::CapabilitySampleMaskPostDepthCoverage);
        CASE(spv::CapabilityStorageBuffer8BitAccess);
        CASE(spv::CapabilityUniformAndStorageBuffer8BitAccess);
        CASE(spv::CapabilityStoragePushConstant8);
        CASE(spv::CapabilityDenormPreserve);
        CASE(spv::CapabilityDenormFlushToZero);
        CASE(spv::CapabilitySignedZeroInfNanPreserve);
        CASE(spv::CapabilityRoundingModeRTE);
        CASE(spv::CapabilityRoundingModeRTZ);
        CASE(spv::CapabilityRayQueryProvisionalKHR);
        CASE(spv::CapabilityRayQueryKHR);
        CASE(spv::CapabilityRayTraversalPrimitiveCullingKHR);
        CASE(spv::CapabilityRayTracingKHR);
        CASE(spv::CapabilityTextureSampleWeightedQCOM);
        CASE(spv::CapabilityTextureBoxFilterQCOM);
        CASE(spv::CapabilityTextureBlockMatchQCOM);
        CASE(spv::CapabilityFloat16ImageAMD);
        CASE(spv::CapabilityImageGatherBiasLodAMD);
        CASE(spv::CapabilityFragmentMaskAMD);
        CASE(spv::CapabilityStencilExportEXT);
        CASE(spv::CapabilityImageReadWriteLodAMD);
        CASE(spv::CapabilityInt64ImageEXT);
        CASE(spv::CapabilityShaderClockKHR);
        CASE(spv::CapabilitySampleMaskOverrideCoverageNV);
        CASE(spv::CapabilityGeometryShaderPassthroughNV);
        CASE(spv::CapabilityShaderViewportIndexLayerEXT);
        CASE(spv::CapabilityShaderViewportMaskNV);
        CASE(spv::CapabilityShaderStereoViewNV);
        CASE(spv::CapabilityPerViewAttributesNV);
        CASE(spv::CapabilityFragmentFullyCoveredEXT);
        CASE(spv::CapabilityMeshShadingNV);
        CASE(spv::CapabilityImageFootprintNV);
        CASE(spv::CapabilityMeshShadingEXT);
        CASE(spv::CapabilityFragmentBarycentricKHR);
        CASE(spv::CapabilityComputeDerivativeGroupQuadsNV);
        CASE(spv::CapabilityFragmentDensityEXT);
        CASE(spv::CapabilityGroupNonUniformPartitionedNV);
        CASE(spv::CapabilityShaderNonUniform);
        CASE(spv::CapabilityRuntimeDescriptorArray);
        CASE(spv::CapabilityInputAttachmentArrayDynamicIndexing);
        CASE(spv::CapabilityUniformTexelBufferArrayDynamicIndexing);
        CASE(spv::CapabilityStorageTexelBufferArrayDynamicIndexing);
        CASE(spv::CapabilityUniformBufferArrayNonUniformIndexing);
        CASE(spv::CapabilitySampledImageArrayNonUniformIndexing);
        CASE(spv::CapabilityStorageBufferArrayNonUniformIndexing);
        CASE(spv::CapabilityStorageImageArrayNonUniformIndexing);
        CASE(spv::CapabilityInputAttachmentArrayNonUniformIndexing);
        CASE(spv::CapabilityUniformTexelBufferArrayNonUniformIndexing);
        CASE(spv::CapabilityStorageTexelBufferArrayNonUniformIndexing);
        CASE(spv::CapabilityRayTracingNV);
        CASE(spv::CapabilityRayTracingMotionBlurNV);
        CASE(spv::CapabilityVulkanMemoryModel);
        CASE(spv::CapabilityVulkanMemoryModelDeviceScope);
        CASE(spv::CapabilityPhysicalStorageBufferAddresses);
        CASE(spv::CapabilityComputeDerivativeGroupLinearNV);
        CASE(spv::CapabilityRayTracingProvisionalKHR);
        CASE(spv::CapabilityCooperativeMatrixNV);
        CASE(spv::CapabilityFragmentShaderSampleInterlockEXT);
        CASE(spv::CapabilityFragmentShaderShadingRateInterlockEXT);
        CASE(spv::CapabilityShaderSMBuiltinsNV);
        CASE(spv::CapabilityFragmentShaderPixelInterlockEXT);
        CASE(spv::CapabilityDemoteToHelperInvocation);
        CASE(spv::CapabilityRayTracingOpacityMicromapEXT);
        CASE(spv::CapabilityShaderInvocationReorderNV);
        CASE(spv::CapabilityBindlessTextureNV);
        CASE(spv::CapabilitySubgroupShuffleINTEL);
        CASE(spv::CapabilitySubgroupBufferBlockIOINTEL);
        CASE(spv::CapabilitySubgroupImageBlockIOINTEL);
        CASE(spv::CapabilitySubgroupImageMediaBlockIOINTEL);
        CASE(spv::CapabilityRoundToInfinityINTEL);
        CASE(spv::CapabilityFloatingPointModeINTEL);
        CASE(spv::CapabilityIntegerFunctions2INTEL);
        CASE(spv::CapabilityFunctionPointersINTEL);
        CASE(spv::CapabilityIndirectReferencesINTEL);
        CASE(spv::CapabilityAsmINTEL);
        CASE(spv::CapabilityAtomicFloat32MinMaxEXT);
        CASE(spv::CapabilityAtomicFloat64MinMaxEXT);
        CASE(spv::CapabilityAtomicFloat16MinMaxEXT);
        CASE(spv::CapabilityVectorComputeINTEL);
        CASE(spv::CapabilityVectorAnyINTEL);
        CASE(spv::CapabilityExpectAssumeKHR);
        CASE(spv::CapabilitySubgroupAvcMotionEstimationINTEL);
        CASE(spv::CapabilitySubgroupAvcMotionEstimationIntraINTEL);
        CASE(spv::CapabilitySubgroupAvcMotionEstimationChromaINTEL);
        CASE(spv::CapabilityVariableLengthArrayINTEL);
        CASE(spv::CapabilityFunctionFloatControlINTEL);
        CASE(spv::CapabilityFPGAMemoryAttributesINTEL);
        CASE(spv::CapabilityFPFastMathModeINTEL);
        CASE(spv::CapabilityArbitraryPrecisionIntegersINTEL);
        CASE(spv::CapabilityArbitraryPrecisionFloatingPointINTEL);
        CASE(spv::CapabilityUnstructuredLoopControlsINTEL);
        CASE(spv::CapabilityFPGALoopControlsINTEL);
        CASE(spv::CapabilityKernelAttributesINTEL);
        CASE(spv::CapabilityFPGAKernelAttributesINTEL);
        CASE(spv::CapabilityFPGAMemoryAccessesINTEL);
        CASE(spv::CapabilityFPGAClusterAttributesINTEL);
        CASE(spv::CapabilityLoopFuseINTEL);
        CASE(spv::CapabilityFPGADSPControlINTEL);
        CASE(spv::CapabilityMemoryAccessAliasingINTEL);
        CASE(spv::CapabilityFPGAInvocationPipeliningAttributesINTEL);
        CASE(spv::CapabilityFPGABufferLocationINTEL);
        CASE(spv::CapabilityArbitraryPrecisionFixedPointINTEL);
        CASE(spv::CapabilityUSMStorageClassesINTEL);
        CASE(spv::CapabilityRuntimeAlignedAttributeINTEL);
        CASE(spv::CapabilityIOPipesINTEL);
        CASE(spv::CapabilityBlockingPipesINTEL);
        CASE(spv::CapabilityFPGARegINTEL);
        CASE(spv::CapabilityDotProductInputAll);
        CASE(spv::CapabilityDotProductInput4x8Bit);
        CASE(spv::CapabilityDotProductInput4x8BitPacked);
        CASE(spv::CapabilityDotProduct);
        CASE(spv::CapabilityRayCullMaskKHR);
        CASE(spv::CapabilityBitInstructions);
        CASE(spv::CapabilityGroupNonUniformRotateKHR);
        CASE(spv::CapabilityAtomicFloat32AddEXT);
        CASE(spv::CapabilityAtomicFloat64AddEXT);
        CASE(spv::CapabilityLongCompositesINTEL);
        CASE(spv::CapabilityOptNoneINTEL);
        CASE(spv::CapabilityAtomicFloat16AddEXT);
        CASE(spv::CapabilityDebugInfoModuleINTEL);
        CASE(spv::CapabilityBFloat16ConversionINTEL);
        CASE(spv::CapabilitySplitBarrierINTEL);
        CASE(spv::CapabilityFPGAKernelAttributesv2INTEL);
        CASE(spv::CapabilityFPGALatencyControlINTEL);
        CASE(spv::CapabilityFPGAArgumentInterfacesINTEL);
        CASE(spv::CapabilityGroupUniformArithmeticKHR);
        CASE(spv::CapabilityMax);
    default:
        return "unknown SPIRV Capability";
    }
}

const char* queue_global_priority_to_string(VkQueueGlobalPriorityKHR priority) {
    switch (priority) {
        CASE(VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR);
        CASE(VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR);
        CASE(VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR);
        CASE(VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR);
    default:
        return "VK_QUEUE_GLOBAL_PRIORITY_UNKNOWN";
    }
}
