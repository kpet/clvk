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

#include "cl_headers.hpp"
#include "init.hpp"
#include "kernel.hpp"
#include "memory.hpp"
#include "objects.hpp"
#include "program.hpp"
#include "queue.hpp"
#include "utils.hpp"

#define LOG_API_CALL(fmt, ...) cvk_info_fn(fmt,  __VA_ARGS__)

// Validation functions
namespace {

bool is_valid_platform(cl_platform_id platform) {
    return platform != nullptr;
}

bool is_valid_device(cl_device_id device) {
    return device != nullptr;
}

bool is_valid_context(cl_context context) {
    return context != nullptr;
}

bool is_valid_program(cl_program program) {
    return program != nullptr;
}

bool is_valid_kernel(cl_kernel kernel) {
    return kernel != nullptr;
}

bool is_valid_sampler(cl_sampler sampler) {
    return sampler != nullptr;
}

bool is_valid_mem_object(cl_mem mem) {
    return mem != nullptr;
}

bool is_valid_buffer(cl_mem mem) {
    return is_valid_mem_object(mem) && mem->is_buffer_type();
}

bool is_valid_image(cl_mem mem) {
    return is_valid_mem_object(mem) && mem->is_image_type();
}

bool is_valid_command_queue(cl_command_queue queue) {
    return queue != nullptr;
}

bool is_valid_event(cl_event event) {
    return event != nullptr;
}

bool is_same_context(cl_command_queue queue, cl_mem mem) {
    return queue->context() == mem->context();
}

bool is_same_context(cl_command_queue queue, cl_uint num_events, const cl_event *event_list) {
    for (cl_uint i = 0; i < num_events; i++) {
        if (queue->context() != event_list[i]->context()) {
            return false;
        }
    }

    return true;
}

bool map_flags_are_valid(cl_map_flags flags) {
    if ((flags & CL_MAP_WRITE_INVALIDATE_REGION) &&
        (flags & (CL_MAP_READ | CL_MAP_WRITE))) {
        return false;
    }
    return true;
}

} // namespace

// Platform API
cl_int
clGetPlatformIDs(
    cl_uint num_entries,
    cl_platform_id *platforms,
    cl_uint *num_platforms
){
    LOG_API_CALL("num_entries = %u, platforms = %p, num_platforms = %p", num_entries, platforms, num_platforms);

    if ((num_platforms == nullptr) && (platforms == nullptr)) {
        return CL_INVALID_VALUE;
    }

    if ((num_entries == 0) && (platforms != nullptr)) {
        return CL_INVALID_VALUE;
    }

    if (platforms != nullptr) {
        platforms[0] = gPlatform;
    }

    if (num_platforms != nullptr) {
        *num_platforms = 1;
    }

    return CL_SUCCESS;
}

cl_int
clGetPlatformInfo(
    cl_platform_id platform,
    cl_platform_info param_name,
    size_t param_value_size,
    void *param_value,
    size_t *param_value_size_ret
){
    LOG_API_CALL("platform = %p, param_name = %u, param_value_size = %zu, param_value = %p, param_value_size_ret = %p", platform, param_name, param_value_size, param_value, param_value_size_ret);
    cl_int ret = CL_SUCCESS;

    size_t size_ret = 0;
    const void *copy_ptr = nullptr;

    const string platform_name{"clvk"};
    const string platform_version{"OpenCL 1.2 clvk"};
    const string platform_vendor{"clvk"};
    const string platform_profile{"FULL_PROFILE"};
    const string platform_extensions{""};

    switch(param_name)
    {
        case CL_PLATFORM_NAME:
            copy_ptr = platform_name.c_str();
            size_ret = platform_name.size_with_null();
            break;
        case CL_PLATFORM_VERSION:
            copy_ptr = platform_version.c_str();
            size_ret = platform_version.size_with_null();
            break;
        case CL_PLATFORM_VENDOR:
            copy_ptr = platform_vendor.c_str();
            size_ret = platform_vendor.size_with_null();
            break;
        case CL_PLATFORM_PROFILE:
            copy_ptr = platform_profile.c_str();
            size_ret = platform_profile.size_with_null();
            break;
        case CL_PLATFORM_EXTENSIONS:
            copy_ptr = platform_extensions.c_str();
            size_ret = platform_extensions.size_with_null();
            break;
        default:
            ret = CL_INVALID_VALUE;
            break;
    }

    if ((param_value != nullptr) && (copy_ptr != nullptr)) {
        memcpy(param_value, copy_ptr, std::min(param_value_size, size_ret));
    }

    if (param_value_size_ret != nullptr) {
        *param_value_size_ret = size_ret;
    }

    return ret;
}

const std::unordered_map<std::string, void*> gExtensionEntrypoints = {
    { "clCreateProgramWithILKHR", reinterpret_cast<void*>(clCreateProgramWithILKHR) },
};

void* cvk_get_extension_function_pointer(const char *funcname)
{
    if (gExtensionEntrypoints.find(funcname) != gExtensionEntrypoints.end()) {
        return gExtensionEntrypoints.at(funcname);
    } else {
        return nullptr;
    }
}

void* clGetExtensionFunctionAddressForPlatform(
    cl_platform_id platform,
    const char    *funcname
){
    LOG_API_CALL("platform = %p, funcname = '%s'",
                 platform, funcname);

    if (platform == nullptr) {
        return nullptr;
    }

    return cvk_get_extension_function_pointer(funcname);
}

void* clGetExtensionFunctionAddress(
    const char *funcname
){
    LOG_API_CALL("funcname = '%s'", funcname);

    return cvk_get_extension_function_pointer(funcname);
}

// Device APIs
cl_int
clGetDeviceIDs(
    cl_platform_id platform,
    cl_device_type device_type,
    cl_uint num_entries,
    cl_device_id *devices, 
    cl_uint *num_devices
){
    LOG_API_CALL("platform = %p, device_type = %lu, num_entries = %u, devices = %p, num_devices = %p",
                 platform, device_type, num_entries, devices, num_devices);

    if (platform == nullptr) {
        platform = gPlatform;
    } else if (platform != gPlatform) {
        return CL_INVALID_PLATFORM;
    }

    if ((num_entries == 0) && (devices != nullptr)) {
        return CL_INVALID_VALUE;
    }

    if ((num_devices == nullptr) && (devices == nullptr)) {
        return CL_INVALID_VALUE;
    }

    cl_uint num = 0;

    for (auto dev : platform->devices) {
        if ((device_type == CL_DEVICE_TYPE_DEFAULT) || (device_type == CL_DEVICE_TYPE_ALL) || (dev->type() == device_type)) {
            if ((devices != nullptr) && (num < num_entries)) {
                devices[num] = dev;
            }
            num++;
        }
    }

    if (num_devices != nullptr) {
        *num_devices = num;
    }

    if (num == 0) {
        return CL_DEVICE_NOT_FOUND;
    }

    return CL_SUCCESS;
}

cl_int
clGetDeviceInfo(
    cl_device_id    device,
    cl_device_info  param_name, 
    size_t          param_value_size, 
    void *          param_value,
    size_t *        param_value_size_ret
){
    LOG_API_CALL("device = %p, param_name = %d, size = %zu, value = %p, size_ret = %p",
                 device, param_name, param_value_size, param_value, param_value_size_ret);
    cl_int ret = CL_SUCCESS;

    size_t size_ret = 0;
    const void *copy_ptr = nullptr;
    size_t val_sizet;
    cl_uint val_uint;
    string val_string;
    cl_device_type val_devicetype;
    cl_bool val_bool;
    cl_device_fp_config val_fpconfig;
    size_t work_item_sizes[3];
    cl_device_mem_cache_type val_cache_type;
    cl_ulong val_ulong;
    cl_device_local_mem_type val_local_mem_type;
    cl_device_partition_property val_partition_property;
    cl_device_affinity_domain val_affinity_domain;
    cl_device_exec_capabilities val_exec_capabilities;
    cl_command_queue_properties val_queue_properties;
    cl_platform_id val_platform;
    cl_device_id val_deviceid;

    if (!is_valid_device(device)) {
        return CL_INVALID_DEVICE;
    }

    switch(param_name)
    {
        case CL_DEVICE_PLATFORM:
            val_platform = gPlatform;
            copy_ptr = &val_platform;
            size_ret = sizeof(val_platform);
            break;
        case CL_DEVICE_TYPE:
            val_devicetype = device->type();
            copy_ptr = &val_devicetype;
            size_ret = sizeof(val_devicetype);
            break;
        case CL_DEVICE_NAME:
            val_string = device->name();
            copy_ptr = val_string.c_str();
            size_ret = val_string.size_with_null();
            break;
        case CL_DEVICE_VENDOR:
            val_string = "FIXME";
            copy_ptr = val_string.c_str();
            size_ret = val_string.size_with_null();
            break;
        case CL_DEVICE_VENDOR_ID:
            val_uint = device->vendor_id();
            copy_ptr = &val_uint;
            size_ret = sizeof(val_uint);
            break;
        case CL_DRIVER_VERSION:
            val_string = "1.2 ";
            val_string += device->version_string();
            copy_ptr = val_string.c_str();
            size_ret = val_string.size_with_null();
            break;
        case CL_DEVICE_VERSION:
            val_string = "OpenCL 1.2 ";
            val_string += device->version_string();
            copy_ptr = val_string.c_str();
            size_ret = val_string.size_with_null();
            break;
        case CL_DEVICE_OPENCL_C_VERSION:
            val_string = "OpenCL C 1.2 ";
            val_string += device->version_string();
            copy_ptr = val_string.c_str();
            size_ret = val_string.size_with_null();
            break;
        case CL_DEVICE_PROFILE:
            val_string = "FULL_PROFILE";
            copy_ptr = val_string.c_str();
            size_ret = val_string.size_with_null();
            break;
        case CL_DEVICE_BUILT_IN_KERNELS:
            val_string = "";
            copy_ptr = val_string.c_str();
            size_ret = val_string.size_with_null();
            break;
        case CL_DEVICE_EXTENSIONS:
            val_string = "cl_khr_global_int32_base_atomics "
                         "cl_khr_global_int32_extended_atomics "
                         "cl_khr_local_int32_base_atomics "
                         "cl_khr_local_int32_extended_atomics "
                         "cl_khr_il_program "
                         "cl_khr_byte_addressable_store";
            copy_ptr = val_string.c_str();
            size_ret = val_string.size_with_null();
            break;
        case CL_DEVICE_AVAILABLE:
        case CL_DEVICE_COMPILER_AVAILABLE:
        case CL_DEVICE_LINKER_AVAILABLE:
            val_bool = CL_TRUE;
            copy_ptr = &val_bool;
            size_ret = sizeof(val_bool);
            break;
        case CL_DEVICE_IMAGE_SUPPORT:
            val_bool = device->supports_images();
            copy_ptr = &val_bool;
            size_ret = sizeof(val_bool);
            break;
        case CL_DEVICE_IMAGE2D_MAX_WIDTH:
        case CL_DEVICE_IMAGE2D_MAX_HEIGHT:
            val_sizet = device->vulkan_limits().maxImageDimension2D;
            copy_ptr = &val_sizet;
            size_ret = sizeof(val_sizet);
            break;
        case CL_DEVICE_IMAGE3D_MAX_WIDTH:
        case CL_DEVICE_IMAGE3D_MAX_HEIGHT:
        case CL_DEVICE_IMAGE3D_MAX_DEPTH:
            val_sizet = device->vulkan_limits().maxImageDimension3D;
            copy_ptr = &val_sizet;
            size_ret = sizeof(val_sizet);
            break;
        case CL_DEVICE_SINGLE_FP_CONFIG:
            val_fpconfig = device->fp_config(CL_DEVICE_SINGLE_FP_CONFIG);
            copy_ptr = &val_fpconfig;
            size_ret = sizeof(val_fpconfig);
            break;
        case CL_DEVICE_DOUBLE_FP_CONFIG:
            val_fpconfig = device->fp_config(CL_DEVICE_DOUBLE_FP_CONFIG);
            copy_ptr = &val_fpconfig;
            size_ret = sizeof(val_fpconfig);
            break;
        case CL_DEVICE_ADDRESS_BITS:
            val_uint = 32; // FIXME
            copy_ptr = &val_uint;
            size_ret = sizeof(val_uint);
            break;
        case CL_DEVICE_MEM_BASE_ADDR_ALIGN:
            val_uint = device->mem_base_addr_align();
            copy_ptr = &val_uint;
            size_ret = sizeof(val_uint);
            break;
        case CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE:
            val_uint = 128; // FIXME do better?
            copy_ptr = &val_uint,
            size_ret = sizeof(val_uint);
            break;
        case CL_DEVICE_GLOBAL_MEM_CACHE_TYPE:
            val_cache_type = CL_NONE; // FIXME
            copy_ptr = &val_cache_type;
            size_ret = sizeof(val_cache_type);
            break;
        case CL_DEVICE_GLOBAL_MEM_CACHE_SIZE:
            val_ulong = 0; // FIXME
            copy_ptr = &val_ulong;
            size_ret = sizeof(val_ulong);
            break;
        case CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE:
            val_uint = device->vulkan_limits().nonCoherentAtomSize;
            copy_ptr = &val_uint;
            size_ret = sizeof(val_uint);
            break;
        case CL_DEVICE_ERROR_CORRECTION_SUPPORT:
            val_bool = CL_FALSE; // FIXME
            copy_ptr = &val_bool;
            size_ret = sizeof(val_bool);
            break;
        case CL_DEVICE_ENDIAN_LITTLE:
            val_bool = CL_TRUE; // FIXME
            copy_ptr = &val_bool;
            size_ret = sizeof(val_bool);
            break;
        case CL_DEVICE_HOST_UNIFIED_MEMORY:
            val_bool = device->has_host_unified_memory();
            copy_ptr = &val_bool;
            size_ret = sizeof(val_bool);
            break;
        case CL_DEVICE_MAX_WORK_GROUP_SIZE:
            val_sizet = device->vulkan_limits().maxComputeWorkGroupInvocations;
            copy_ptr = &val_sizet;
            size_ret = sizeof(val_sizet);
            break;
        case CL_DEVICE_MAX_COMPUTE_UNITS:
            val_uint = 1; // TODO can we do any better here?
            copy_ptr = &val_uint;
            size_ret = sizeof(val_uint);
            break;
        case CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:
            val_uint = 3;
            copy_ptr = &val_uint;
            size_ret = sizeof(val_uint);
            break;
        case CL_DEVICE_MAX_WORK_ITEM_SIZES:
            work_item_sizes[0] = device->vulkan_limits().maxComputeWorkGroupSize[0];
            work_item_sizes[1] = device->vulkan_limits().maxComputeWorkGroupSize[1];
            work_item_sizes[2] = device->vulkan_limits().maxComputeWorkGroupSize[2];
            copy_ptr = work_item_sizes;
            size_ret = sizeof(work_item_sizes);
            break;
        case CL_DEVICE_MAX_PARAMETER_SIZE:
            val_sizet = 1024; // FIXME this is the minimum, revisit when looking into push constants
            copy_ptr = &val_sizet;
            size_ret = sizeof(val_sizet);
            break;
        case CL_DEVICE_MAX_CONSTANT_ARGS:
            val_uint = 8; // TODO be smarter
            copy_ptr = &val_uint;
            size_ret = sizeof(val_uint);
            break;
        case CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: // TODO be smarter
            val_ulong = 64*1024;
            copy_ptr = &val_ulong;
            size_ret = sizeof(val_ulong);
            break;
        case CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR:
        case CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT:
        case CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT:
        case CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG:
        case CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT:
        case CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE:
        case CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF:
        case CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR:
        case CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT:
        case CL_DEVICE_NATIVE_VECTOR_WIDTH_INT:
        case CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG:
        case CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT:
        case CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE:
        case CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF:
            val_uint = 1; // FIXME can we do better?
            copy_ptr = &val_uint;
            size_ret = sizeof(val_uint);
            break;
        case CL_DEVICE_PROFILING_TIMER_RESOLUTION:
            val_sizet = 1;
            copy_ptr = &val_sizet;
            size_ret = sizeof(val_sizet);
            break;
        case CL_DEVICE_GLOBAL_MEM_SIZE:
            val_ulong = device->memory_size();
            copy_ptr = &val_ulong;
            size_ret = sizeof(val_ulong);
            break;
        case CL_DEVICE_MAX_MEM_ALLOC_SIZE:
            val_ulong = device->max_alloc_size();
            copy_ptr = &val_ulong;
            size_ret = sizeof(val_ulong);
            break;
        case CL_DEVICE_LOCAL_MEM_SIZE:
            val_ulong = device->vulkan_limits().maxComputeSharedMemorySize;
            copy_ptr = &val_ulong;
            size_ret = sizeof(val_ulong);
            break;
        case CL_DEVICE_LOCAL_MEM_TYPE:
            val_local_mem_type = CL_LOCAL; // FIXME try to be a bit smarter
            copy_ptr = &val_local_mem_type;
            size_ret = sizeof(val_local_mem_type);
            break;
        case CL_DEVICE_MAX_CLOCK_FREQUENCY:
            val_uint = 0; // FIXME can we do better?
            copy_ptr = &val_uint;
            size_ret = sizeof(val_uint);
            break;
        case CL_DEVICE_PARTITION_MAX_SUB_DEVICES:
            val_uint = 0; // TODO support
            copy_ptr = &val_uint;
            size_ret = sizeof(val_uint);
            break;
        case CL_DEVICE_PARTITION_PROPERTIES:
        case CL_DEVICE_PARTITION_TYPE:
            val_partition_property = 0; // TODO
            copy_ptr = &val_partition_property;
            size_ret = sizeof(val_partition_property);
            break;
        case CL_DEVICE_PARTITION_AFFINITY_DOMAIN:
            val_affinity_domain = 0; // TODO
            copy_ptr = &val_affinity_domain;
            size_ret = sizeof(val_affinity_domain);
            break;
        case CL_DEVICE_EXECUTION_CAPABILITIES:
            val_exec_capabilities = CL_EXEC_KERNEL;
            copy_ptr = &val_exec_capabilities;
            size_ret = sizeof(val_exec_capabilities);
            break;
        case CL_DEVICE_QUEUE_PROPERTIES:
            val_queue_properties = CL_QUEUE_PROFILING_ENABLE;
            copy_ptr = &val_queue_properties;
            size_ret = sizeof(val_queue_properties);
            break;
        case CL_DEVICE_REFERENCE_COUNT:
            val_uint = 1; // FIXME partitioning
            copy_ptr = &val_uint;
            size_ret = sizeof(val_uint);
            break;
        case CL_DEVICE_PARENT_DEVICE:
            val_deviceid = nullptr; // TODO partitioning
            copy_ptr = &val_deviceid;
            size_ret = sizeof(val_deviceid);
            break;
        case CL_DEVICE_PREFERRED_INTEROP_USER_SYNC:
            val_bool = CL_TRUE;
            copy_ptr = &val_bool;
            size_ret = sizeof(val_bool);
            break;
        case CL_DEVICE_PRINTF_BUFFER_SIZE:
            val_sizet = 1024 * 1024; // FIXME
            copy_ptr = &val_sizet;
            size_ret = sizeof(val_sizet);
            break;
        case CL_DEVICE_MAX_SAMPLERS:
            val_uint = device->vulkan_limits().maxPerStageDescriptorSamplers;
            copy_ptr = &val_uint;
            size_ret = sizeof(val_uint);
            break;
        case CL_DEVICE_IMAGE_MAX_BUFFER_SIZE:
            val_sizet = device->vulkan_limits().maxImageDimension1D;
            copy_ptr = &val_sizet;
            size_ret = sizeof(val_sizet);
            break;
        case CL_DEVICE_IMAGE_MAX_ARRAY_SIZE:
            val_sizet = device->vulkan_limits().maxImageArrayLayers;
            copy_ptr = &val_sizet;
            size_ret = sizeof(val_sizet);
            break;
        case CL_DEVICE_MAX_READ_IMAGE_ARGS:
            val_uint = device->vulkan_limits().maxPerStageDescriptorSampledImages;
            copy_ptr = &val_uint;
            size_ret = sizeof(val_uint);
            break;
        case CL_DEVICE_MAX_WRITE_IMAGE_ARGS:
            val_uint = device->vulkan_limits().maxPerStageDescriptorStorageImages;
            copy_ptr = &val_uint;
            size_ret = sizeof(val_uint);
            break;
        case CL_DEVICE_IL_VERSION_KHR:
            val_string = "SPIR-V_1.0";
            copy_ptr = val_string.c_str();
            size_ret = val_string.size_with_null();
            break;
        default:
            ret = CL_INVALID_VALUE;
            break;
    }

    if ((param_value != nullptr) && (copy_ptr != nullptr)) {
        memcpy(param_value, copy_ptr, std::min(param_value_size, size_ret));
    }

    if (param_value_size_ret != nullptr) {
        *param_value_size_ret = size_ret;
    }

    return ret;
}

cl_int clCreateSubDevices(
    cl_device_id                        in_device,
    const cl_device_partition_property *properties,
    cl_uint                             num_devices,
    cl_device_id                       *out_devices,
    cl_uint                            *num_devices_ret
){
    LOG_API_CALL("in_device = %p, properties = %p, num_devices = %u, "
                 "out_devices = %p, num_devices_ret = %p",
                 in_device, properties, num_devices, out_devices, num_devices_ret);

    return CL_INVALID_OPERATION;
}

cl_int clRetainDevice(
    cl_device_id device
){
    LOG_API_CALL("device = %p", device);

    if (!is_valid_device(device)) {
        return CL_INVALID_DEVICE;
    }

    return CL_SUCCESS;
}

cl_int clReleaseDevice(
    cl_device_id device
){
    LOG_API_CALL("device = %p", device);

    if (!is_valid_device(device)) {
        return CL_INVALID_DEVICE;
    }

    return CL_SUCCESS;
}

// Context APIs
cl_context
clCreateContext(
    const cl_context_properties * properties,
    cl_uint                       num_devices,
    const cl_device_id *          devices,
    void (CL_CALLBACK * pfn_notify)(const char *, const void *, size_t, void *),
    void *                        user_data,
    cl_int *                      errcode_ret
){
    LOG_API_CALL("properties = %p, num_devices = %u, devices = %p, pfn_notify = %p, user_data = %p, errcode_ret = %p", properties, num_devices, devices, pfn_notify, user_data, errcode_ret);

    if (num_devices > 1) {
        cvk_error("Only one device per context is supported.");
        return nullptr;
    }

    cl_context context = new cvk_context(devices[0], properties);

    if (errcode_ret != nullptr) {
        *errcode_ret = CL_SUCCESS;
    }

    return context;
}

cl_context
clCreateContextFromType(
    const cl_context_properties  *properties,
    cl_device_type                device_type,
    void  (CL_CALLBACK *pfn_notify) (const char *, const void  *, size_t, void  *),
    void                         *user_data,
    cl_int                       *errcode_ret
){
    LOG_API_CALL("properties = %p, device_type = %lu, pfn_notify = %p, "
                 "user_data = %p, errcode_ret = %p",
                 properties, device_type, pfn_notify, user_data, errcode_ret);

    cl_device_id device;

    cl_int err = clGetDeviceIDs(nullptr, device_type, 1, &device, nullptr);

    if (err == CL_SUCCESS) {
        return clCreateContext(properties, 1, &device, pfn_notify, user_data, errcode_ret);
    } else {
        *errcode_ret = err;
        return nullptr;
    }
}

cl_int
clRetainContext(
    cl_context context
){
    LOG_API_CALL("context = %p", context);

    if (!is_valid_context(context)) {
        return CL_INVALID_CONTEXT;
    }

    context->retain();

    return CL_SUCCESS;
}

cl_int
clReleaseContext(
    cl_context context
){
    LOG_API_CALL("context = %p", context);

    if (!is_valid_context(context)) {
        return CL_INVALID_CONTEXT;
    }

    context->release();

    return CL_SUCCESS;
}

cl_int
clGetContextInfo(
    cl_context       context,
    cl_context_info  param_name,
    size_t           param_value_size,
    void            *param_value,
    size_t          *param_value_size_ret
){
    LOG_API_CALL("context = %p, param_name = %u, size = %zu, value = %p, size_ret = %p",
                 context, param_name, param_value_size, param_value, param_value_size_ret);

    cl_int ret = CL_SUCCESS;
    size_t size_ret = 0;
    const void *copy_ptr = nullptr;
    cl_uint val_uint;
    cl_device_id val_device;

    switch (param_name) {
    case CL_CONTEXT_REFERENCE_COUNT:
        val_uint = context->refcount();
        copy_ptr = &val_uint;
        size_ret = sizeof(val_uint);
        break;
    case CL_CONTEXT_DEVICES:
        val_device = context->device();
        copy_ptr = &val_device;
        size_ret = sizeof(val_device);
        break;
    case CL_CONTEXT_NUM_DEVICES:
        val_uint = context->num_devices();
        copy_ptr = &val_uint;
        size_ret = sizeof(cl_uint);
        break;
    case CL_CONTEXT_PROPERTIES:
        if (context->properties().size() == 0) {
            size_ret = 0;
            copy_ptr = nullptr;
        } else {
            copy_ptr = context->properties().data();
            size_ret = context->properties().size() * sizeof(cl_context_properties);
        }
        break;
    default:
        ret = CL_INVALID_VALUE;
        break;
    }

    if ((param_value != nullptr) && (copy_ptr != nullptr)) {
        memcpy(param_value, copy_ptr, std::min(param_value_size, size_ret));
    }

    if (param_value_size_ret != nullptr) {
        *param_value_size_ret = size_ret;
    }

    return ret;
}

// Event APIs
cl_int
clWaitForEvents(
    cl_uint         num_events,
    const cl_event *event_list
){
    LOG_API_CALL("num_events = %u, event_list = %p", num_events, event_list);

    if ((num_events == 0) || (event_list == nullptr)) {
        return CL_INVALID_VALUE;
    }

    // TODO validate that all events belong to the same context
    for (cl_uint i = 0; i < num_events; i++) {
        if (!is_valid_event(event_list[i])) {
            return CL_INVALID_EVENT;
        }
    }

    return cvk_command_queue::wait_for_events(num_events, event_list);
}

cl_int clEnqueueWaitForEvents(
    cl_command_queue command_queue,
    cl_uint          num_events,
    const cl_event  *event_list
){
    LOG_API_CALL("command_queue = %p, num_events = %u, event_list = %p",
                 command_queue, num_events, event_list);

    return CL_INVALID_OPERATION; // TODO implement
}

cl_int
clReleaseEvent(
    cl_event event
){
    LOG_API_CALL("event = %p", event);

    if (!is_valid_event(event)) {
        return CL_INVALID_EVENT;
    }

    event->release();

    return CL_SUCCESS;
}

cl_int
clRetainEvent(
    cl_event event
){
    LOG_API_CALL("event = %p", event);

    if (!is_valid_event(event)) {
        return CL_INVALID_EVENT;
    }

    event->retain();

    return CL_SUCCESS;
}

cl_event clCreateUserEvent(
    cl_context context,
    cl_int    *errcode_ret
){
    LOG_API_CALL("context = %p, errcode_ret = %p", context, errcode_ret);

    if (!is_valid_context(context)) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_CONTEXT;
        }
    }

    auto event = new cvk_event(context, CL_SUBMITTED, CL_COMMAND_USER, nullptr);

    if (errcode_ret != nullptr) {
        *errcode_ret = CL_SUCCESS;
    }

    return event;
}

cl_int clSetUserEventStatus(
    cl_event event,
    cl_int execution_status
){
    LOG_API_CALL("event = %p, execution_status = %d", event, execution_status);

    if (!is_valid_event(event)) {
        return CL_INVALID_EVENT;
    }

    event->set_status(execution_status);

    return CL_SUCCESS;
}

cl_int clSetEventCallback(
    cl_event event,
    cl_int  command_exec_callback_type ,
    void (CL_CALLBACK  *pfn_event_notify) (cl_event event, cl_int event_command_exec_status, void *user_data),
    void *user_data
){
    LOG_API_CALL("event = %p, callback_type = %d, pfn_event_notify = %p, user_data = %p", event, command_exec_callback_type, pfn_event_notify, user_data);

    if (!is_valid_event(event)) {
        return CL_INVALID_EVENT;
    }

    if (pfn_event_notify == nullptr) {
        return CL_INVALID_VALUE;
    }

    event->register_callback(command_exec_callback_type, pfn_event_notify, user_data);

    return CL_SUCCESS;
}

static bool event_wait_list_is_valid(cl_uint num_events_in_wait_list,
                                     const cl_event *event_wait_list) {

    if (((num_events_in_wait_list > 0) && (event_wait_list == nullptr)) ||
        ((num_events_in_wait_list == 0) && (event_wait_list != nullptr))) {
        return false;
    }

    for (cl_uint i = 0; i < num_events_in_wait_list; i++) {
        if (!is_valid_event(event_wait_list[i])) {
            return false;
        }
    }

    return true;
}

cl_int cvk_enqueue_marker_with_wait_list(
    cl_command_queue command_queue,
    cl_uint          num_events_in_wait_list,
    const cl_event  *event_wait_list,
    cl_event        *event
){
    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    if (!event_wait_list_is_valid(num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    auto cmd = new cvk_command_dep(command_queue, CL_COMMAND_MARKER);

    command_queue->enqueue_command_with_deps(cmd, num_events_in_wait_list, event_wait_list, event);

    return CL_SUCCESS;
}

cl_int clEnqueueMarkerWithWaitList(
    cl_command_queue command_queue,
    cl_uint          num_events_in_wait_list,
    const cl_event  *event_wait_list,
    cl_event        *event
){
    LOG_API_CALL("command_queue = %p, num_events_in_wait_list = %u, "
                 "event_wait_list = %p, event = %p",
                 command_queue, num_events_in_wait_list, event_wait_list, event);

    return cvk_enqueue_marker_with_wait_list(command_queue,
                                             num_events_in_wait_list,
                                             event_wait_list, event);
}

cl_int clEnqueueMarker(cl_command_queue command_queue, cl_event *event)
{
    LOG_API_CALL("command_queue = %p, event = %p", command_queue, event);

    return cvk_enqueue_marker_with_wait_list(command_queue, 0, nullptr, event);
}

cl_int cvk_enqueue_barrier_with_wait_list(
    cl_command_queue command_queue,
    cl_uint          num_events_in_wait_list,
    const cl_event  *event_wait_list,
    cl_event        *event
){
    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    if (!event_wait_list_is_valid(num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    auto cmd = new cvk_command_dep(command_queue, CL_COMMAND_BARRIER);

    command_queue->enqueue_command_with_deps(cmd, num_events_in_wait_list, event_wait_list, event);

    return CL_SUCCESS;
}

cl_int clEnqueueBarrierWithWaitList(
    cl_command_queue command_queue,
    cl_uint          num_events_in_wait_list,
    const cl_event  *event_wait_list,
    cl_event        *event
){
    LOG_API_CALL("command_queue = %p, num_events_in_wait_list = %u, "
                 "event_wait_list = %p, event = %p",
                 command_queue, num_events_in_wait_list, event_wait_list, event);

    return cvk_enqueue_barrier_with_wait_list(command_queue,
                                              num_events_in_wait_list,
                                              event_wait_list, event);
}

cl_int clEnqueueBarrier(cl_command_queue command_queue)
{
    LOG_API_CALL("command_queue = %p", command_queue);

    return cvk_enqueue_barrier_with_wait_list(command_queue, 0, nullptr, nullptr);
}

cl_int clGetEventInfo(
    cl_event      event,
    cl_event_info param_name,
    size_t        param_value_size,
    void         *param_value,
    size_t       *param_value_size_ret
){
    LOG_API_CALL("event = %p, param_name = %x, param_value_size = %zu, param_value = %p, param_value_size_ret = %p", event, param_name, param_value_size, param_value, param_value_size_ret);

    cl_int ret = CL_SUCCESS;
    size_t ret_size = 0;
    const void *copy_ptr = nullptr;
    cl_uint val_uint;
    cl_int val_int;
    cl_context val_context;
    cl_command_type val_command_type;
    cl_command_queue val_command_queue;

    if (!is_valid_event(event)) {
        return CL_INVALID_VALUE;
    }

    switch (param_name) {
    case CL_EVENT_REFERENCE_COUNT:
        val_uint = event->refcount();
        copy_ptr = &val_uint;
        ret_size = sizeof(val_uint);
        break;
    case CL_EVENT_COMMAND_EXECUTION_STATUS:
        val_int = event->get_status();
        copy_ptr = &val_int;
        ret_size = sizeof(val_int);
        break;
    case CL_EVENT_COMMAND_QUEUE:
        if (event->is_user_event()) {
            val_command_queue = nullptr;
        } else {
            val_command_queue = event->queue();
        }
        copy_ptr = &val_command_queue;
        ret_size = sizeof(val_command_queue);
        break;
    case CL_EVENT_COMMAND_TYPE:
        if (event->is_user_event()) {
            val_command_type = CL_COMMAND_USER;
        } else {
            val_command_type = event->command_type();
        }
        copy_ptr = &val_command_type;
        ret_size = sizeof(val_command_type);
        break;
    case CL_EVENT_CONTEXT:
        val_context = event->context();
        copy_ptr = &val_context;
        ret_size = sizeof(val_context);
        break;
    default:
        ret = CL_INVALID_VALUE;
    }

    if ((param_value != nullptr) && (copy_ptr != nullptr)) {
        if (param_value_size < ret_size) {
            ret = CL_INVALID_VALUE;
        }
        memcpy(param_value, copy_ptr, std::min(param_value_size, ret_size));
    }

    if (param_value_size_ret != nullptr) {
        *param_value_size_ret = ret_size;
    }

    return ret;
}

// Command Queue APIs
cl_command_queue
clCreateCommandQueue(
    cl_context                     context,
    cl_device_id                   device,
    cl_command_queue_properties    properties,
    cl_int *                       errcode_ret
){
    LOG_API_CALL("context = %p, device = %p, properties = %lu, errcode_ret = %p",
                 context, device, properties, errcode_ret);

    if (!is_valid_context(context)) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_CONTEXT;
        }
        return nullptr;
    }

    auto queue = std::make_unique<cvk_command_queue>(context, device, properties);

    cl_int err = queue->init();

    if (errcode_ret != nullptr) {
        *errcode_ret = err;
    }

    if (err != CL_SUCCESS) {
        return nullptr;
    } else {
        return queue.release();
    }
}

cl_int
clReleaseCommandQueue(
    cl_command_queue command_queue
){
    LOG_API_CALL("command_queue = %p", command_queue);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    command_queue->release();
    return CL_SUCCESS;
}


cl_int
clRetainCommandQueue(
    cl_command_queue command_queue
){
    LOG_API_CALL("command_queue = %p", command_queue);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    command_queue->retain();
    return CL_SUCCESS;
}

cl_int clGetCommandQueueInfo(
    cl_command_queue      command_queue,
    cl_command_queue_info param_name,
    size_t                param_value_size,
    void                 *param_value,
    size_t               *param_value_size_ret
){
    LOG_API_CALL("command_queue = %p, param_name = %x, param_value_size = %zu, param_value = %p, param_value_size_ret = %p", command_queue, param_name, param_value_size, param_value, param_value_size_ret);

    cl_int ret = CL_SUCCESS;
    size_t ret_size = 0;
    const void *copy_ptr = nullptr;
    cl_uint val_uint;
    cl_context val_context;
    cl_device_id val_device;
    cl_command_queue_properties val_properties;

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    switch (param_name) {
    case CL_QUEUE_REFERENCE_COUNT:
        val_uint = command_queue->refcount();
        copy_ptr = &val_uint;
        ret_size = sizeof(val_uint);
        break;
    case CL_QUEUE_CONTEXT:
        val_context = command_queue->context();
        copy_ptr = &val_context;
        ret_size = sizeof(val_context);
        break;
    case CL_QUEUE_DEVICE:
        val_device = command_queue->device();
        copy_ptr = &val_device;
        ret_size = sizeof(val_device);
        break;
    case CL_QUEUE_PROPERTIES:
        val_properties = command_queue->properties();
        copy_ptr = &val_properties;
        ret_size = sizeof(val_properties);
        break;
    default:
        ret = CL_INVALID_VALUE;
    }

    if ((param_value != nullptr) && (copy_ptr != nullptr)) {
        if (param_value_size < ret_size) {
            ret = CL_INVALID_VALUE;
        }
        memcpy(param_value, copy_ptr, std::min(param_value_size, ret_size));
    }

    if (param_value_size_ret != nullptr) {
        *param_value_size_ret = ret_size;
    }

    return ret;
}

// Memory Object APIs
cl_mem
clCreateBuffer(
    cl_context   context,
    cl_mem_flags flags,
    size_t       size,
    void *       host_ptr,
    cl_int *     errcode_ret
){
    LOG_API_CALL("context = %p, flags = %lu, size = %zu, host_ptr = %p, errcode_ret = %p",
                 context, flags, size, host_ptr, errcode_ret);

    cl_int err;
    auto buffer = cvk_buffer::create(context, flags, size, host_ptr, &err);

    if (errcode_ret != nullptr) {
        *errcode_ret = err;
    }

    if (err != CL_SUCCESS) {
        return nullptr;
    } else {
        return buffer.release();
    }
}

cl_mem clCreateSubBuffer(
    cl_mem                buffer,
    cl_mem_flags          flags,
    cl_buffer_create_type buffer_create_type,
    const void           *buffer_create_info,
    cl_int               *errcode_ret
){
    LOG_API_CALL("buffer = %p, flags = %lu, buffer_create_type = %u, "
                 "buffer_create_info = %p, errcode_ret = %p",
                 buffer, flags, buffer_create_type, buffer_create_info, errcode_ret);

    if (!is_valid_buffer(buffer) || buffer->is_sub_buffer()) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_MEM_OBJECT;
        }
        return nullptr;
    }

    // TODO CL_INVALID_VALUE if buffer was created with CL_MEM_WRITE_ONLY and flags specifies CL_MEM_READ_WRITE or CL_MEM_READ_ONLY, or if buffer was created with CL_MEM_READ_ONLY and flags specifies CL_MEM_READ_WRITE or CL_MEM_WRITE_ONLY, or if flags specifies CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR or CL_MEM_COPY_HOST_PTR.
    // TODO CL_INVALID_VALUE if buffer was created with CL_MEM_HOST_WRITE_ONLY and flags specifies CL_MEM_HOST_READ_ONLY or if buffer was created with CL_MEM_HOST_READ_ONLY and flags specifies CL_MEM_HOST_WRITE_ONLY, or if buffer was created with CL_MEM_HOST_NO_ACCESS and flags specifies CL_MEM_HOST_READ_ONLY or CL_MEM_HOST_WRITE_ONLY.
    // TODO CL_INVALID_VALUE if value specified in buffer_create_type is not valid.
    // TODO CL_INVALID_VALUE if value(s) specified in buffer_create_info (for a given buffer_create_type) is not valid or if buffer_create_info is NULL.
    // TODO CL_INVALID_BUFFER_SIZE if size is 0.
    // TODO CL_MEM_OBJECT_ALLOCATION_FAILURE if there is a failure to allocate memory for sub-buffer object.
    // TODO CL_OUT_OF_RESOURCES if there is a failure to allocate resources required by the OpenCL implementation on the device.
    // TODO CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required by the OpenCL implementation on the host.

    if (buffer_create_type != CL_BUFFER_CREATE_TYPE_REGION) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_VALUE;
        }
        return nullptr;
    }

    auto region = static_cast<const cl_buffer_region*>(buffer_create_info);
    LOG_API_CALL("CL_BUFFER_CREATE_TYPE_REGION, origin = %zu, size = %zu", region->origin, region->size);

    cl_int err = CL_SUCCESS;
    auto buf = static_cast<cvk_buffer*>(buffer);
    auto sub = buf->create_subbuffer(flags, region->origin, region->size);

    if (sub == nullptr) {
        err = CL_OUT_OF_RESOURCES;
    }

    if (errcode_ret != nullptr) {
        *errcode_ret = err;
    }

    return sub;
}

cl_int
clRetainMemObject(
    cl_mem memobj
){
    LOG_API_CALL("memobj = %p", memobj);

    if (!is_valid_mem_object(memobj)) {
        return CL_INVALID_MEM_OBJECT;
    }

    memobj->retain();

    return CL_SUCCESS;
}

cl_int
clReleaseMemObject(
    cl_mem memobj
){
    LOG_API_CALL("memobj = %p", memobj);

    if (!is_valid_mem_object(memobj)) {
        return CL_INVALID_MEM_OBJECT;
    }

    memobj->release();

    return CL_SUCCESS;
}

cl_int
clSetMemObjectDestructorCallback(
    cl_mem memobj,
    void (CL_CALLBACK  *pfn_notify) (cl_mem memobj,	void *user_data),
    void *user_data
){
    LOG_API_CALL("memobj = %p, pfn_notify = %p, user_data = %p", memobj, pfn_notify, user_data);

    if (!is_valid_mem_object(memobj)) {
        return CL_INVALID_MEM_OBJECT;
    }

    if (pfn_notify == nullptr) {
        return CL_INVALID_VALUE;
    }

    memobj->add_destructor_callback(pfn_notify, user_data);

    return CL_SUCCESS;
}

cl_int clEnqueueMigrateMemObjects(
    cl_command_queue       command_queue ,
    cl_uint                num_mem_objects ,
    const cl_mem          *mem_objects ,
    cl_mem_migration_flags flags ,
    cl_uint                num_events_in_wait_list ,
    const cl_event        *event_wait_list ,
    cl_event              *event
){
    LOG_API_CALL("command_queue = %p, num_mem_objects = %u, mem_objects = %p, "
                 "flags = %lx, num_events_in_wait_list = %u, "
                 "event_wait_list = %p, event = %p",
                 command_queue, num_mem_objects, mem_objects, flags,
                 num_events_in_wait_list, event_wait_list, event);

    if ((num_mem_objects == 0) || (mem_objects == nullptr)) {
        return CL_INVALID_VALUE;
    }

    for (cl_uint i = 0; i < num_mem_objects; i++) {
        if (!is_valid_mem_object(mem_objects[i])) {
            return CL_INVALID_MEM_OBJECT;
        }
    }

    // TODO CL_INVALID_VALUE if flags is not 0 or is not any of the values
    // described in the table above.
    // TODO CL_MEM_OBJECT_ALLOCATION_FAILURE if there is a failure to allocate
    // memory for the specified set of memory objects in mem_objects.

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    if (!event_wait_list_is_valid(num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    if (!is_same_context(command_queue, num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_CONTEXT;
    }

    for (cl_uint i = 0; i < num_mem_objects; i++) {
        if (!is_same_context(command_queue, mem_objects[i])) {
            return CL_INVALID_CONTEXT;
        }
    }

    auto cmd = new cvk_command_dep(command_queue,
                                   CL_COMMAND_MIGRATE_MEM_OBJECTS);

    command_queue->enqueue_command_with_deps(cmd, num_events_in_wait_list,
                                             event_wait_list, event);

    return CL_SUCCESS;
}

cl_int clGetMemObjectInfo(
    cl_mem      memobj,
    cl_mem_info param_name,
    size_t      param_value_size,
    void       *param_value,
    size_t     *param_value_size_ret
){
    LOG_API_CALL("memobj = %p, param_name = %x, param_value_size = %zu, param_value = %p, param_value_size_ret = %p", memobj, param_name, param_value_size, param_value, param_value_size_ret);

    cl_int ret = CL_SUCCESS;
    size_t ret_size = 0;
    const void *copy_ptr = nullptr;
    cl_uint val_uint;
    cl_context val_context;
    cl_mem_object_type val_object_type;
    cl_mem_flags val_flags;
    size_t val_sizet;
    cl_mem val_memobj;
    void *val_ptr;

    if (!is_valid_mem_object(memobj)) {
        return CL_INVALID_MEM_OBJECT;
    }

    switch (param_name) {
    case CL_MEM_REFERENCE_COUNT:
        val_uint = memobj->refcount();
        copy_ptr = &val_uint;
        ret_size = sizeof(val_uint);
        break;
    case CL_MEM_CONTEXT:
        val_context = memobj->context();
        copy_ptr = &val_context;
        ret_size = sizeof(val_context);
        break;
    case CL_MEM_TYPE:
        val_object_type = memobj->type();
        copy_ptr = &val_object_type;
        ret_size = sizeof(val_object_type);
        break;
    case CL_MEM_FLAGS:
        val_flags = memobj->flags();
        copy_ptr = &val_flags;
        ret_size = sizeof(val_flags);
        break;
    case CL_MEM_SIZE:
        val_sizet = memobj->size();
        copy_ptr = &val_sizet;
        ret_size = sizeof(val_sizet);
        break;
    case CL_MEM_MAP_COUNT:
        val_uint = memobj->map_count();
        copy_ptr = &val_uint;
        ret_size = sizeof(val_uint);
        break;
    case CL_MEM_ASSOCIATED_MEMOBJECT:
        val_memobj = memobj->parent();
        copy_ptr = &val_memobj;
        ret_size = sizeof(val_memobj);
        break;
    case CL_MEM_OFFSET:
        val_sizet = memobj->parent_offset();
        copy_ptr = &val_sizet;
        ret_size = sizeof(val_sizet);
        break;
    case CL_MEM_HOST_PTR:
        val_ptr = memobj->host_ptr();
        copy_ptr = &val_ptr;
        ret_size = sizeof(val_ptr);
        break;
    default:
        ret = CL_INVALID_VALUE;
    }

    if ((param_value != nullptr) && (copy_ptr != nullptr)) {
        if (param_value_size < ret_size) {
            ret = CL_INVALID_VALUE;
        }
        memcpy(param_value, copy_ptr, std::min(param_value_size, ret_size));
    }

    if (param_value_size_ret != nullptr) {
        *param_value_size_ret = ret_size;
    }

    return ret;
}

// Program Object APIs
cl_program
clCreateProgramWithSource(
    cl_context        context,
    cl_uint           count,
    const char **     strings,
    const size_t *    lengths,
    cl_int *          errcode_ret
){
    LOG_API_CALL("context = %p, count = %u, lengths = %p", context, count, lengths);

    if (!is_valid_context(context)) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_CONTEXT;
        }
        return nullptr;
    }

    cl_program prog = new cvk_program(context);

    for (cl_uint i = 0; i < count; i++) {
        size_t len = (lengths != nullptr) ? lengths[i] : 0;
        prog->append_source(strings[i], len);
    }

    if (errcode_ret != nullptr) {
        *errcode_ret = CL_SUCCESS;
    }

    cvk_debug_fn("\n%s", prog->source().c_str());

    return prog;
}

cl_program
clCreateProgramWithBinary(
    cl_context            context,
    cl_uint               num_devices,
    const cl_device_id   *device_list,
    const size_t         *lengths,
    const unsigned char **binaries,
    cl_int               *binary_status,
    cl_int               *errcode_ret
){
    LOG_API_CALL("context = %p, num_devices = %u, device_list = %p, lengths = %p, binaries = %p, binary_status = %p, errcode_ret = %p", context, num_devices, device_list, lengths, binaries, binary_status, errcode_ret);

    if (!is_valid_context(context)) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_CONTEXT;
        }
        return nullptr;
    }

    if ((num_devices != 1) || (device_list == nullptr)) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_VALUE;
        }
        return nullptr;
    }

    if (device_list[0] != context->device()) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_DEVICE;
        }
        return nullptr;
    }

    if ((lengths == nullptr) || (binaries == nullptr)) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_VALUE;
        }
        return nullptr;
    }

    for (cl_uint i = 0; i < num_devices; i++) {
        if ((lengths[i] == 0) || (binaries[i] == nullptr)) {
            if (errcode_ret != nullptr) {
                *errcode_ret = CL_INVALID_VALUE;
            }
            return nullptr;
        }
    }

    cl_program prog = new cvk_program(context);

    cl_int load_status = CL_SUCCESS;
    if (!prog->read(binaries[0], lengths[0])) {
        load_status = CL_INVALID_BINARY;
    }

    if (binary_status != nullptr) {
        binary_status[0] = load_status;
    }

    if (errcode_ret != nullptr) {
        *errcode_ret = load_status;
    }

    return prog;
}

cl_int
clBuildProgram(
    cl_program           program,
    cl_uint              num_devices,
    const cl_device_id  *device_list,
    const char          *options,
    void (CL_CALLBACK *pfn_notify)(cl_program /* program */, void * /* user_data */),
    void                *user_data
){
    LOG_API_CALL("program = %p, num_device = %d, device_list = %p, options = %s, pfn_notify = %p, user_data = %p", program, num_devices, device_list, options, pfn_notify, user_data);

    if (!is_valid_program(program)) {
        return CL_INVALID_PROGRAM;
    }

    if (((num_devices > 0) && (device_list == nullptr)) || ((num_devices == 0) && (device_list != nullptr))) {
        return CL_INVALID_VALUE;
    }

    if ((pfn_notify == nullptr) && (user_data != nullptr)) {
        return CL_INVALID_VALUE;
    }

    // TODO CL_INVALID_DEVICE if OpenCL devices listed in device_list are not in the list of devices associated with program.
    // TODO CL_INVALID_BINARY if program is created with clCreateProgramWithBinary and devices listed in device_list do not have a valid program binary loaded.
    // TODO CL_INVALID_BUILD_OPTIONS if the build options specified by options are invalid.
    // TODO CL_COMPILER_NOT_AVAILABLE if program is created with clCreateProgramWithSource and a compiler is not available i.e. CL_DEVICE_COMPILER_AVAILABLE specified in the table of OpenCL Device Queries for clGetDeviceInfo is set to CL_FALSE.
    // TODO CL_BUILD_PROGRAM_FAILURE if there is a failure to build the program executable. This error will be returned if clBuildProgram does not return until the build has completed.
    // TODO CL_INVALID_OPERATION if there are kernel objects attached to program.
    // TODO CL_INVALID_OPERATION if program was not created with clCreateProgramWithSource or clCreateProgramWithBinary or clCreateProgramWithILKHR.

    if (!program->build(build_operation::build, num_devices, device_list, options, 0, nullptr, nullptr, pfn_notify, user_data)) {
        return CL_INVALID_OPERATION;
    }

    if (pfn_notify == nullptr) {

        program->wait_for_operation();

        if (program->build_status() != CL_BUILD_SUCCESS) {
            return CL_BUILD_PROGRAM_FAILURE;
        }
    }

    return CL_SUCCESS;
}

cl_int clCompileProgram(
    cl_program          program,
    cl_uint             num_devices,
    const cl_device_id *device_list,
    const char         *options,
    cl_uint             num_input_headers,
    const cl_program   *input_headers,
    const char        **header_include_names,
    void (CL_CALLBACK *pfn_notify)( cl_program program, void *user_data),
    void               *user_data
){
    LOG_API_CALL("program = %p, num_devices = %u, device_list = %p, options = %p, num_input_headers = %u, input_headers = %p, header_include_names = %p, pfn_notify = %p, user_data = %p", program, num_devices, device_list, options, num_input_headers, input_headers, header_include_names, pfn_notify, user_data);

    if (!is_valid_program(program)) {
        return CL_INVALID_PROGRAM;
    }

    if (((device_list == nullptr) && (num_devices > 0)) || ((device_list != nullptr) && (num_devices == 0))) {
        return CL_INVALID_VALUE;
    }

    if (((num_input_headers == 0) && ((header_include_names != nullptr) || (input_headers != nullptr))) ||
        ((num_input_headers != 0) && ((header_include_names == nullptr) || (input_headers == nullptr)))) {
        return CL_INVALID_VALUE;
    }

    if ((pfn_notify == nullptr) && (user_data != nullptr)) {
        return CL_INVALID_VALUE;
    }

    // TODO CL_INVALID_DEVICE if OpenCL devices listed in device_list are not in the list of devices associated with program. 
    // TODO CL_INVALID_COMPILER_OPTIONS if the compiler options specified by options are invalid.

    // TODO CL_COMPILER_NOT_AVAILABLE if a compiler is not available i.e. CL_DEVICE_COMPILER_AVAILABLE specified in in the table of allowed values for param_name for clGetDeviceInfo is set to CL_FALSE.
    // TODO CL_COMPILE_PROGRAM_FAILURE if there is a failure to compile the program source. This error will be returned if clCompileProgram does not return until the compile has completed.
    // TODO CL_INVALID_OPERATION if there are kernel objects attached to program.
    if (program->loaded_from_binary()) {
        return CL_INVALID_OPERATION;
    }

    // TODO Validate program
    if (!program->build(build_operation::compile, num_devices, device_list, options, num_input_headers, input_headers, header_include_names, pfn_notify, user_data)) {
        return CL_INVALID_OPERATION;
    }

    if (pfn_notify == nullptr) {

        program->wait_for_operation();

        if (program->build_status() != CL_BUILD_SUCCESS) {
            return CL_BUILD_PROGRAM_FAILURE;
        }
    }

    return CL_SUCCESS;
}

cl_program
clLinkProgram(
    cl_context          context,
    cl_uint             num_devices,
    const cl_device_id *device_list,
    const char         *options,
    cl_uint             num_input_programs,
    const cl_program   *input_programs,
    void (CL_CALLBACK *pfn_notify) (cl_program program, void *user_data),
    void               *user_data,
    cl_int             *errcode_ret
){
    LOG_API_CALL("context = %p, num_devices = %d, device_list = %p, options = %p, num_input_programs = %d, input_programs = %p, pfn_notify = %p, user_data = %p, errcode_ret = %p", context, num_devices, device_list, options, num_input_programs, input_programs, pfn_notify, user_data, errcode_ret);

    if (!is_valid_context(context)) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_CONTEXT;
        }
        return nullptr;
    }

    if (((device_list == nullptr) && (num_devices > 0)) || ((device_list != nullptr) && (num_devices == 0))) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_VALUE;
        }
        return nullptr;
    }

    if (((input_programs == nullptr) && (num_input_programs == 0)) || ((num_input_programs == 0) && (input_programs != nullptr)) || ((num_input_programs != 0) && (input_programs == nullptr))) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_VALUE;
        }
        return nullptr;
    }

    for (cl_uint i = 0; i < num_input_programs; i++) {
        if (!is_valid_program(input_programs[i])) {
            if (errcode_ret != nullptr) {
                *errcode_ret = CL_INVALID_PROGRAM;
            }
            return nullptr;
        }
    }

    if ((pfn_notify == nullptr) && (user_data != nullptr)) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_VALUE;
        }
        return nullptr;
    }

    // TODO CL_INVALID_DEVICE if OpenCL devices listed in device_list are not in the list of devices associated with context.
    // TODO CL_INVALID_LINKER_OPTIONS if the linker options specified by options are invalid
    // TODO CL_INVALID_OPERATION if the rules for devices containing compiled binaries or libraries as described in input_programs argument above are not followed.
    for (cl_uint i = 0; i < num_input_programs; i++) {
        if (!input_programs[i]->can_be_linked()) {
            if (errcode_ret != nullptr) {
                *errcode_ret = CL_INVALID_OPERATION;
            }
            return nullptr;
        }
    }

    cvk_program *prog_ret = new cvk_program(context);

    if (!prog_ret->build(build_operation::link, num_devices, device_list, options, num_input_programs, input_programs, nullptr, pfn_notify, user_data)) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_OPERATION;
        }
        return nullptr;
    }

    if (pfn_notify == nullptr) {

        prog_ret->wait_for_operation();

        if (prog_ret->build_status() != CL_BUILD_SUCCESS) {
            if (errcode_ret != nullptr) {
                *errcode_ret = CL_LINK_PROGRAM_FAILURE;
            }
            return nullptr;
        }
    }

    if (errcode_ret != nullptr) {
        *errcode_ret = CL_SUCCESS;
    }

    return prog_ret;
}

cl_int clUnloadPlatformCompiler(cl_platform_id platform)
{
    LOG_API_CALL("platform = %p", platform);

    if (!is_valid_platform(platform)) {
        return CL_INVALID_PLATFORM;
    }

    return CL_SUCCESS;
}

cl_int clUnloadCompiler()
{
    LOG_API_CALL("%s", "");

    return CL_SUCCESS;
}

cl_int
clGetProgramInfo(
    cl_program      program,
    cl_program_info param_name,
    size_t          param_value_size,
    void           *param_value,
    size_t         *param_value_size_ret
){
    LOG_API_CALL("program = %p, param_name = %x, param_value_size = %zu, param_value = %p, param_value_size_ret = %p", program, param_name, param_value_size, param_value, param_value_size_ret);

    cl_int ret = CL_SUCCESS;
    size_t ret_size = 0;
    const void *copy_ptr = nullptr;
    cl_uint val_uint;
    cl_context val_context;
    size_t val_sizet;
    string val_string;
    std::vector<size_t> val_sizet_vec;
    std::vector<const cvk_device*> val_devices;

    if (!is_valid_program(program)) {
        return CL_INVALID_PROGRAM;
    }

    // TODO CL_INVALID_PROGRAM_EXECUTABLE if param_name is CL_PROGRAM_NUM_KERNELS or CL_PROGRAM_KERNEL_NAMES and a successful program executable has not been built for at least one device in the list of devices associated with program. 

    switch (param_name) {
    case CL_PROGRAM_NUM_DEVICES:
        val_uint = program->num_devices();
        copy_ptr = &val_uint;
        ret_size = sizeof(val_uint);
        break;
    case CL_PROGRAM_REFERENCE_COUNT:
        val_uint = program->refcount();
        copy_ptr = &val_uint;
        ret_size = sizeof(val_uint);
        break;
    case CL_PROGRAM_CONTEXT:
        val_context = program->context();
        copy_ptr = &val_context;
        ret_size = sizeof(val_context);
        break;
    case CL_PROGRAM_DEVICES:
        val_devices = program->devices();
        copy_ptr = val_devices.data();
        ret_size = sizeof(cl_device_id) * val_devices.size();
        break;
    case CL_PROGRAM_NUM_KERNELS:
        val_sizet = program->num_kernels();
        copy_ptr = &val_sizet;
        ret_size = sizeof(val_sizet);
        break;
    case CL_PROGRAM_SOURCE:
        copy_ptr = program->source().c_str();
        ret_size = program->source().size() + 1;
        break;
    case CL_PROGRAM_KERNEL_NAMES: {
        val_string = "";
        std::string sep = "";
        for (auto kname : program->kernel_names()) {
            val_string += sep + kname;
            sep = ";";
        }
        copy_ptr = val_string.c_str();
        ret_size = val_string.size_with_null();
        break;
    }
    case CL_PROGRAM_BINARY_SIZES:
        for (uint32_t i = 0; i < program->num_devices(); i++) {
            val_sizet_vec.push_back(program->binary_size());
        }
        copy_ptr = val_sizet_vec.data();
        ret_size = val_sizet_vec.size() * sizeof(size_t);
        break;
    case CL_PROGRAM_BINARIES:
        ret_size = program->num_devices() * sizeof(unsigned char*);
        if (param_value != nullptr) {
            for (uint32_t i = 0; i < program->num_devices(); i++) {
                auto dst = static_cast<unsigned char**>(param_value)[i];
                if (dst != nullptr) {
                    auto success = program->write(dst);
                    if (!success) {
                        ret = CL_OUT_OF_RESOURCES;
                        break;
                    }
                }
            }
        }
        break;
    case CL_PROGRAM_IL_KHR:
        copy_ptr = program->il().data();
        ret_size = program->il().size();
        break;
    default:
        ret = CL_INVALID_VALUE;
    }

    if ((param_value != nullptr) && (copy_ptr != nullptr) && (param_name != CL_PROGRAM_BINARIES)) {
        if (param_value_size < ret_size) {
            ret = CL_INVALID_VALUE;
        }
        memcpy(param_value, copy_ptr, std::min(param_value_size, ret_size));
    }

    if (param_value_size_ret != nullptr) {
        *param_value_size_ret = ret_size;
    }

    return ret;
}

cl_int
clGetProgramBuildInfo(
    cl_program            program,
    cl_device_id          device,
    cl_program_build_info param_name,
    size_t                param_value_size,
    void *                param_value,
    size_t *              param_value_size_ret
){
    LOG_API_CALL("program = %p, device = %p, param_name = %x, param_value_size = %zu, param_value = %p, param_value_size_ret = %p", program, device, param_name, param_value_size, param_value, param_value_size_ret);
    cl_int ret = CL_SUCCESS;
    size_t ret_size = 0;
    const void *copy_ptr = nullptr;
    cl_build_status val_status;
    string val_string;
    cl_program_binary_type val_binarytype;

    if (!is_valid_program(program)) {
        return CL_INVALID_PROGRAM;
    }

    if (!is_valid_device(device)) { // TODO check the program knows the device
        return CL_INVALID_DEVICE;
    }

    switch (param_name) {
    case CL_PROGRAM_BUILD_STATUS:
        val_status = program->build_status(device);
        copy_ptr = &val_status;
        ret_size = sizeof(val_status);
        break;
    case CL_PROGRAM_BUILD_LOG: // TODO
        val_string = "BUILD LOG UNSUPPORTED";
        copy_ptr = val_string.c_str();
        ret_size = val_string.size_with_null();
        break;
    case CL_PROGRAM_BUILD_OPTIONS:
        copy_ptr = program->build_options().c_str();
        ret_size = program->build_options().size() + 1;
        break;
    case CL_PROGRAM_BINARY_TYPE:
        val_binarytype = program->binary_type(device);
        copy_ptr = &val_binarytype;
        ret_size = sizeof(val_binarytype);
        break;
    default:
        ret = CL_INVALID_VALUE;
    }

    if ((param_value != nullptr) && (copy_ptr != nullptr)) {
        memcpy(param_value, copy_ptr, std::min(param_value_size, ret_size));
    }

    if (param_value_size_ret != nullptr) {
        *param_value_size_ret = ret_size;
    }

    return ret;
}

cl_int
clRetainProgram(
    cl_program program
){
    LOG_API_CALL("program = %p", program);

    if (!is_valid_program(program)) {
        return CL_INVALID_PROGRAM;
    }

    program->retain();
    return CL_SUCCESS;
}

cl_int
clReleaseProgram(
    cl_program program
){
    LOG_API_CALL("program = %p", program);

    if (!is_valid_program(program)) {
        return CL_INVALID_PROGRAM;
    }

    program->release();
    return CL_SUCCESS;
}

// Kernel Object APIs
cl_kernel
cvk_create_kernel(
    cl_program      program,
    const char *    kernel_name,
    cl_int *        errcode_ret
){
    auto kernel = std::make_unique<cvk_kernel>(program, kernel_name);

    *errcode_ret = kernel->init();

    if (*errcode_ret != CL_SUCCESS) {
        return nullptr;
    } else {
        return kernel.release();
    }
}

cl_kernel
clCreateKernel(
    cl_program      program,
    const char *    kernel_name,
    cl_int *        errcode_ret
){
    LOG_API_CALL("program = %p, kernel_name = %s", program, kernel_name);

    if (!is_valid_program(program)) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_PROGRAM;
        }
        return nullptr;
    }

    if (program->build_status() != CL_BUILD_SUCCESS) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_PROGRAM_EXECUTABLE;
        }
        return nullptr;
    }

    cl_int err;
    cl_kernel ret = cvk_create_kernel(program, kernel_name, &err);

    if (errcode_ret != nullptr) {
        *errcode_ret = err;
    }

    return ret;
}

cl_int
clCreateKernelsInProgram(
    cl_program program,
    cl_uint    num_kernels,
    cl_kernel *kernels,
    cl_uint   *num_kernels_ret
){
    LOG_API_CALL("program = %p, num_kernels = %u, kernels = %p, num_kernels_ret = %p", program, num_kernels, kernels, num_kernels_ret);

    if (!is_valid_program(program)) {
        return CL_INVALID_PROGRAM;
    }

    if (program->build_status() != CL_BUILD_SUCCESS) {
        return CL_INVALID_PROGRAM_EXECUTABLE;
    }

    cl_uint num_kernels_in_program = program->num_kernels();

    if ((kernels != nullptr) && (num_kernels < num_kernels_in_program)) {
        return CL_INVALID_VALUE;
    }

    if (kernels != nullptr) {
        cl_uint i = 0;
        cl_int err;
        for (auto &kname : program->kernel_names()) {
            kernels[i] = cvk_create_kernel(program, kname, &err);
            if (err != CL_SUCCESS) {
                return err;
            }
            ++i;
        }
    }

    if (num_kernels_ret != nullptr) {
        *num_kernels_ret = num_kernels_in_program;
    }

    return CL_SUCCESS;
}

cl_int
clSetKernelArg(
    cl_kernel    kernel,
    cl_uint      arg_index,
    size_t       arg_size,
    const void * arg_value
){

    LOG_API_CALL("kernel = %p, arg_index = %u, arg_size = %zu, arg_value = %p", kernel, arg_index, arg_size, arg_value);

    if (!is_valid_kernel(kernel)) {
        return CL_INVALID_KERNEL;
    }

// TODO CL_INVALID_ARG_VALUE if arg_value specified is not a valid value.
// TODO CL_INVALID_MEM_OBJECT for an argument declared to be a memory object when the specified arg_value is not a valid memory object.
// TODO CL_INVALID_SAMPLER for an argument declared to be of type sampler_t when the specified arg_value is not a valid sampler object.
// TODO CL_INVALID_ARG_SIZE if arg_size does not match the size of the data type for an argument that is not a memory object or if the argument is a memory object and arg_size != sizeof(cl_mem) or if arg_size is zero and the argument is declared with the __local qualifier or if the argument is a sampler and arg_size != sizeof(cl_sampler).
// TODO CL_INVALID_ARG_VALUE if the argument is an image declared with the read_only qualifier and arg_value refers to an image object created with cl_mem_flags of CL_MEM_WRITE or if the image argument is declared with the write_only qualifier and arg_value refers to an image object created with cl_mem_flags of CL_MEM_READ.
// TODO CL_OUT_OF_RESOURCES if there is a failure to allocate resources required by the OpenCL implementation on the device.
// TODO CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required by the OpenCL implementation on the host.

    if (arg_index >= kernel->num_args()) {
        cvk_error_fn("the program has only %u arguments", kernel->num_args());
        return CL_INVALID_ARG_INDEX;
    }

    if ((arg_value == nullptr) && (kernel->arg_kind(arg_index) != kernel_argument_kind::local)) {
        cvk_error_fn("passing a null pointer to clSetKernelArg is only supported for local arguments");
        return CL_INVALID_ARG_VALUE;
    }

    return kernel->set_arg(arg_index, arg_size, arg_value);
}

cl_int clGetKernelInfo(
    cl_kernel      kernel,
    cl_kernel_info param_name,
    size_t         param_value_size,
    void          *param_value,
    size_t        *param_value_size_ret
){
    LOG_API_CALL("kernel = %p, param_name = %x, param_value_size = %zu, param_value = %p, param_value_size_ret = %p", kernel, param_name, param_value_size, param_value, param_value_size_ret);

    cl_int ret = CL_SUCCESS;
    size_t ret_size = 0;
    const void *copy_ptr = nullptr;
    cl_uint val_uint;
    cl_context val_context;
    cl_program val_program;

    if (!is_valid_kernel(kernel)) {
        return CL_INVALID_KERNEL;
    }

    switch (param_name) {
    case CL_KERNEL_REFERENCE_COUNT:
        val_uint = kernel->refcount();
        copy_ptr = &val_uint;
        ret_size = sizeof(val_uint);
        break;
    case CL_KERNEL_CONTEXT:
        val_context = kernel->context();
        copy_ptr = &val_context;
        ret_size = sizeof(val_context);
        break;
    case CL_KERNEL_FUNCTION_NAME:
        copy_ptr = kernel->name().c_str();
        ret_size = kernel->name().size() + 1;
        break;
    case CL_KERNEL_NUM_ARGS:
        val_uint = kernel->num_args();
        copy_ptr = &val_uint;
        ret_size = sizeof(val_uint);
        break;
    case CL_KERNEL_PROGRAM:
        val_program = kernel->program();
        copy_ptr = &val_program;
        ret_size = sizeof(val_program);
        break;
    case CL_KERNEL_ATTRIBUTES: // TODO implement
    default:
        ret = CL_INVALID_VALUE;
    }

    if ((param_value != nullptr) && (copy_ptr != nullptr)) {
        if (param_value_size < ret_size) {
            ret = CL_INVALID_VALUE;
        }
        memcpy(param_value, copy_ptr, std::min(param_value_size, ret_size));
    }

    if (param_value_size_ret != nullptr) {
        *param_value_size_ret = ret_size;
    }

    return ret;
}

cl_int clGetKernelArgInfo(
    cl_kernel          kernel,
    cl_uint            arg_indx,
    cl_kernel_arg_info param_name,
    size_t             param_value_size,
    void              *param_value,
    size_t            *param_value_size_ret
){
    LOG_API_CALL("kernel = %p, arg_indx = %u, param_name = %x, "
                 "param_value_size = %zu, param_value = %p, "
                 "param_value_size_ret = %p",
                 kernel, arg_indx, param_name, param_value_size, param_value,
                 param_value_size_ret);

    return CL_INVALID_OPERATION;
}

cl_int
clGetKernelWorkGroupInfo(
    cl_kernel                  kernel,
    cl_device_id               device,
    cl_kernel_work_group_info  param_name,
    size_t                     param_value_size,
    void *                     param_value,
    size_t *                   param_value_size_ret
){
    LOG_API_CALL("kernel = %p, device = %p, param_name = %x, param_value_size = %zu, "
                 "param_value = %p, param_value_size_ret = %p",
                 kernel, device, param_name, param_value_size, param_value, param_value_size_ret);

    cl_int ret = CL_SUCCESS;
    const void *copy_ptr = nullptr;
    size_t val_sizet, ret_size = 0;
    cl_ulong val_ulong;

    if (!is_valid_kernel(kernel)) {
        return CL_INVALID_KERNEL;
    }

    switch (param_name)
    {
        case CL_KERNEL_WORK_GROUP_SIZE:
            val_sizet = device->vulkan_limits().maxComputeWorkGroupInvocations;
            copy_ptr = &val_sizet;
            ret_size = sizeof(val_sizet);
            break;
        case CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE:
            // FIXME what to do here?
            val_sizet = 4;
            copy_ptr = &val_sizet;
            ret_size = sizeof(val_sizet);
            break;
        case CL_KERNEL_LOCAL_MEM_SIZE:
            val_ulong = kernel->local_mem_size();
            copy_ptr = &val_ulong;
            ret_size = sizeof(val_ulong);
            break;
        case CL_KERNEL_GLOBAL_WORK_SIZE: // TODO
        case CL_KERNEL_COMPILE_WORK_GROUP_SIZE: // TODO
        case CL_KERNEL_PRIVATE_MEM_SIZE: // TODO
        default:
            ret = CL_INVALID_VALUE;
            break;
    }

    if ((param_value != nullptr) && (copy_ptr != nullptr)) {
        if (param_value_size < ret_size) {
            ret = CL_INVALID_VALUE;
        }
        memcpy(param_value, copy_ptr, std::min(param_value_size, ret_size));
    }

    if (param_value_size_ret != nullptr) {
        *param_value_size_ret = ret_size;
    }

    return ret;
}

cl_int
clRetainKernel(
    cl_kernel kernel
){
    LOG_API_CALL("kernel = %p", kernel);

    if (!is_valid_kernel(kernel)) {
        return CL_INVALID_KERNEL;
    }

    kernel->retain();

    return CL_SUCCESS;
}

cl_int
clReleaseKernel(
    cl_kernel   kernel
){
    LOG_API_CALL("kernel = %p", kernel);

    if (!is_valid_kernel(kernel)) {
        return CL_INVALID_KERNEL;
    }

    kernel->release();

    return CL_SUCCESS;
}


/* Profiling APIs  */
cl_int
clGetEventProfilingInfo(
    cl_event            event,
    cl_profiling_info   param_name,
    size_t              param_value_size,
    void *              param_value,
    size_t *            param_value_size_ret
){
    LOG_API_CALL("event = %p, param_name = %x, param_value_size = %zu, "
                 "param_value = %p, param_value_size_ret = %p",
                 event, param_name, param_value_size, param_value, param_value_size_ret);

    if (!is_valid_event(event)) {
        return CL_INVALID_EVENT;
    }

    switch (param_name) {
    case CL_PROFILING_COMMAND_QUEUED:
    case CL_PROFILING_COMMAND_SUBMIT:
    case CL_PROFILING_COMMAND_START:
    case CL_PROFILING_COMMAND_END:
        break;
    default:
        return CL_INVALID_VALUE;
    }

    if ((param_value_size < sizeof(cl_ulong)) && (param_value != nullptr)) {
        return CL_INVALID_VALUE;
    }

    if (event->is_user_event() || (event->get_status() != CL_COMPLETE)) {
        return CL_PROFILING_INFO_NOT_AVAILABLE;
    }

    if (!event->queue()->has_property(CL_QUEUE_PROFILING_ENABLE)) {
        return CL_PROFILING_INFO_NOT_AVAILABLE;
    }

    if (param_value_size_ret != nullptr) {
        *param_value_size_ret = sizeof(cl_ulong);
    }

    if (param_value != nullptr) {
        cl_ulong value = event->get_profiling_info(param_name);
        memcpy(param_value, &value, sizeof(cl_ulong));
    }

    return CL_SUCCESS;
}
               
/* Flush and Finish APIs */
cl_int
clFlush(
    cl_command_queue command_queue
){
    LOG_API_CALL("command_queue = %p", command_queue);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    return command_queue->flush();
}

cl_int
clFinish(
    cl_command_queue command_queue
){
    LOG_API_CALL("command_queue = %p", command_queue);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    cvk_event *event = nullptr;
    cl_int status = command_queue->flush(&event);

    if ((status == CL_SUCCESS) && (event != nullptr)) {
        event->wait();
        event->release();
    }

    return status;
}

/* Enqueued Commands APIs */

cl_int
clEnqueueReadBuffer(
    cl_command_queue    command_queue,
    cl_mem              buffer,
    cl_bool             blocking_read,
    size_t              offset,
    size_t              size,
    void *              ptr,
    cl_uint             num_events_in_wait_list,
    const cl_event *    event_wait_list,
    cl_event *          event
){
    LOG_API_CALL("command_queue = %p, buffer = %p, blocking = %d, offset = %zu, size = %zu, ptr = %p",
                 command_queue, buffer, blocking_read, offset, size, ptr);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    if (!is_valid_buffer(buffer)) {
        return CL_INVALID_MEM_OBJECT;
    }

    if (buffer->has_any_flag(CL_MEM_HOST_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS)) {
        return CL_INVALID_OPERATION;
    }

    if (!event_wait_list_is_valid(num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    auto cmd = new cvk_command_copy(command_queue, CL_COMMAND_READ_BUFFER, buffer, ptr, offset, size);

    auto err = command_queue->enqueue_command_with_deps(cmd, blocking_read,
                                                        num_events_in_wait_list,
                                                        event_wait_list, event);

    return err;
}

cl_int
clEnqueueWriteBuffer(
    cl_command_queue command_queue,
    cl_mem           buffer,
    cl_bool          blocking_write,
    size_t           offset,
    size_t           size,
    const void      *ptr,
    cl_uint          num_events_in_wait_list,
    const cl_event  *event_wait_list,
    cl_event        *event
){
    LOG_API_CALL("command_queue = %p, buffer = %p, blocking = %d, offset = %zu, size = %zu, ptr = %p",
                 command_queue, buffer, blocking_write, offset, size, ptr);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    // TODO validate the contexts

    if (!is_valid_buffer(buffer)) {
        return CL_INVALID_MEM_OBJECT;
    }

    if (buffer->has_any_flag(CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_NO_ACCESS)) {
        return CL_INVALID_OPERATION;
    }

    if (!event_wait_list_is_valid(num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    auto cmd = new cvk_command_copy(command_queue, CL_COMMAND_WRITE_BUFFER, buffer, ptr, offset, size);

    auto err = command_queue->enqueue_command_with_deps(cmd, blocking_write,
                                                        num_events_in_wait_list,
                                                        event_wait_list, event);

    return err;
}

cl_int
clEnqueueReadBufferRect(
    cl_command_queue command_queue,
    cl_mem           buffer,
    cl_bool          blocking_read,
    const size_t    *buffer_origin,
    const size_t    *host_origin,
    const size_t    *region,
    size_t           buffer_row_pitch,
    size_t           buffer_slice_pitch,
    size_t           host_row_pitch,
    size_t           host_slice_pitch,
    void            *ptr,
    cl_uint          num_events_in_wait_list,
    const cl_event  *event_wait_list,
    cl_event        *event
){
    LOG_API_CALL("command_queue = %p, buffer = %p, blocking = %d",
                 command_queue, buffer, blocking_read);
    LOG_API_CALL("buffer_origin = {%zu,%zu,%zu}, host_origin = {%zu,%zu,%zu}, region = {%zu,%zu,%zu}",
                 buffer_origin[0], buffer_origin[1], buffer_origin[2],
                 host_origin[0], host_origin[1], host_origin[2],
                 region[0], region[1], region[2]);
    LOG_API_CALL("buffer_row_pitch = %zu, buffer_slice_pitch = %zu,"
                 "host_row_pitch = %zu, host_slice_pitch = %zu",
                 buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch);
    LOG_API_CALL("ptr = %p, num_events = %u, event_wait_list = %p, event = %p",
                 ptr, num_events_in_wait_list, event_wait_list, event);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    if (!is_valid_buffer(buffer)) {
        return CL_INVALID_MEM_OBJECT;
    }

    if (buffer->has_any_flag(CL_MEM_HOST_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS)) {
        return CL_INVALID_OPERATION;
    }

    if (!event_wait_list_is_valid(num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    auto buf = static_cast<cvk_buffer*>(buffer);

    auto cmd = new cvk_command_copy_host_buffer_rect(
        command_queue, CL_COMMAND_READ_BUFFER_RECT, buf, ptr, host_origin,
        buffer_origin, region, host_row_pitch, host_slice_pitch,
        buffer_row_pitch, buffer_slice_pitch);

    auto err = command_queue->enqueue_command_with_deps(cmd, blocking_read,
                                                        num_events_in_wait_list,
                                                        event_wait_list, event);

    return err;
}

cl_int clEnqueueWriteBufferRect(
    cl_command_queue command_queue,
    cl_mem           buffer,
    cl_bool          blocking_write,
    const size_t    *buffer_origin,
    const size_t    *host_origin,
    const size_t    *region,
    size_t           buffer_row_pitch,
    size_t           buffer_slice_pitch,
    size_t           host_row_pitch,
    size_t           host_slice_pitch,
    const void      *ptr,
    cl_uint          num_events_in_wait_list,
    const cl_event  *event_wait_list,
    cl_event        *event
){
    LOG_API_CALL("command_queue = %p, buffer = %p, blocking = %d",
                 command_queue, buffer, blocking_write);
    LOG_API_CALL("buffer_origin = {%zu,%zu,%zu}, host_origin = {%zu,%zu,%zu}, region = {%zu,%zu,%zu}",
                 buffer_origin[0], buffer_origin[1], buffer_origin[2],
                 host_origin[0], host_origin[1], host_origin[2],
                 region[0], region[1], region[2]);
    LOG_API_CALL("buffer_row_pitch = %zu, buffer_slice_pitch = %zu, "
                 "host_row_pitch = %zu, host_slice_pitch = %zu",
                 buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch);
    LOG_API_CALL("ptr = %p, num_events = %u, event_wait_list = %p, event = %p",
                 ptr, num_events_in_wait_list, event_wait_list, event);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    if (!is_valid_buffer(buffer)) {
        return CL_INVALID_MEM_OBJECT;
    }

    if (buffer->has_any_flag(CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_NO_ACCESS)) {
        return CL_INVALID_OPERATION;
    }

    if (!event_wait_list_is_valid(num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    auto buf = static_cast<cvk_buffer*>(buffer);

    auto cmd = new cvk_command_copy_host_buffer_rect(
        command_queue, CL_COMMAND_WRITE_BUFFER_RECT, buf, const_cast<void*>(ptr), host_origin,
        buffer_origin, region, host_row_pitch, host_slice_pitch,
        buffer_row_pitch, buffer_slice_pitch);

    auto err = command_queue->enqueue_command_with_deps(cmd, blocking_write,
                                                        num_events_in_wait_list,
                                                        event_wait_list, event);

    return err;
}

cl_int clEnqueueFillBuffer(
    cl_command_queue command_queue,
    cl_mem           buffer,
    const void      *pattern,
    size_t           pattern_size,
    size_t           offset,
    size_t           size,
    cl_uint          num_events_in_wait_list,
    const cl_event  *event_wait_list,
    cl_event        *event
){
    LOG_API_CALL("command_queue = %p, buffer = %p, pattern = %p, pattern_size = %zu,"
                 "offset = %zu, size = %zu, num_events = %u, event_wait_list = %p, event = %p",
                 command_queue, buffer, pattern, pattern_size, offset, size,
                 num_events_in_wait_list, event_wait_list, event);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    if (!is_valid_buffer(buffer)) {
        return CL_INVALID_MEM_OBJECT;
    }

    // TODO check context
    // TODO check buffer bounds

    if (pattern == nullptr) {
        return CL_INVALID_VALUE;
    }

    // Check the pattern size is valid
    size_t valid_pattern_sizes[] = {1,2,4,8,16,32,64,128};
    bool pattern_size_valid = false;
    for (auto size : valid_pattern_sizes) {
        if (size == pattern_size) {
            pattern_size_valid = true;
            break;
        }
    }
    if (!pattern_size_valid) {
        return CL_INVALID_VALUE;
    }

    // Check that offset and size are a multiple of pattern_size
    if ((offset % pattern_size != 0) || (size % pattern_size != 0)) {
        return CL_INVALID_VALUE;
    }

    if (!event_wait_list_is_valid(num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    // TODO check sub-buffer alignment

    auto cmd = new cvk_command_fill(command_queue, buffer, offset, size, pattern, pattern_size);

    command_queue->enqueue_command_with_deps(cmd, num_events_in_wait_list, event_wait_list, event);

    return CL_SUCCESS;
}

cl_int clEnqueueCopyBuffer(
    cl_command_queue command_queue,
    cl_mem           src_buffer,
    cl_mem           dst_buffer,
    size_t           src_offset,
    size_t           dst_offset,
    size_t           size,
    cl_uint          num_events_in_wait_list,
    const cl_event  *event_wait_list,
    cl_event        *event
){
    LOG_API_CALL("command_queue = %p, src_buffer = %p, dst_buffer = %p, src_offset = %zu,"
                 "dst_offset = %zu, size = %zu, num_events = %u, event_wait_list = %p, event = %p",
                 command_queue, src_buffer, dst_buffer, src_offset, dst_offset, size,
                 num_events_in_wait_list, event_wait_list, event);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    // TODO validate the contexts

    if (!is_valid_buffer(src_buffer) || !is_valid_buffer(dst_buffer)) {
        return CL_INVALID_MEM_OBJECT;
    }

    if (!event_wait_list_is_valid(num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    auto cmd = new cvk_command_copy_buffer(command_queue, CL_COMMAND_COPY_BUFFER, src_buffer, dst_buffer,
                                           src_offset, dst_offset, size);

    command_queue->enqueue_command_with_deps(cmd, num_events_in_wait_list, event_wait_list, event);

    return CL_SUCCESS;
}

cl_int clEnqueueCopyBufferRect(
    cl_command_queue command_queue,
    cl_mem           src_buffer,
    cl_mem           dst_buffer,
    const size_t    *src_origin,
    const size_t    *dst_origin,
    const size_t    *region,
    size_t           src_row_pitch,
    size_t           src_slice_pitch,
    size_t           dst_row_pitch,
    size_t           dst_slice_pitch,
    cl_uint          num_events_in_wait_list,
    const cl_event  *event_wait_list,
    cl_event        *event
){
    LOG_API_CALL("command_queue = %p, src_buffer = %p, dst_buffer = %p, "
                 "src_origin = {%zu,%zu,%zu}, dst_origin = {%zu,%zu,%zu}, "
                 "region = {%zu,%zu,%zu}, src_row_pitch = %zu, "
                 "src_slice_pitch = %zu, dst_row_pitch = %zu, "
                 "dst_slice_pitch = %zu, num_events_in_wait_list = %u, "
                 "event_wait_list = %p, event = %p",
                 command_queue, src_buffer, dst_buffer, src_origin[0],
                 src_origin[1], src_origin[2], dst_origin[0], dst_origin[1],
                 dst_origin[2], region[0], region[1], region[2], src_row_pitch,
                 src_slice_pitch, dst_row_pitch, dst_slice_pitch,
                 num_events_in_wait_list, event_wait_list, event);

    // TODO CL_INVALID_COMMAND_QUEUE if command_queue is not a valid command-queue.

    if (!is_valid_buffer(src_buffer) || !is_valid_buffer(dst_buffer)) {
        return CL_INVALID_MEM_OBJECT;
    }
    // TODO CL_INVALID_VALUE if (src_offset, region) or (dst_offset, region) require accessing elements outside the src_buffer and dst_buffer objects respectively.
    // TODO CL_INVALID_VALUE if any region array element is 0.
    // TODO CL_INVALID_VALUE if src_row_pitch is not 0 and is less than region[0].
    // TODO CL_INVALID_VALUE if dst_row_pitch is not 0 and is less than region[0].
    // TODO CL_INVALID_VALUE if src_slice_pitch is not 0 and is less than region[1] * src_row_pitch or if src_slice_pitch is not 0 and is not a multiple of src_row_pitch.
    // TODO CL_INVALID_VALUE if dst_slice_pitch is not 0 and is less than region[1] * dst_row_pitch or if dst_slice_pitch is not 0 and is not a multiple of dst_row_pitch.
    // TODO CL_INVALID_VALUE if src_buffer and dst_buffer are the same buffer object and src_slice_pitch is not equal to dst_slice_pitch and src_row_pitch is not equal to dst_row_pitch.
    //
    if (!event_wait_list_is_valid(num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    if (!is_same_context(command_queue, src_buffer) ||
        !is_same_context(command_queue, dst_buffer) ||
        !is_same_context(command_queue, num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_CONTEXT;
    }
    // TODO CL_MEM_COPY_OVERLAP if src_buffer and dst_buffer are the same buffer object and the source and destination regions overlap or if src_buffer and dst_buffer are different sub-buffers of the same associated buffer object and they overlap. Refer to Appendix E in the OpenCL specification for details on how to determine if source and destination regions overlap.
    // TODO CL_MISALIGNED_SUB_BUFFER_OFFSET if src_buffer is a sub-buffer object and offset specified when the sub-buffer object is created is not aligned to CL_DEVICE_MEM_BASE_ADDR_ALIGN value for device associated with queue.
    // TODO CL_MISALIGNED_SUB_BUFFER_OFFSET if dst_buffer is a sub-buffer object and offset specified when the sub-buffer object is created is not aligned to CL_DEVICE_MEM_BASE_ADDR_ALIGN value for device associated with queue.
    // TODO CL_MEM_OBJECT_ALLOCATION_FAILURE if there is a failure to allocate memory for data store associated with src_buffer or dst_buffer.
    // TODO CL_OUT_OF_RESOURCES if there is a failure to allocate resources required by the OpenCL implementation on the device.
    // TODO CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required by the OpenCL implementation on the host.
    //
    auto srcbuf = static_cast<cvk_buffer*>(src_buffer);
    auto dstbuf = static_cast<cvk_buffer*>(dst_buffer);
    auto cmd = new cvk_command_copy_buffer_rect(command_queue, srcbuf, dstbuf,
                                                src_origin, dst_origin, region,
                                                src_row_pitch, src_slice_pitch,
                                                dst_row_pitch, dst_slice_pitch);
    command_queue->enqueue_command_with_deps(cmd, num_events_in_wait_list,
                                             event_wait_list, event);

    return CL_SUCCESS;
}

void *
clEnqueueMapBuffer(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_map,
    cl_map_flags map_flags,
    size_t offset,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event,
    cl_int *errcode_ret
){
    LOG_API_CALL("command_queue = %p, buffer = %p, offset = %zu, size = %zu",
                 command_queue, buffer, offset, size);

    if (!is_valid_command_queue(command_queue)) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_COMMAND_QUEUE;
        }
        return nullptr;
    }

    if (!is_valid_buffer(buffer)) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_MEM_OBJECT;
        }
        return nullptr;
    }

    if (!event_wait_list_is_valid(num_events_in_wait_list, event_wait_list)) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_EVENT_WAIT_LIST;
        }
        return nullptr;
    }

    if (!is_same_context(command_queue, buffer)) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_CONTEXT;
        }
        return nullptr;
    }

    if ((size == 0) || (offset + size > buffer->size())) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_VALUE;
        }
        return nullptr;
    }

    if (!map_flags_are_valid(map_flags)) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_VALUE;
        }
        return nullptr;
    }

    // TODO CL_MISALIGNED_SUB_BUFFER_OFFSET if buffer is a sub-buffer object and offset specified when the sub-buffer object is created is not aligned to CL_DEVICE_MEM_BASE_ADDR_ALIGN value for device associated with queue.

    if ((map_flags & CL_MAP_READ) && (buffer->has_any_flag(CL_MEM_HOST_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS))) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_OPERATION;
        }
        return nullptr;
    }

    if (((map_flags & CL_MAP_WRITE) || (map_flags & CL_MAP_WRITE_INVALIDATE_REGION)) &&
        (buffer->has_any_flag(CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_NO_ACCESS))) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_OPERATION;
        }
        return nullptr;
    }

    // TODO enqueue barriers to VK command buffer
    // TODO handle map flags

    // FIXME This error cannot occur for objects created with CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR.
    if (!buffer->map()) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_MAP_FAILURE;
        }
        return nullptr;
    }

    auto buf = static_cast<cvk_buffer*>(buffer);
    auto map_ptr = buf->map_ptr(offset);
    auto cmd = new cvk_command_map_buffer(command_queue, buf, offset, size);

    auto err = command_queue->enqueue_command_with_deps(cmd, blocking_map,
                                                        num_events_in_wait_list,
                                                        event_wait_list, event);

    if (err != CL_SUCCESS) {
        buffer->unmap();
    }

    if (errcode_ret != nullptr) {
        *errcode_ret = err;
    }

    return map_ptr;
}

cl_int
clEnqueueUnmapMemObject(
    cl_command_queue  command_queue,
    cl_mem            memobj,
    void             *mapped_ptr,
    cl_uint           num_events_in_wait_list,
    const cl_event   *event_wait_list,
    cl_event         *event
){
    LOG_API_CALL("command_queue = %p, memobj = %p, mapped_ptr = %p",
                 command_queue, memobj, mapped_ptr);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    if (!is_valid_mem_object(memobj)) {
        return CL_INVALID_MEM_OBJECT;
    }

    auto buffer = static_cast<cvk_buffer*>(memobj);
    auto cmd = new cvk_command_unmap_buffer(command_queue, buffer);

    command_queue->enqueue_command_with_deps(cmd, num_events_in_wait_list, event_wait_list, event);

    return CL_SUCCESS;
}

cl_int
cvk_enqueue_ndrange_kernel(
    cl_command_queue command_queue,
    cl_kernel        kernel,
    uint32_t        *num_workgroups,
    uint32_t        *workgroup_size,
    cl_uint          num_events_in_wait_list,
    const cl_event * event_wait_list,
    cl_event *       event
){
    auto cmd = new cvk_command_kernel(command_queue, kernel, num_workgroups, workgroup_size);

    cl_int err = cmd->build();

    if (err != CL_SUCCESS) {
        return err;
    }

    command_queue->enqueue_command_with_deps(cmd, num_events_in_wait_list, event_wait_list, event);

    return CL_SUCCESS;
}

cl_int clEnqueueTask(
    cl_command_queue command_queue,
    cl_kernel        kernel,
    cl_uint          num_events_in_wait_list,
    const cl_event  *event_wait_list,
    cl_event        *event
){
    LOG_API_CALL("command_queue = %p, kernel = %p, num_events_in_wait_list = %d,"
                 " event_wait_list = %p, event = %p",
                 command_queue, kernel, num_events_in_wait_list, event_wait_list, event);

    uint32_t num_workgroups[] = {1, 1, 1};
    uint32_t workgroup_size[] = {1, 1, 1};

    return cvk_enqueue_ndrange_kernel(command_queue, kernel, num_workgroups, workgroup_size,
                                      num_events_in_wait_list, event_wait_list, event);
}

cl_int
clEnqueueNDRangeKernel(
    cl_command_queue command_queue,
    cl_kernel        kernel,
    cl_uint          work_dim,
    const size_t *   global_work_offset,
    const size_t *   global_work_size,
    const size_t *   local_work_size,
    cl_uint          num_events_in_wait_list,
    const cl_event * event_wait_list,
    cl_event *       event
){
    LOG_API_CALL("command_queue = %p, kernel = %p", command_queue, kernel);
    LOG_API_CALL("work_dim = %u", work_dim);

    uint32_t gws[3] = {1, 1, 1};
    uint32_t lws[3] = {1, 1, 1}; // FIXME pick a sensible default

    for (cl_uint i = 0; i < work_dim; i++) {
        gws[i] = global_work_size[i];
        if (local_work_size != nullptr) {
            lws[i] = local_work_size[i];
        }
    }

    for (int i = 0; i < 3; i++) {
        LOG_API_CALL("gws[%d] = %u", i, gws[i]);
    }

    for (int i = 0; i < 3; i++) {
        LOG_API_CALL("lws[%d] = %u", i, lws[i]);
    }

    // TODO support global offset
    if (global_work_offset != nullptr) {
        cvk_error_fn("global offset unsupported");
        return CL_INVALID_GLOBAL_OFFSET;
    }

    // TODO CL_INVALID_WORK_GROUP_SIZE lws does not match the work-group size specified for kernel using the __attribute__ ((reqd_work_group_size(X, Y, Z))) qualifier in program source. 

    // Check uniformity of the NDRange
    for (cl_uint i = 0; i < 3; i++) {
        if (gws[i] % lws[i] != 0) {
            return CL_INVALID_WORK_GROUP_SIZE;
        }
    }

    // Calculate dispatch size
    uint32_t num_workgroups[3];
    for (cl_uint i = 0; i < 3; i++) {
        num_workgroups[i] = gws[i] / lws[i];
    };

    return cvk_enqueue_ndrange_kernel(command_queue, kernel, num_workgroups, lws,
                                      num_events_in_wait_list, event_wait_list, event);
}

cl_int clEnqueueNativeKernel(
    cl_command_queue command_queue,
    void (*user_func)(void *),
    void            *args,
    size_t           cb_args,
    cl_uint          num_mem_objects,
    const cl_mem    *mem_list,
    const void     **args_mem_loc,
    cl_uint          num_events_in_wait_list,
    const cl_event  *event_wait_list,
    cl_event        *event
){
    LOG_API_CALL("command_queue = %p, user_func = %p, args = %p, cb_args = %zu, "
                 "num_mem_objects = %u, mem_list = %p, args_mem_loc = %p, "
                 "num_events_in_wait_list = %u, event_wait_list = %p, event = %p",
                 command_queue, user_func, args, cb_args, num_mem_objects, mem_list,
                 args_mem_loc, num_events_in_wait_list, event_wait_list, event);

    return CL_INVALID_OPERATION;
}

cl_sampler clCreateSampler(
    cl_context         context,
    cl_bool            normalized_coords,
    cl_addressing_mode addressing_mode,
    cl_filter_mode     filter_mode,
    cl_int            *errcode_ret
){
    LOG_API_CALL("context = %p, normalized_coords = %d, addressing_mode = %d, "
                 "filter_mode = %d, errcode_ret = %p",
                 context, normalized_coords, addressing_mode, filter_mode, errcode_ret);

    if (!is_valid_context(context)) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_CONTEXT;
        }
        return nullptr;
    }

    auto sampler = cvk_sampler::create(context, normalized_coords, addressing_mode, filter_mode);

    cl_int err = CL_SUCCESS;

    if (sampler == nullptr) {
        err = CL_OUT_OF_RESOURCES;
    }

    if (errcode_ret != nullptr) {
        *errcode_ret = err;
    }

    return sampler;
}

cl_int clRetainSampler(
    cl_sampler sampler
){
    LOG_API_CALL("sampler = %p", sampler);

    if (!is_valid_sampler(sampler)) {
        return CL_INVALID_SAMPLER;
    }

    sampler->retain();

    return CL_SUCCESS;
}

cl_int clReleaseSampler(
    cl_sampler sampler
){
    LOG_API_CALL("sampler = %p", sampler);

    if (!is_valid_sampler(sampler)) {
        return CL_INVALID_SAMPLER;
    }

    sampler->release();

    return CL_SUCCESS;
}

cl_int clGetSamplerInfo(
    cl_sampler      sampler,
    cl_sampler_info param_name,
    size_t          param_value_size,
    void           *param_value,
    size_t         *param_value_size_ret
){
    LOG_API_CALL("sampler = %p, param_name = %d, param_value_size = %zu, "
                 "param_value = %p, param_value_size_ret = %p",
                 sampler, param_name, param_value_size, param_value, param_value_size_ret);

    cl_int ret = CL_SUCCESS;
    size_t ret_size = 0;
    const void *copy_ptr = nullptr;
    cl_uint val_uint;
    cl_context val_context;
    cl_bool val_bool;
    cl_addressing_mode val_addressing_mode;
    cl_filter_mode val_filter_mode;

    if (!is_valid_sampler(sampler)) {
        return CL_INVALID_SAMPLER;
    }

    switch (param_name) {
    case CL_SAMPLER_REFERENCE_COUNT:
        val_uint = sampler->refcount();
        copy_ptr = &val_uint;
        ret_size = sizeof(val_uint);
        break;
    case CL_SAMPLER_CONTEXT:
        val_context = sampler->context();
        copy_ptr = &val_context;
        ret_size = sizeof(val_context);
        break;
    case CL_SAMPLER_NORMALIZED_COORDS:
        val_bool = sampler->normalized_coords();
        copy_ptr = &val_bool;
        ret_size = sizeof(val_bool);
        break;
    case CL_SAMPLER_ADDRESSING_MODE:
        val_addressing_mode = sampler->addressing_mode();
        copy_ptr = &val_addressing_mode;
        ret_size = sizeof(val_addressing_mode);
        break;
    case CL_SAMPLER_FILTER_MODE:
        val_filter_mode = sampler->filter_mode();
        copy_ptr = &val_filter_mode;
        ret_size = sizeof(val_filter_mode);
        break;
    default:
        ret = CL_INVALID_VALUE;
    }

    if ((param_value != nullptr) && (copy_ptr != nullptr)) {
        if (param_value_size < ret_size) {
            ret = CL_INVALID_VALUE;
        }
        memcpy(param_value, copy_ptr, std::min(param_value_size, ret_size));
    }

    if (param_value_size_ret != nullptr) {
        *param_value_size_ret = ret_size;
    }

    return ret;
}

cl_mem cvk_create_image(
    cl_context             context,
    cl_mem_flags           flags,
    const cl_image_format *image_format,
    const cl_image_desc   *image_desc,
    void                  *host_ptr,
    cl_int                *errcode_ret
){
    // TODO CL_INVALID_CONTEXT if context is not a valid context.
    // TODO CL_INVALID_VALUE if values specified in flags are not valid.
    // TODO CL_INVALID_IMAGE_FORMAT_DESCRIPTOR if values specified in image_format are not valid or if image_format is NULL.
    // TODO CL_INVALID_IMAGE_DESCRIPTOR if values specified in image_desc are not valid or if image_desc is NULL.
    // TODO CL_INVALID_IMAGE_SIZE if image dimensions specified in image_desc exceed the minimum maximum image dimensions described in the table of allowed values for param_name for clGetDeviceInfo for all devices in context.
    // TODO CL_INVALID_HOST_PTR if host_ptr in image_desc is NULL and CL_MEM_USE_HOST_PTR or CL_MEM_COPY_HOST_PTR are set in flags or if host_ptr is not NULL but CL_MEM_COPY_HOST_PTR or CL_MEM_USE_HOST_PTR are not set in flags.
    // TODO CL_INVALID_VALUE if a 1D image buffer is being created and the buffer object was created with CL_MEM_WRITE_ONLY and flags specifies CL_MEM_READ_WRITE or CL_MEM_READ_ONLY, or if the buffer object was created with CL_MEM_READ_ONLY and flags specifies CL_MEM_READ_WRITE or CL_MEM_WRITE_ONLY, or if flags specifies CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR or CL_MEM_COPY_HOST_PTR.
    // TODO CL_INVALID_VALUE if a 1D image buffer is being created and the buffer object was created with CL_MEM_HOST_WRITE_ONLY and flags specifies CL_MEM_HOST_READ_ONLY, or if the buffer object was created with CL_MEM_HOST_READ_ONLY and flags specifies CL_MEM_HOST_WRITE_ONLY, or if the buffer object was created with CL_MEM_HOST_NO_ACCESS and flags specifies CL_MEM_HOST_READ_ONLY or CL_MEM_HOST_WRITE_ONLY.
    // TODO CL_IMAGE_FORMAT_NOT_SUPPORTED if the image_format is not supported.
    // TODO CL_MEM_OBJECT_ALLOCATION_FAILURE if there is a failure to allocate memory for image object.

    auto image = cvk_image::create(context, flags, image_desc, image_format, host_ptr);

    *errcode_ret = (image != nullptr) ? CL_SUCCESS : CL_OUT_OF_RESOURCES; // FIXME do this properly

    return image;
}

cl_mem clCreateImage(
    cl_context             context,
    cl_mem_flags           flags,
    const cl_image_format *image_format,
    const cl_image_desc   *image_desc,
    void                  *host_ptr,
    cl_int                *errcode_ret
){
    LOG_API_CALL("context = %p, flags = %lu, image_format = %p, image_desc = %p,"
                 " host_ptr = %p, errcode_ret = %p",
                 context, flags, image_format, image_desc, host_ptr, errcode_ret);

    cl_int err;
    auto image = cvk_create_image(context, flags, image_format, image_desc, host_ptr, &err);

    if (errcode_ret != nullptr) {
        *errcode_ret = err;
    }

    return image;
}

cl_mem clCreateImage2D(
    cl_context             context,
    cl_mem_flags           flags,
    const cl_image_format *image_format,
    size_t                 image_width,
    size_t                 image_height,
    size_t                 image_row_pitch,
    void                  *host_ptr,
    cl_int                *errcode_ret
){
    LOG_API_CALL("context = %p, flags = %lu, image_format = %p, image_width = %zu, "
                 "image_height = %zu, image_row_pitch = %zu, host_ptr = %p, "
                 "errcode_ret = %p",
                 context, flags, image_format, image_width, image_height,
                 image_row_pitch, host_ptr, errcode_ret);

    cl_image_desc desc = {
        CL_MEM_OBJECT_IMAGE2D,
        image_width,
        image_height,
        0, // image_depth
        0, // image_array_size
        image_row_pitch,
        0, // image_slice_pitch
        0, // num_mip_levels
        0, // num_samples
        nullptr // buffer
    };

    cl_int err;
    auto image = cvk_create_image(context, flags, image_format, &desc, host_ptr, &err);

    if (errcode_ret != nullptr) {
        *errcode_ret = err;
    }

    return image;
}

cl_mem clCreateImage3D(
    cl_context             context,
    cl_mem_flags           flags,
    const cl_image_format *image_format,
    size_t                 image_width,
    size_t                 image_height,
    size_t                 image_depth,
    size_t                 image_row_pitch,
    size_t                 image_slice_pitch,
    void                  *host_ptr,
    cl_int                *errcode_ret
){
    LOG_API_CALL("context = %p, flags = %lu, image_format = %p, image_width = %zu, "
                 "image_height = %zu, image_depth = %zu, image_row_pitch = %zu, "
                 "image_slice_pitch = %zu, host_ptr = %p, errcode_ret = %p",
                 context, flags, image_format, image_width, image_height,
                 image_depth, image_row_pitch, image_slice_pitch, host_ptr,
                 errcode_ret);

    cl_image_desc desc = {
        CL_MEM_OBJECT_IMAGE3D,
        image_width,
        image_height,
        image_depth,
        0, // image_array_size
        image_row_pitch,
        image_slice_pitch,
        0, // num_mip_levels
        0, // num_samples
        nullptr // buffer
    };

    cl_int err;
    auto image = cvk_create_image(context, flags, image_format, &desc, host_ptr, &err);

    if (errcode_ret != nullptr) {
        *errcode_ret = err;
    }

    return image;
}
cl_int clGetImageInfo(
    cl_mem        image,
    cl_image_info param_name,
    size_t        param_value_size,
    void         *param_value,
    size_t       *param_value_size_ret
){
    LOG_API_CALL("image = %p, param_name = %x, param_value_size = %zu, "
                 "param_value = %p, param_value_size_ret = %p",
                 image, param_name, param_value_size, param_value, param_value_size_ret);

    cl_int ret = CL_SUCCESS;
    size_t ret_size = 0;
    const void *copy_ptr = nullptr;
    cl_image_format val_image_format;
    size_t val_sizet;
    cl_mem val_mem;
    cl_uint val_uint;

    if (!is_valid_image(image)) {
        return CL_INVALID_MEM_OBJECT;
    }

    auto img = static_cast<cvk_image*>(image);

    switch (param_name) {
    case CL_IMAGE_FORMAT:
        val_image_format = img->format();
        copy_ptr = &val_image_format;
        ret_size = sizeof(val_image_format);
        break;
    case CL_IMAGE_ELEMENT_SIZE:
        val_sizet = img->element_size();
        copy_ptr = &val_sizet;
        ret_size = sizeof(val_sizet);
        break;
    case CL_IMAGE_ROW_PITCH:
        val_sizet = img->row_pitch();
        copy_ptr = &val_sizet;
        ret_size = sizeof(val_sizet);
        break;
    case CL_IMAGE_SLICE_PITCH:
        val_sizet = img->slice_pitch();
        copy_ptr = &val_sizet;
        ret_size = sizeof(val_sizet);
        break;
    case CL_IMAGE_WIDTH:
        val_sizet = img->width();
        copy_ptr = &val_sizet;
        ret_size = sizeof(val_sizet);
        break;
    case CL_IMAGE_HEIGHT:
        val_sizet = img->height();
        copy_ptr = &val_sizet;
        ret_size = sizeof(val_sizet);
        break;
    case CL_IMAGE_DEPTH:
        val_sizet = img->depth();
        copy_ptr = &val_sizet;
        ret_size = sizeof(val_sizet);
        break;
    case CL_IMAGE_ARRAY_SIZE:
        val_sizet = img->array_size();
        copy_ptr = &val_sizet;
        ret_size = sizeof(val_sizet);
        break;
    case CL_IMAGE_BUFFER:
        val_mem = img->buffer();
        copy_ptr = &val_mem;
        ret_size = sizeof(val_mem);
        break;
    case CL_IMAGE_NUM_MIP_LEVELS:
        val_uint = img->num_mip_levels();
        copy_ptr = &val_uint;
        ret_size = sizeof(val_uint);
        break;
    case CL_IMAGE_NUM_SAMPLES:
        val_uint = img->num_samples();
        copy_ptr = &val_uint;
        ret_size = sizeof(val_uint);
        break;
    default:
        ret = CL_INVALID_VALUE;
    }

    if ((param_value != nullptr) && (copy_ptr != nullptr)) {
        if (param_value_size < ret_size) {
            ret = CL_INVALID_VALUE;
        }
        memcpy(param_value, copy_ptr, std::min(param_value_size, ret_size));
    }

    if (param_value_size_ret != nullptr) {
        *param_value_size_ret = ret_size;
    }

    return ret;
}

std::unordered_map<VkFormat, cl_image_format> gFormatMaps = {
    // R formats
    {VK_FORMAT_R8_UNORM,  {CL_R, CL_UNORM_INT8}},
    {VK_FORMAT_R8_SNORM,  {CL_R, CL_SNORM_INT8}},
    {VK_FORMAT_R8_UINT,   {CL_R, CL_UNSIGNED_INT8}},
    {VK_FORMAT_R8_SINT,   {CL_R, CL_SIGNED_INT8}},
    {VK_FORMAT_R16_UNORM, {CL_R, CL_UNORM_INT16}},
    {VK_FORMAT_R16_SNORM, {CL_R, CL_SNORM_INT16}},
    {VK_FORMAT_R16_UINT,  {CL_R, CL_UNSIGNED_INT16}},
    {VK_FORMAT_R16_SINT,  {CL_R, CL_SIGNED_INT16}},
    {VK_FORMAT_R16_SFLOAT,{CL_R, CL_HALF_FLOAT}},
    {VK_FORMAT_R32_UINT,  {CL_R, CL_UNSIGNED_INT32}},
    {VK_FORMAT_R32_SINT,  {CL_R, CL_SIGNED_INT32}},
    {VK_FORMAT_R32_SFLOAT,{CL_R, CL_FLOAT}},

    // RG formats
    {VK_FORMAT_R8G8_UNORM,    {CL_RG, CL_UNORM_INT8}},
    {VK_FORMAT_R8G8_SNORM,    {CL_RG, CL_SNORM_INT8}},
    {VK_FORMAT_R8G8_UINT,     {CL_RG, CL_UNSIGNED_INT8}},
    {VK_FORMAT_R8G8_SINT,     {CL_RG, CL_SIGNED_INT8}},
    {VK_FORMAT_R16G16_UNORM,  {CL_RG, CL_UNORM_INT16}},
    {VK_FORMAT_R16G16_SNORM,  {CL_RG, CL_SNORM_INT16}},
    {VK_FORMAT_R16G16_UINT,   {CL_RG, CL_UNSIGNED_INT16}},
    {VK_FORMAT_R16G16_SINT,   {CL_RG, CL_SIGNED_INT16}},
    {VK_FORMAT_R16G16_SFLOAT, {CL_RG, CL_HALF_FLOAT}},
    {VK_FORMAT_R32G32_UINT,   {CL_RG, CL_UNSIGNED_INT32}},
    {VK_FORMAT_R32G32_SINT,   {CL_RG, CL_SIGNED_INT32}},
    {VK_FORMAT_R32G32_SFLOAT, {CL_RG, CL_FLOAT}},

    // RGB formats
    {VK_FORMAT_R8G8B8_UNORM, {CL_RGB, CL_UNORM_INT8}},
    {VK_FORMAT_R8G8B8_SNORM, {CL_RGB, CL_SNORM_INT8}},
    {VK_FORMAT_R8G8B8_UINT,  {CL_RGB, CL_UNSIGNED_INT8}},
    {VK_FORMAT_R8G8B8_SINT,  {CL_RGB, CL_SIGNED_INT8}},
    {VK_FORMAT_R16G16B16_UNORM,  {CL_RGB, CL_UNORM_INT16}},
    {VK_FORMAT_R16G16B16_SNORM,  {CL_RGB, CL_SNORM_INT16}},
    {VK_FORMAT_R16G16B16_UINT,   {CL_RGB, CL_UNSIGNED_INT16}},
    {VK_FORMAT_R16G16B16_SINT,   {CL_RGB, CL_SIGNED_INT16}},
    {VK_FORMAT_R16G16B16_SFLOAT, {CL_RGB, CL_HALF_FLOAT}},
    {VK_FORMAT_R32G32B32_UINT,   {CL_RGB, CL_UNSIGNED_INT32}},
    {VK_FORMAT_R32G32B32_SINT,   {CL_RGB, CL_SIGNED_INT32}},
    {VK_FORMAT_R32G32B32_SFLOAT, {CL_RGB, CL_FLOAT}},
    {VK_FORMAT_R5G6B5_UNORM_PACK16, {CL_RGB, CL_UNORM_SHORT_565}},

    // RGBA formats
    {VK_FORMAT_R8G8B8A8_UNORM,      {CL_RGBA, CL_UNORM_INT8}},
    {VK_FORMAT_R8G8B8A8_SNORM,      {CL_RGBA, CL_SNORM_INT8}},
    {VK_FORMAT_R8G8B8A8_UINT,       {CL_RGBA, CL_UNSIGNED_INT8}},
    {VK_FORMAT_R8G8B8A8_SINT,       {CL_RGBA, CL_SIGNED_INT8}},
    {VK_FORMAT_R16G16B16A16_UNORM,  {CL_RGBA, CL_UNORM_INT16}},
    {VK_FORMAT_R16G16B16A16_SNORM,  {CL_RGBA, CL_SNORM_INT16}},
    {VK_FORMAT_R16G16B16A16_UINT,   {CL_RGBA, CL_UNSIGNED_INT16}},
    {VK_FORMAT_R16G16B16A16_SINT,   {CL_RGBA, CL_SIGNED_INT16}},
    {VK_FORMAT_R16G16B16A16_SFLOAT, {CL_RGBA, CL_HALF_FLOAT}},
    {VK_FORMAT_R32G32B32A32_UINT,   {CL_RGBA, CL_UNSIGNED_INT32}},
    {VK_FORMAT_R32G32B32A32_SINT,   {CL_RGBA, CL_SIGNED_INT32}},
    {VK_FORMAT_R32G32B32A32_SFLOAT, {CL_RGBA, CL_FLOAT}},

    // BGRA formats
    {VK_FORMAT_B8G8R8A8_UNORM, {CL_BGRA, CL_UNORM_INT8}},
    {VK_FORMAT_B8G8R8A8_SNORM, {CL_BGRA, CL_SNORM_INT8}},
    {VK_FORMAT_B8G8R8A8_UINT,  {CL_BGRA, CL_UNSIGNED_INT8}},
    {VK_FORMAT_B8G8R8A8_SINT,  {CL_BGRA, CL_SIGNED_INT8}},
};

static bool vulkan_format_to_cl_image_format(VkFormat format, cl_image_format *clfmt) {
    auto m = gFormatMaps.find(format);
    bool success = false;

    if (m != gFormatMaps.end()) {
        *clfmt = (*m).second;
        success = true;
    }

    return success;
}

bool cl_image_format_to_vulkan_format(const cl_image_format &clformat, VkFormat &format) {
    for (auto const &vkcl : gFormatMaps) {
        auto const &clfmt = vkcl.second;
        if ((clfmt.image_channel_order == clformat.image_channel_order) &&
            (clfmt.image_channel_data_type == clformat.image_channel_data_type)) {
            format = vkcl.first;
            return true;
        }
    }

    return false;
}

cl_int clGetSupportedImageFormats(
    cl_context         context,
    cl_mem_flags       flags,
    cl_mem_object_type image_type,
    cl_uint            num_entries,
    cl_image_format   *image_formats,
    cl_uint           *num_image_formats
){
    LOG_API_CALL("context = %p, flags = %lu, image_type = %d, num_entries = %u, "
                 "image_formats = %p, num_image_formats = %p",
                 context, flags, image_type, num_entries, image_formats, num_image_formats);

    if (!is_valid_context(context)) {
        return CL_INVALID_CONTEXT;
    }

    if ((num_entries == 0) && (image_formats != nullptr)) {
        return CL_INVALID_VALUE;
    }

    if (!cvk_mem::is_image_type(image_type)) {
        return CL_INVALID_VALUE;
    }

    cl_uint num_formats_found = 0;

    auto pdev = context->device()->vulkan_physical_device();

    // Iterate over all vulkan formats
    for (int fmt = VK_FORMAT_BEGIN_RANGE; fmt < VK_FORMAT_END_RANGE; fmt++) {
        //cvk_debug_fn("fmt = %u", fmt);

        VkFormat format = static_cast<VkFormat>(fmt);
        VkFormatProperties properties;

        vkGetPhysicalDeviceFormatProperties(pdev, format, &properties);

        // FIXME do that properly
        if (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) {

            cl_image_format clfmt;

            bool success = vulkan_format_to_cl_image_format(format, &clfmt);

            if (success) {
                if ((image_formats != nullptr) && (num_formats_found < num_entries)) {
                    image_formats[num_formats_found] = clfmt;
                    //cvk_debug_fn("reporting image format {%d, %d}",
                    //          clfmt.image_channel_order, clfmt.image_channel_data_type);
                }
                num_formats_found++;
            }
        }
    }

    cvk_debug_fn("reporting %u formats", num_formats_found);

    if (num_image_formats != nullptr) {
        *num_image_formats = num_formats_found;
    }

    return CL_SUCCESS;
}

cl_int clEnqueueReadImage(
    cl_command_queue command_queue,
    cl_mem           image,
    cl_bool          blocking_read,
    const size_t    *origin,
    const size_t    *region,
    size_t           row_pitch,
    size_t           slice_pitch,
    void            *ptr,
    cl_uint          num_events_in_wait_list,
    const cl_event  *event_wait_list,
    cl_event        *event
){
    LOG_API_CALL("command_queue = %p, image = %p, blocking_read = %d, "
                 "origin = {%zu,%zu,%zu}, region = {%zu, %zu, %zu}, "
                 "row_pitch = %zu, slice_pitch = %zu, ptr = %p, "
                 "num_events_in_wait_list = %u, event_wait_list = %p, "
                 "event = %p",
                 command_queue, image, blocking_read,
                 origin[0], origin[1], origin[2],
                 region[0], region[1], region[2],
                 row_pitch, slice_pitch, ptr, num_events_in_wait_list,
                 event_wait_list, event);

    return CL_INVALID_OPERATION;
}

cl_int clEnqueueWriteImage(
    cl_command_queue command_queue,
    cl_mem           image,
    cl_bool          blocking_write,
    const size_t    *origin,
    const size_t    *region,
    size_t           input_row_pitch,
    size_t           input_slice_pitch,
    const void      *ptr,
    cl_uint          num_events_in_wait_list,
    const cl_event  *event_wait_list,
    cl_event        *event
){
    LOG_API_CALL("command_queue = %p, image = %p, blocking_write = %d, "
                 "origin = {%zu,%zu,%zu}, region = {%zu, %zu, %zu}, "
                 "input_row_pitch = %zu, input_slice_pitch = %zu, ptr = %p, "
                 "num_events_in_wait_list = %u, event_wait_list = %p, "
                 "event = %p",
                 command_queue, image, blocking_write,
                 origin[0], origin[1], origin[2],
                 region[0], region[1], region[2],
                 input_row_pitch, input_slice_pitch, ptr, num_events_in_wait_list,
                 event_wait_list, event);

    return CL_INVALID_OPERATION;
}

cl_int clEnqueueCopyImage(
    cl_command_queue command_queue,
    cl_mem           src_image,
    cl_mem           dst_image,
    const size_t    *src_origin,
    const size_t    *dst_origin,
    const size_t    *region,
    cl_uint          num_events_in_wait_list,
    const cl_event  *event_wait_list,
    cl_event        *event
){
    LOG_API_CALL("command_queue = %p, src_image = %p, dst_image = %p, "
                 "src_origin = {%zu,%zu,%zu}, dst_origin = {%zu, %zu, %zu}, "
                 "region = {%zu, %zu, %zu}, "
                 "num_events_in_wait_list = %u, event_wait_list = %p, "
                 "event = %p",
                 command_queue, src_image, dst_image,
                 src_origin[0], src_origin[1], src_origin[2],
                 dst_origin[0], dst_origin[1], dst_origin[2],
                 region[0], region[1], region[2],
                 num_events_in_wait_list, event_wait_list, event);

    return CL_INVALID_OPERATION;
}

cl_int clEnqueueFillImage(
    cl_command_queue command_queue,
    cl_mem           image,
    const void      *fill_color,
    const size_t    *origin,
    const size_t    *region,
    cl_uint          num_events_in_wait_list,
    const cl_event  *event_wait_list,
    cl_event        *event
){
    LOG_API_CALL("command_queue = %p, image = %p, fill_color = %p, "
                 "origin = {%zu,%zu,%zu}, region = {%zu, %zu, %zu}, "
                 "num_events_in_wait_list = %u, event_wait_list = %p, "
                 "event = %p",
                 command_queue, image, fill_color,
                 origin[0], origin[1], origin[2],
                 region[0], region[1], region[2],
                 num_events_in_wait_list, event_wait_list, event);

    return CL_INVALID_OPERATION;
}

cl_int clEnqueueCopyImageToBuffer(
    cl_command_queue command_queue,
    cl_mem           src_image,
    cl_mem           dst_buffer,
    const size_t    *src_origin,
    const size_t    *region,
    size_t           dst_offset,
    cl_uint          num_events_in_wait_list,
    const cl_event  *event_wait_list,
    cl_event        *event
){
    LOG_API_CALL("command_queue = %p, src_image = %p, dst_buffer = %p, "
                 "src_origin = {%zu,%zu,%zu}, region = {%zu, %zu, %zu}, "
                 "dst_offset = %zu, num_events_in_wait_list = %u, "
                 "event_wait_list = %p, event = %p",
                 command_queue, src_image, dst_buffer,
                 src_origin[0], src_origin[1], src_origin[2],
                 region[0], region[1], region[2],
                 dst_offset,
                 num_events_in_wait_list, event_wait_list, event);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    if (!is_valid_image(src_image) || !is_valid_buffer(dst_buffer)) {
        return CL_INVALID_MEM_OBJECT;
    }

    if (!event_wait_list_is_valid(num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    if (!is_same_context(command_queue, src_image) ||
        !is_same_context(command_queue, dst_buffer) ||
        !is_same_context(command_queue, num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_CONTEXT;
    }
    // TODO CL_INVALID_MEM_OBJECT if src_image is a 1D image buffer object created from dst_buffer.
    // TODO CL_INVALID_VALUE if the 1D, 2D, or 3D rectangular region specified by src_origin and src_origin + region refers to a region outside src_image, or if the region specified by dst_offset and dst_offset + dst_cb refers to a region outside dst_buffer.
    // TODO CL_INVALID_VALUE if values in src_origin and region do not follow rules described in the argument description for src_origin and region.

    // TODO CL_MISALIGNED_SUB_BUFFER_OFFSET if dst_buffer is a sub-buffer object and offset specified when the sub-buffer object is created is not aligned to CL_DEVICE_MEM_BASE_ADDR_ALIGN value for device associated with queue.
    // TODO CL_INVALID_IMAGE_SIZE if image dimensions (image width, height, specified or compute row and/or slice pitch) for src_image are not supported by device associated with queue.
    // TODO CL_INVALID_IMAGE_FORMAT if image format (image channel order and data type) for src_image are not supported by device associated with queue.
    // TODO CL_MEM_OBJECT_ALLOCATION_FAILURE if there is a failure to allocate memory for data store associated with src_image or dst_buffer.

    if (!command_queue->device()->supports_images()) {
        return CL_INVALID_OPERATION;
    }

    //
    auto image = static_cast<cvk_image*>(src_image);
    auto buffer = static_cast<cvk_buffer*>(dst_buffer);
    std::array<size_t, 3> origin = { src_origin[0], src_origin[1], src_origin[2] };
    std::array<size_t, 3> reg = { region[0], region[1], region[2] };

    auto cmd = std::make_unique<cvk_command_buffer_image_copy>(
                    CL_COMMAND_COPY_IMAGE_TO_BUFFER, command_queue, buffer,
                    image, dst_offset, origin, reg);
    auto err = cmd->build();
    if (err != CL_SUCCESS) {
        return err;
    }

    command_queue->enqueue_command_with_deps(cmd.release(),
                                             num_events_in_wait_list,
                                             event_wait_list, event);

    return CL_SUCCESS;
}

cl_int clEnqueueCopyBufferToImage(
    cl_command_queue command_queue,
    cl_mem           src_buffer,
    cl_mem           dst_image,
    size_t           src_offset,
    const size_t    *dst_origin,
    const size_t    *region,
    cl_uint          num_events_in_wait_list,
    const cl_event  *event_wait_list,
    cl_event        *event
){
    LOG_API_CALL("command_queue = %p, src_buffer = %p, dst_image = %p, "
                 "src_offset = %zu, "
                 "dst_origin = {%zu,%zu,%zu}, region = {%zu, %zu, %zu}, "
                 "num_events_in_wait_list = %u, event_wait_list = %p, "
                 "event = %p",
                 command_queue, src_buffer, dst_image, src_offset,
                 dst_origin[0], dst_origin[1], dst_origin[2],
                 region[0], region[1], region[2],
                 num_events_in_wait_list, event_wait_list, event);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    if (!is_valid_image(dst_image) || !is_valid_buffer(src_buffer)) {
        return CL_INVALID_MEM_OBJECT;
    }

    if (!event_wait_list_is_valid(num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    if (!is_same_context(command_queue, src_buffer) ||
        !is_same_context(command_queue, dst_image) ||
        !is_same_context(command_queue, num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_CONTEXT;
    }
    // TODO CL_INVALID_MEM_OBJECT if dst_image is a 1D image buffer object created from src_buffer.
    // TODO CL_INVALID_VALUE if the 1D, 2D, or 3D rectangular region specified by dst_origin and dst_origin + region refers to a region outside dst_origin, or if the region specified by src_offset and src_offset + src_cb refers to a region outside src_buffer.
    // TODO CL_INVALID_VALUE if values in dst_origin and region do not follow rules described in the argument description for dst_origin and region.
    // TODO CL_MISALIGNED_SUB_BUFFER_OFFSET if src_buffer is a sub-buffer object and offset specified when the sub-buffer object is created is not aligned to CL_DEVICE_MEM_BASE_ADDR_ALIGN value for device associated with queue.
    // TODO CL_INVALID_IMAGE_SIZE if image dimensions (image width, height, specified or compute row and/or slice pitch) for dst_image are not supported by device associated with queue.
    // TODO CL_INVALID_IMAGE_FORMAT if image format (image channel order and data type) for dst_image are not supported by device associated with queue.
    // TODO CL_MEM_OBJECT_ALLOCATION_FAILURE if there is a failure to allocate memory for data store associated with src_buffer or dst_image.
    if (!command_queue->device()->supports_images()) {
        return CL_INVALID_OPERATION;
    }

    auto image = static_cast<cvk_image*>(dst_image);
    auto buffer = static_cast<cvk_buffer*>(src_buffer);
    std::array<size_t, 3> origin = { dst_origin[0], dst_origin[1], dst_origin[2] };
    std::array<size_t, 3> reg = { region[0], region[1], region[2] };

    auto cmd = std::make_unique<cvk_command_buffer_image_copy>(
                    CL_COMMAND_COPY_BUFFER_TO_IMAGE, command_queue, buffer,
                    image, src_offset, origin, reg);
    auto err = cmd->build();
    if (err != CL_SUCCESS) {
        return err;
    }

    command_queue->enqueue_command_with_deps(cmd.release(),
                                             num_events_in_wait_list,
                                             event_wait_list, event);

    return CL_SUCCESS;
}

void* clEnqueueMapImage(
    cl_command_queue command_queue,
    cl_mem           image,
    cl_bool          blocking_map,
    cl_map_flags     map_flags,
    const size_t    *origin,
    const size_t    *region,
    size_t          *image_row_pitch,
    size_t          *image_slice_pitch,
    cl_uint          num_events_in_wait_list,
    const cl_event  *event_wait_list ,
    cl_event        *event,
    cl_int          *errcode_ret
){
    LOG_API_CALL("command_queue = %p, image = %p, blocking_map = %d, "
                 "map_flags = %lx, "
                 "origin = {%zu,%zu,%zu}, region = {%zu, %zu, %zu}, "
                 "image_row_pitch = %p, image_slice_pitch = %p, "
                 "num_events_in_wait_list = %u, event_wait_list = %p, "
                 "event = %p, errcode_ret = %p",
                 command_queue, image, blocking_map, map_flags,
                 origin[0], origin[1], origin[2],
                 region[0], region[1], region[2],
                 image_row_pitch, image_slice_pitch, num_events_in_wait_list,
                 event_wait_list, event, errcode_ret);

    if (errcode_ret != nullptr) {
        *errcode_ret = CL_INVALID_OPERATION;
    }

    return nullptr;
}

cl_program cvk_create_program_with_il_khr(
    cl_context context,
    const void *il,
    size_t length,
    cl_int *errcode_ret
){
    if (!is_valid_context(context)) {
        *errcode_ret = CL_INVALID_CONTEXT;
        return nullptr;
    }

    if ((il == nullptr) || (length == 0)) {
        *errcode_ret = CL_INVALID_VALUE;
        return nullptr;
    }

    // TODO CL_INVALID_VALUE if the length-byte block of memory pointed to by il does not contain well-formed intermediate language.

    auto program = new cvk_program(context, il, length);

    *errcode_ret = CL_SUCCESS;
    return program;
}

cl_program clCreateProgramWithILKHR(
    cl_context context,
    const void *il,
    size_t length,
    cl_int *errcode_ret
){
    LOG_API_CALL("context = %p, il = %p, length = %zu, errcode_ret = %p",
                 context, il, length, errcode_ret);

    cl_int errcode;
    auto program = cvk_create_program_with_il_khr(context, il, length, &errcode);

    if (errcode_ret != nullptr) {
        *errcode_ret = errcode;
    }

    return program;
}
