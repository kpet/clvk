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
#include "icd.hpp"
#include "image_format.hpp"
#include "init.hpp"
#include "kernel.hpp"
#include "log.hpp"
#include "memory.hpp"
#include "objects.hpp"
#include "program.hpp"
#include "queue.hpp"
#include "semaphore.hpp"
#include "tracing.hpp"

#define LOG_API_CALL(fmt, ...)                                                 \
    cvk_debug_group_fn(loggroup::api, fmt, __VA_ARGS__)

#define CLVK_API_CALL CL_API_CALL

namespace {

// Validation functions
bool is_valid_platform(cl_platform_id platform) {
    return platform != nullptr && icd_downcast(platform)->is_valid();
}

bool is_valid_device(cl_device_id device) {
    return device != nullptr && icd_downcast(device)->is_valid();
}

bool is_valid_context(cl_context context) {
    return context != nullptr && icd_downcast(context)->is_valid();
}

bool is_valid_program(cl_program program) {
    return program != nullptr && icd_downcast(program)->is_valid();
}

bool is_valid_kernel(cl_kernel kernel) {
    return kernel != nullptr && icd_downcast(kernel)->is_valid();
}

bool is_valid_sampler(cl_sampler sampler) {
    return sampler != nullptr && icd_downcast(sampler)->is_valid();
}

bool is_valid_mem_object(cl_mem mem) {
    return mem != nullptr && icd_downcast(mem)->is_valid();
}

bool is_valid_buffer(cl_mem mem) {
    return is_valid_mem_object(mem) && icd_downcast(mem)->is_buffer_type();
}

bool is_valid_image(cl_mem mem) {
    return is_valid_mem_object(mem) && icd_downcast(mem)->is_image_type();
}

bool is_valid_command_queue(cl_command_queue queue) {
    return queue != nullptr && icd_downcast(queue)->is_valid();
}

bool is_valid_event(cl_event event) {
    return event != nullptr && icd_downcast(event)->is_valid();
}

bool is_valid_semaphore(cl_semaphore_khr sem) {
    return sem != nullptr && icd_downcast(sem)->is_valid();
}

bool is_valid_event_wait_list(cl_uint num_events_in_wait_list,
                              const cl_event* event_wait_list) {

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

bool is_same_context(cl_command_queue queue, cl_mem mem) {
    return icd_downcast(queue)->context() == icd_downcast(mem)->context();
}

bool is_same_context(cl_command_queue queue, cl_kernel kernel) {
    return icd_downcast(queue)->context() == icd_downcast(kernel)->context();
}

bool is_same_context(cl_command_queue queue, cl_uint num_events,
                     const cl_event* event_list) {
    for (cl_uint i = 0; i < num_events; i++) {
        if (icd_downcast(queue)->context() !=
            icd_downcast(event_list[i])->context()) {
            return false;
        }
    }

    return true;
}

bool is_same_context(cl_command_queue queue, cl_uint num_semas,
                     const cl_semaphore_khr* semas) {
    for (cl_uint i = 0; i < num_semas; i++) {
        if (icd_downcast(queue)->context() !=
            icd_downcast(semas[i])->context()) {
            return false;
        }
    }

    return true;
}

bool is_valid_device_type(cl_device_type type) {
    return (type < (CL_DEVICE_TYPE_CUSTOM << 1)) ||
           (type == CL_DEVICE_TYPE_ALL);
}

bool map_flags_are_valid(cl_map_flags flags) {
    if ((flags & CL_MAP_WRITE_INVALIDATE_REGION) &&
        (flags & (CL_MAP_READ | CL_MAP_WRITE))) {
        return false;
    }
    return true;
}

bool is_compiler_available(cl_uint num_devices, const cl_device_id* devices) {
    for (cl_uint i = 0; i < num_devices; i++) {
        auto dev = icd_downcast(devices[i]);
        if (!dev->compiler_available()) {
            return false;
        }
    }

    return true;
}

// Utilities
struct api_query_string : public std::string {
    api_query_string() : std::string() {}
    api_query_string(const char* init) : std::string(init) {}
    api_query_string(const std::string& other) : std::string(other) {}

    size_t size_with_null() const { return size() + 1; }
};

} // namespace

// Platform API

cl_int cvk_get_platform_ids(const clvk_global_state* state, cl_uint num_entries,
                            cl_platform_id* platforms, cl_uint* num_platforms) {
    if ((num_platforms == nullptr) && (platforms == nullptr)) {
        return CL_INVALID_VALUE;
    }

    if ((num_entries == 0) && (platforms != nullptr)) {
        return CL_INVALID_VALUE;
    }

    if (platforms != nullptr) {
        platforms[0] = state->platform();
    }

    if (num_platforms != nullptr) {
        *num_platforms = 1;
    }

    return CL_SUCCESS;
}

cl_int CLVK_API_CALL clGetPlatformIDs(cl_uint num_entries,
                                      cl_platform_id* platforms,
                                      cl_uint* num_platforms) {
    auto state = get_or_init_global_state();

    TRACE_FUNCTION("num_entries", num_entries);
    LOG_API_CALL("num_entries = %u, platforms = %p, num_platforms = %p",
                 num_entries, platforms, num_platforms);

    return cvk_get_platform_ids(state, num_entries, platforms, num_platforms);
}

cl_int CLVK_API_CALL clGetPlatformInfo(cl_platform_id platform,
                                       cl_platform_info param_name,
                                       size_t param_value_size,
                                       void* param_value,
                                       size_t* param_value_size_ret) {
    auto state = get_or_init_global_state();

    TRACE_FUNCTION("platform", (uintptr_t)platform, "param_name", param_name);
    LOG_API_CALL("platform = %p, param_name = %u, param_value_size = %zu, "
                 "param_value = %p, param_value_size_ret = %p",
                 platform, param_name, param_value_size, param_value,
                 param_value_size_ret);
    cl_int ret = CL_SUCCESS;

    size_t size_ret = 0;
    const void* copy_ptr = nullptr;
    cl_version val_version;
    api_query_string val_string;
    cl_ulong val_ulong;

    if (!is_valid_platform(platform)) {
        return CL_INVALID_PLATFORM;
    }

    const cvk_platform* plat = state->platform();
    if (platform != nullptr) {
        plat = icd_downcast(platform);
    }

    switch (param_name) {
    case CL_PLATFORM_NAME:
        val_string = plat->name();
        copy_ptr = val_string.c_str();
        size_ret = val_string.size_with_null();
        break;
    case CL_PLATFORM_VERSION:
        val_string = plat->version_string();
        copy_ptr = val_string.c_str();
        size_ret = val_string.size_with_null();
        break;
    case CL_PLATFORM_VENDOR:
        val_string = plat->vendor();
        copy_ptr = val_string.c_str();
        size_ret = val_string.size_with_null();
        break;
    case CL_PLATFORM_PROFILE:
        val_string = plat->profile();
        copy_ptr = val_string.c_str();
        size_ret = val_string.size_with_null();
        break;
    case CL_PLATFORM_EXTENSIONS:
        val_string = plat->extension_string();
        copy_ptr = val_string.c_str();
        size_ret = val_string.size_with_null();
        break;
    case CL_PLATFORM_ICD_SUFFIX_KHR:
        val_string = plat->icd_suffix();
        copy_ptr = val_string.c_str();
        size_ret = val_string.size_with_null();
        break;
    case CL_PLATFORM_NUMERIC_VERSION:
        val_version = plat->version();
        copy_ptr = &val_version;
        size_ret = sizeof(val_version);
        break;
    case CL_PLATFORM_EXTENSIONS_WITH_VERSION:
        copy_ptr = plat->extensions().data();
        size_ret = plat->extensions().size() * sizeof(cl_name_version);
        break;
    case CL_PLATFORM_HOST_TIMER_RESOLUTION:
        val_ulong = plat->host_timer_resolution();
        copy_ptr = &val_ulong;
        size_ret = sizeof(val_ulong);
        break;
    default:
        ret = CL_INVALID_VALUE;
        break;
    }

    if ((param_value != nullptr) && (param_value_size < size_ret)) {
        ret = CL_INVALID_VALUE;
    }

    if ((param_value != nullptr) && (copy_ptr != nullptr)) {
        memcpy(param_value, copy_ptr, std::min(param_value_size, size_ret));
    }

    if (param_value_size_ret != nullptr) {
        *param_value_size_ret = size_ret;
    }

    return ret;
}

static const std::unordered_map<std::string, void*> gExtensionEntrypoints = {
#define FUNC_PTR(X) reinterpret_cast<void*>(X)
#define EXTENSION_ENTRYPOINT(X)                                                \
    { #X, FUNC_PTR(X) }
    EXTENSION_ENTRYPOINT(clCreateProgramWithILKHR),
    EXTENSION_ENTRYPOINT(clIcdGetPlatformIDsKHR),
    EXTENSION_ENTRYPOINT(clCreateCommandQueueWithPropertiesKHR),
    EXTENSION_ENTRYPOINT(clGetKernelSuggestedLocalWorkSizeKHR),
    {"clGetKernelSubGroupInfoKHR", FUNC_PTR(clGetKernelSubGroupInfo)},
    EXTENSION_ENTRYPOINT(clCreateSemaphoreWithPropertiesKHR),
    EXTENSION_ENTRYPOINT(clEnqueueWaitSemaphoresKHR),
    EXTENSION_ENTRYPOINT(clEnqueueSignalSemaphoresKHR),
    EXTENSION_ENTRYPOINT(clGetSemaphoreInfoKHR),
    EXTENSION_ENTRYPOINT(clRetainSemaphoreKHR),
    EXTENSION_ENTRYPOINT(clReleaseSemaphoreKHR),
#undef EXTENSION_ENTRYPOINT
#undef FUNC_PTR
};

void* cvk_get_extension_function_pointer(const char* funcname) {
    if (gExtensionEntrypoints.find(funcname) != gExtensionEntrypoints.end()) {
        return gExtensionEntrypoints.at(funcname);
    } else {
        return nullptr;
    }
}

void* CLVK_API_CALL clGetExtensionFunctionAddressForPlatform(
    cl_platform_id platform, const char* funcname) {
    TRACE_FUNCTION("platform", (uintptr_t)platform);
    LOG_API_CALL("platform = %p, funcname = '%s'", platform, funcname);

    if (platform == nullptr) {
        return nullptr;
    }

    return cvk_get_extension_function_pointer(funcname);
}

void* CLVK_API_CALL clGetExtensionFunctionAddress(const char* funcname) {
    TRACE_FUNCTION();
    LOG_API_CALL("funcname = '%s'", funcname);

    return cvk_get_extension_function_pointer(funcname);
}

// Device APIs
cl_int CLVK_API_CALL clGetDeviceIDs(cl_platform_id platform,
                                    cl_device_type device_type,
                                    cl_uint num_entries, cl_device_id* devices,
                                    cl_uint* num_devices) {

    auto state = get_or_init_global_state();

    TRACE_FUNCTION("platform", (uintptr_t)platform, "device_type",
                   TRACE_STRING(cl_device_type_to_string(device_type)),
                   "num_entries", num_entries);
    LOG_API_CALL(
        "platform = %p, device_type = %lu (%s), num_entries = %u, devices "
        "= %p, num_devices = %p",
        platform, device_type, cl_device_type_to_string(device_type),
        num_entries, devices, num_devices);

    if (platform == nullptr) {
        platform = state->platform();
    } else if (platform != state->platform()) {
        return CL_INVALID_PLATFORM;
    }

    if ((num_entries == 0) && (devices != nullptr)) {
        return CL_INVALID_VALUE;
    }

    if ((num_devices == nullptr) && (devices == nullptr)) {
        return CL_INVALID_VALUE;
    }

    if (!is_valid_device_type(device_type)) {
        return CL_INVALID_DEVICE_TYPE;
    }

    cl_uint num = 0;

    for (auto dev : icd_downcast(platform)->devices()) {
        if (dev->type() & device_type) {
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

cl_int CLVK_API_CALL clGetDeviceInfo(cl_device_id dev,
                                     cl_device_info param_name,
                                     size_t param_value_size, void* param_value,
                                     size_t* param_value_size_ret) {
    TRACE_FUNCTION("device", (uintptr_t)dev, "param_name", param_name);
    LOG_API_CALL(
        "device = %p, param_name = %d, size = %zu, value = %p, size_ret = %p",
        dev, param_name, param_value_size, param_value, param_value_size_ret);
    cl_int ret = CL_SUCCESS;

    size_t size_ret = 0;
    const void* copy_ptr = nullptr;
    size_t val_sizet;
    cl_uint val_uint;
    api_query_string val_string;
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
    cl_version val_version;
    cl_device_svm_capabilities val_svmcaps;
    cl_device_device_enqueue_capabilities val_dev_enqueue_caps;
    cl_device_pci_bus_info_khr val_pci_bus_info;
    cl_device_atomic_capabilities val_atomic_capabilities;
    cl_device_integer_dot_product_capabilities_khr val_int_dot_product;
    cl_device_integer_dot_product_acceleration_properties_khr
        val_int_dot_product_props;
    std::vector<size_t> val_subgroup_sizes;

    auto device = icd_downcast(dev);

    if (!is_valid_device(device)) {
        return CL_INVALID_DEVICE;
    }

    switch (param_name) {
    case CL_DEVICE_PLATFORM:
        val_platform = device->platform();
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
        val_string = device->vendor();
        copy_ptr = val_string.c_str();
        size_ret = val_string.size_with_null();
        break;
    case CL_DEVICE_VENDOR_ID:
        val_uint = device->vendor_id();
        copy_ptr = &val_uint;
        size_ret = sizeof(val_uint);
        break;
    case CL_DRIVER_VERSION:
        val_string = device->driver_version();
        copy_ptr = val_string.c_str();
        size_ret = val_string.size_with_null();
        break;
    case CL_DEVICE_VERSION:
        val_string = device->version_string();
        copy_ptr = val_string.c_str();
        size_ret = val_string.size_with_null();
        break;
    case CL_DEVICE_OPENCL_C_VERSION:
        val_string = device->c_version_string();
        copy_ptr = val_string.c_str();
        size_ret = val_string.size_with_null();
        break;
    case CL_DEVICE_PROFILE:
        val_string = device->profile();
        copy_ptr = val_string.c_str();
        size_ret = val_string.size_with_null();
        break;
    case CL_DEVICE_BUILT_IN_KERNELS:
        val_string = "";
        copy_ptr = val_string.c_str();
        size_ret = val_string.size_with_null();
        break;
    case CL_DEVICE_EXTENSIONS:
        val_string = device->extension_string();
        copy_ptr = val_string.c_str();
        size_ret = val_string.size_with_null();
        break;
    case CL_DEVICE_AVAILABLE:
        val_bool = CL_TRUE;
        copy_ptr = &val_bool;
        size_ret = sizeof(val_bool);
        break;
    case CL_DEVICE_COMPILER_AVAILABLE:
    case CL_DEVICE_LINKER_AVAILABLE:
        val_bool = device->compiler_available();
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
    case CL_DEVICE_HALF_FP_CONFIG:
    case CL_DEVICE_SINGLE_FP_CONFIG:
    case CL_DEVICE_DOUBLE_FP_CONFIG:
        val_fpconfig = device->fp_config(param_name);
        copy_ptr = &val_fpconfig;
        size_ret = sizeof(val_fpconfig);
        break;
    case CL_DEVICE_ADDRESS_BITS:
        val_uint = device->address_bits();
        copy_ptr = &val_uint;
        size_ret = sizeof(val_uint);
        break;
    case CL_DEVICE_MEM_BASE_ADDR_ALIGN:
        val_uint = device->mem_base_addr_align();
        copy_ptr = &val_uint;
        size_ret = sizeof(val_uint);
        break;
    case CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE:
        val_uint = 128; // Alignment in bytes of long16
        copy_ptr = &val_uint, size_ret = sizeof(val_uint);
        break;
    case CL_DEVICE_GLOBAL_MEM_CACHE_TYPE:
        val_cache_type = CL_NONE; // FIXME
        copy_ptr = &val_cache_type;
        size_ret = sizeof(val_cache_type);
        break;
    case CL_DEVICE_GLOBAL_MEM_CACHE_SIZE:
        val_ulong = device->global_mem_cache_size();
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
        val_sizet = device->max_work_group_size();
        copy_ptr = &val_sizet;
        size_ret = sizeof(val_sizet);
        break;
    case CL_DEVICE_MAX_COMPUTE_UNITS:
        val_uint = device->num_compute_units();
        copy_ptr = &val_uint;
        size_ret = sizeof(val_uint);
        break;
    case CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:
        val_uint = device->max_work_item_dimensions();
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
        val_sizet = 1024; // FIXME this is the minimum, revisit when looking
                          // into push constants
        copy_ptr = &val_sizet;
        size_ret = sizeof(val_sizet);
        break;
    case CL_DEVICE_MAX_CONSTANT_ARGS:
        val_uint = 8; // TODO be smarter
        copy_ptr = &val_uint;
        size_ret = sizeof(val_uint);
        break;
    case CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: // TODO be smarter
        val_ulong = 64 * 1024;
        copy_ptr = &val_ulong;
        size_ret = sizeof(val_ulong);
        break;
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR:
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT:
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT:
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG:
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT:
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF:
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR:
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT:
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_INT:
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG:
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT:
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF:
        val_uint = 1; // FIXME can we do better?
        copy_ptr = &val_uint;
        size_ret = sizeof(val_uint);
        break;
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE:
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE:
        val_uint = 0;
        copy_ptr = &val_uint;
        size_ret = sizeof(val_uint);
        break;
    case CL_DEVICE_PROFILING_TIMER_RESOLUTION:
        val_sizet = 1;
        copy_ptr = &val_sizet;
        size_ret = sizeof(val_sizet);
        break;
    case CL_DEVICE_GLOBAL_MEM_SIZE:
        val_ulong = device->global_mem_size();
        copy_ptr = &val_ulong;
        size_ret = sizeof(val_ulong);
        break;
    case CL_DEVICE_MAX_MEM_ALLOC_SIZE:
        val_ulong = device->max_mem_alloc_size();
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
        val_uint = device->max_samplers();
        copy_ptr = &val_uint;
        size_ret = sizeof(val_uint);
        break;
    case CL_DEVICE_IMAGE_MAX_BUFFER_SIZE:
        val_sizet = device->image_max_buffer_size();
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
    case CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS:
        if (!device->supports_read_write_images()) {
            val_uint = 0;
            copy_ptr = &val_uint;
            size_ret = sizeof(val_uint);
            break;
        }
        [[fallthrough]];
    case CL_DEVICE_MAX_WRITE_IMAGE_ARGS:
        val_uint = device->vulkan_limits().maxPerStageDescriptorStorageImages;
        copy_ptr = &val_uint;
        size_ret = sizeof(val_uint);
        break;
    case CL_DEVICE_IL_VERSION:
        val_string = device->ils_string();
        copy_ptr = val_string.c_str();
        size_ret = val_string.size_with_null();
        break;
    case CL_DEVICE_NUMERIC_VERSION:
        val_version = device->version();
        copy_ptr = &val_version;
        size_ret = sizeof(val_version);
        break;
    case CL_DEVICE_OPENCL_C_NUMERIC_VERSION_KHR:
        val_version = device->c_version();
        copy_ptr = &val_version;
        size_ret = sizeof(val_version);
        break;
    case CL_DEVICE_EXTENSIONS_WITH_VERSION:
        copy_ptr = device->extensions().data();
        size_ret = device->extensions().size() * sizeof(cl_name_version);
        break;
    case CL_DEVICE_ILS_WITH_VERSION:
        copy_ptr = device->ils().data();
        size_ret = device->ils().size() * sizeof(cl_name_version);
        break;
    case CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION:
        copy_ptr = nullptr;
        size_ret = 0;
        break;
    case CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT:
        val_bool = device->supports_non_uniform_workgroup();
        copy_ptr = &val_bool;
        size_ret = sizeof(val_bool);
        break;
    case CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE:
    case CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE:
        val_sizet = 0;
        copy_ptr = &val_sizet;
        size_ret = sizeof(val_sizet);
        break;
    case CL_DEVICE_IMAGE_PITCH_ALIGNMENT:
    case CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT:
        val_uint = 0;
        copy_ptr = &val_uint;
        size_ret = sizeof(val_uint);
        break;
    case CL_DEVICE_SVM_CAPABILITIES:
        val_svmcaps = 0;
        copy_ptr = &val_svmcaps;
        size_ret = sizeof(val_svmcaps);
        break;
    case CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS:
    case CL_DEVICE_PIPE_SUPPORT:
    case CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT:
        val_bool = CL_FALSE;
        copy_ptr = &val_bool;
        size_ret = sizeof(val_bool);
        break;
    case CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT:
        // TODO(#216) re-enable when clspv ready
        val_bool = CL_FALSE;
        copy_ptr = &val_bool;
        size_ret = sizeof(val_bool);
        break;
    case CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES:
        val_dev_enqueue_caps = 0;
        copy_ptr = &val_dev_enqueue_caps;
        size_ret = sizeof(val_dev_enqueue_caps);
        break;
    case CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES:
        val_queue_properties = 0;
        copy_ptr = &val_queue_properties;
        size_ret = sizeof(val_queue_properties);
        break;
    case CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE:
    case CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE:
    case CL_DEVICE_MAX_ON_DEVICE_QUEUES:
    case CL_DEVICE_MAX_ON_DEVICE_EVENTS:
    case CL_DEVICE_MAX_PIPE_ARGS:
    case CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS:
    case CL_DEVICE_PIPE_MAX_PACKET_SIZE:
        val_uint = 0;
        copy_ptr = &val_uint;
        size_ret = sizeof(val_uint);
        break;
    case CL_DEVICE_MAX_NUM_SUB_GROUPS:
        val_uint = device->max_num_sub_groups();
        copy_ptr = &val_uint;
        size_ret = sizeof(val_uint);
        break;
    case CL_DEVICE_OPENCL_C_ALL_VERSIONS:
        copy_ptr = device->opencl_c_versions().data();
        size_ret = device->opencl_c_versions().size() * sizeof(cl_name_version);
        break;
    case CL_DEVICE_OPENCL_C_FEATURES:
        copy_ptr = device->opencl_c_features().data();
        size_ret = device->opencl_c_features().size() * sizeof(cl_name_version);
        break;
    case CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE:
        val_sizet = device->preferred_work_group_size_multiple();
        copy_ptr = &val_sizet;
        size_ret = sizeof(val_sizet);
        break;
    case CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES:
        val_atomic_capabilities =
            CL_DEVICE_ATOMIC_ORDER_RELAXED | CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP;
        if (device->supports_atomic_order_acq_rel()) {
            val_atomic_capabilities |= CL_DEVICE_ATOMIC_ORDER_ACQ_REL;
        }
        if (device->supports_atomic_scope_device()) {
            val_atomic_capabilities |= CL_DEVICE_ATOMIC_SCOPE_DEVICE;
        }
        copy_ptr = &val_atomic_capabilities;
        size_ret = sizeof(val_atomic_capabilities);
        break;
    case CL_DEVICE_ATOMIC_FENCE_CAPABILITIES:
        val_atomic_capabilities = CL_DEVICE_ATOMIC_ORDER_RELAXED |
                                  CL_DEVICE_ATOMIC_SCOPE_WORK_ITEM |
                                  CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP;
        if (device->vulkan_memory_model_features().vulkanMemoryModel) {
            val_atomic_capabilities |= CL_DEVICE_ATOMIC_ORDER_ACQ_REL;
        }
        if (device->vulkan_memory_model_features()
                .vulkanMemoryModelDeviceScope) {
            val_atomic_capabilities |= CL_DEVICE_ATOMIC_SCOPE_DEVICE;
        }
        copy_ptr = &val_atomic_capabilities;
        size_ret = sizeof(val_atomic_capabilities);
        break;
    case CL_DEVICE_LATEST_CONFORMANCE_VERSION_PASSED:
        val_string = "v2023-12-12-00";
        copy_ptr = val_string.c_str();
        size_ret = val_string.size_with_null();
        break;
    case CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT:
    case CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT:
    case CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT:
        val_uint = 0; // Natural size of the types
        copy_ptr = &val_uint;
        size_ret = sizeof(val_uint);
        break;
    case CL_DEVICE_UUID_KHR:
        copy_ptr = device->uuid();
        size_ret = CL_UUID_SIZE_KHR;
        break;
    case CL_DRIVER_UUID_KHR:
        copy_ptr = device->driver_uuid();
        size_ret = CL_UUID_SIZE_KHR;
        break;
    case CL_DEVICE_LUID_VALID_KHR:
        val_bool = device->luid_valid();
        copy_ptr = &val_bool;
        size_ret = sizeof(cl_bool);
        break;
    case CL_DEVICE_LUID_KHR:
        copy_ptr = device->luid();
        size_ret = CL_LUID_SIZE_KHR;
        break;
    case CL_DEVICE_NODE_MASK_KHR:
        val_uint = device->node_mask();
        copy_ptr = &val_uint;
        size_ret = sizeof(val_uint);
        break;
    case CL_DEVICE_PCI_BUS_INFO_KHR:
        val_pci_bus_info = device->pci_bus_info();
        copy_ptr = &val_pci_bus_info;
        size_ret = sizeof(val_pci_bus_info);
        break;
    case CL_DEVICE_INTEGER_DOT_PRODUCT_CAPABILITIES_KHR:
        val_int_dot_product = device->dot_product_capabilities();
        copy_ptr = &val_int_dot_product;
        size_ret = sizeof(val_int_dot_product);
        break;
    case CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_8BIT_KHR:
        val_int_dot_product_props = device->dot_product_8bit_properties();
        copy_ptr = &val_int_dot_product_props;
        size_ret = sizeof(val_int_dot_product_props);
        break;
    case CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_4x8BIT_PACKED_KHR:
        val_int_dot_product_props =
            device->dot_product_4x8bit_packed_properties();
        copy_ptr = &val_int_dot_product_props;
        size_ret = sizeof(val_int_dot_product_props);
        break;
    case CL_DEVICE_SUB_GROUP_SIZES_INTEL:
        if (device->supports_subgroup_size_selection()) {
            uint32_t size = device->min_sub_group_size();
            while (size <= device->max_sub_group_size()) {
                val_subgroup_sizes.push_back((size_t)size);
                size *= 2;
            }
            copy_ptr = val_subgroup_sizes.data();
            size_ret = sizeof(size_t) * val_subgroup_sizes.size();
            break;
        }
        [[fallthrough]];
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

cl_int CLVK_API_CALL clCreateSubDevices(
    cl_device_id in_device, const cl_device_partition_property* properties,
    cl_uint num_devices, cl_device_id* out_devices, cl_uint* num_devices_ret) {
    TRACE_FUNCTION("in_device", (uintptr_t)in_device, "num_devices",
                   num_devices);
    LOG_API_CALL("in_device = %p, properties = %p, num_devices = %u, "
                 "out_devices = %p, num_devices_ret = %p",
                 in_device, properties, num_devices, out_devices,
                 num_devices_ret);

    // TODO CL_INVALID_DEVICE if in_device is not valid.
    // TODO CL_INVALID_VALUE if values specified in properties are not valid or
    // if values specified in properties are valid but not supported by the
    // device.
    // TODO CL_INVALID_VALUE if out_devices is not NULL and num_devices is less
    // than the number of sub-devices created by the partition scheme.
    // TODO CL_DEVICE_PARTITION_FAILED if the partition name is supported by the
    // implementation but in_device could not be further partitioned.
    // TODO CL_INVALID_DEVICE_PARTITION_COUNT if the partition name specified in
    // properties is CL_DEVICE_PARTITION_BY_COUNTS and the number of sub-devices
    // requested exceeds CL_DEVICE_PARTITION_MAX_SUB_DEVICES or the total number
    // of compute units requested exceeds CL_DEVICE_PARTITION_MAX_COMPUTE_UNITS
    // for in_device, or the number of compute units requested for one or more
    // sub-devices is less than zero or the number of sub-devices requested
    // exceeds CL_DEVICE_PARTITION_MAX_COMPUTE_UNITS for in_device.
    // TODO CL_OUT_OF_RESOURCES if there is a failure to allocate resources
    // required by the OpenCL implementation on the device.
    // TODO CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources
    // required by the OpenCL implementation on the host.

    return CL_INVALID_OPERATION;
}

cl_int CLVK_API_CALL clRetainDevice(cl_device_id device) {
    TRACE_FUNCTION("device", (uintptr_t)device);
    LOG_API_CALL("device = %p", device);

    if (!is_valid_device(device)) {
        return CL_INVALID_DEVICE;
    }

    return CL_SUCCESS;
}

cl_int CLVK_API_CALL clReleaseDevice(cl_device_id device) {
    TRACE_FUNCTION("device", (uintptr_t)device);
    LOG_API_CALL("device = %p", device);

    if (!is_valid_device(device)) {
        return CL_INVALID_DEVICE;
    }

    return CL_SUCCESS;
}

// Context APIs
cl_context CLVK_API_CALL clCreateContext(
    const cl_context_properties* properties, cl_uint num_devices,
    const cl_device_id* devices,
    void(CL_CALLBACK* pfn_notify)(const char*, const void*, size_t, void*),
    void* user_data, cl_int* errcode_ret) {
    TRACE_FUNCTION("num_devices", num_devices);
    LOG_API_CALL("properties = %p, num_devices = %u, devices = %p, pfn_notify "
                 "= %p, user_data = %p, errcode_ret = %p",
                 properties, num_devices, devices, pfn_notify, user_data,
                 errcode_ret);

    if ((devices == nullptr) || (num_devices == 0) ||
        ((pfn_notify == nullptr) && (user_data != nullptr))) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_VALUE;
        }
        return nullptr;
    }

    if (num_devices > 1) {
        cvk_error("Only one device per context is supported.");
        return nullptr;
    }

    cl_context context =
        new cvk_context(icd_downcast(devices[0]), properties, user_data);

    if (errcode_ret != nullptr) {
        *errcode_ret = CL_SUCCESS;
    }

    return context;
}

cl_context CLVK_API_CALL clCreateContextFromType(
    const cl_context_properties* properties, cl_device_type device_type,
    void(CL_CALLBACK* pfn_notify)(const char*, const void*, size_t, void*),
    void* user_data, cl_int* errcode_ret) {
    TRACE_FUNCTION("device_type",
                   TRACE_STRING(cl_device_type_to_string(device_type)));
    LOG_API_CALL("properties = %p, device_type = %lu (%s), pfn_notify = %p, "
                 "user_data = %p, errcode_ret = %p",
                 properties, device_type, cl_device_type_to_string(device_type),
                 pfn_notify, user_data, errcode_ret);

    cl_device_id device;

    // TODO introduce cvk_ functions to get correct logging
    cl_int err = clGetDeviceIDs(nullptr, device_type, 1, &device, nullptr);

    if (err == CL_SUCCESS) {
        return clCreateContext(properties, 1, &device, pfn_notify, user_data,
                               errcode_ret);
    } else {
        if (errcode_ret != nullptr) {
            *errcode_ret = err;
        }
        return nullptr;
    }
}

cl_int CLVK_API_CALL clRetainContext(cl_context context) {
    TRACE_FUNCTION("context", (uintptr_t)context);
    LOG_API_CALL("context = %p", context);

    if (!is_valid_context(context)) {
        return CL_INVALID_CONTEXT;
    }

    icd_downcast(context)->retain();

    return CL_SUCCESS;
}

cl_int CLVK_API_CALL clReleaseContext(cl_context context) {
    TRACE_FUNCTION("context", (uintptr_t)context);
    LOG_API_CALL("context = %p", context);

    if (!is_valid_context(context)) {
        return CL_INVALID_CONTEXT;
    }

    icd_downcast(context)->release();

    return CL_SUCCESS;
}

cl_int CLVK_API_CALL clSetContextDestructorCallback(
    cl_context context,
    void(CL_CALLBACK* pfn_notify)(cl_context context, void* user_data),
    void* user_data) {
    TRACE_FUNCTION("context", (uintptr_t)context);
    LOG_API_CALL("context = %p, pfn_notify = %p, user_data = %p", context,
                 pfn_notify, user_data);

    if (!is_valid_context(context)) {
        return CL_INVALID_CONTEXT;
    }

    if (pfn_notify == nullptr) {
        return CL_INVALID_VALUE;
    }

    icd_downcast(context)->add_destructor_callback(pfn_notify, user_data);

    return CL_SUCCESS;
}

cl_int CLVK_API_CALL clGetContextInfo(cl_context ctx,
                                      cl_context_info param_name,
                                      size_t param_value_size,
                                      void* param_value,
                                      size_t* param_value_size_ret) {
    TRACE_FUNCTION("context", (uintptr_t)ctx, "param_name", param_name);
    LOG_API_CALL(
        "context = %p, param_name = %u, size = %zu, value = %p, size_ret = %p",
        ctx, param_name, param_value_size, param_value, param_value_size_ret);

    cl_int ret = CL_SUCCESS;
    size_t size_ret = 0;
    const void* copy_ptr = nullptr;
    cl_uint val_uint;
    cl_device_id val_device;

    if (!is_valid_context(ctx)) {
        return CL_INVALID_CONTEXT;
    }

    auto context = icd_downcast(ctx);

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
            size_ret =
                context->properties().size() * sizeof(cl_context_properties);
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
cl_int CLVK_API_CALL clWaitForEvents(cl_uint num_events,
                                     const cl_event* event_list) {
    TRACE_FUNCTION("num_events", num_events);
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

cl_int CLVK_API_CALL clEnqueueWaitForEvents(cl_command_queue command_queue,
                                            cl_uint num_events,
                                            const cl_event* event_list) {
    TRACE_FUNCTION("command_queue", (uintptr_t)command_queue, "num_events",
                   num_events);
    LOG_API_CALL("command_queue = %p, num_events = %u, event_list = %p",
                 command_queue, num_events, event_list);

    return CL_INVALID_OPERATION; // TODO implement
}

cl_int CLVK_API_CALL clReleaseEvent(cl_event event) {
    TRACE_FUNCTION("event", (uintptr_t)event);
    LOG_API_CALL("event = %p", event);

    if (!is_valid_event(event)) {
        return CL_INVALID_EVENT;
    }

    icd_downcast(event)->release();

    return CL_SUCCESS;
}

cl_int CLVK_API_CALL clRetainEvent(cl_event event) {
    TRACE_FUNCTION("event", (uintptr_t)event);
    LOG_API_CALL("event = %p", event);

    if (!is_valid_event(event)) {
        return CL_INVALID_EVENT;
    }

    icd_downcast(event)->retain();

    return CL_SUCCESS;
}

cl_event CLVK_API_CALL clCreateUserEvent(cl_context context,
                                         cl_int* errcode_ret) {
    TRACE_FUNCTION("context", (uintptr_t)context);
    LOG_API_CALL("context = %p, errcode_ret = %p", context, errcode_ret);

    if (!is_valid_context(context)) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_CONTEXT;
        }
    }

    auto event = new cvk_event(icd_downcast(context), nullptr, nullptr);

    if (errcode_ret != nullptr) {
        *errcode_ret = CL_SUCCESS;
    }

    return event;
}

cl_int CLVK_API_CALL clSetUserEventStatus(cl_event event,
                                          cl_int execution_status) {
    TRACE_FUNCTION(
        "event", (uintptr_t)event, "execution_status",
        TRACE_STRING(cl_command_execution_status_to_string(execution_status)));
    LOG_API_CALL("event = %p, execution_status = %d (%s)", event,
                 execution_status,
                 cl_command_execution_status_to_string(execution_status));

    if (!is_valid_event(event) || !icd_downcast(event)->is_user_event()) {
        return CL_INVALID_EVENT;
    }

    if (execution_status > CL_COMPLETE) {
        return CL_INVALID_VALUE;
    }

    icd_downcast(event)->set_status(execution_status);

    return CL_SUCCESS;
}

cl_int CLVK_API_CALL clSetEventCallback(
    cl_event event, cl_int command_exec_callback_type,
    void(CL_CALLBACK* pfn_event_notify)(cl_event event,
                                        cl_int event_command_exec_status,
                                        void* user_data),
    void* user_data) {
    TRACE_FUNCTION("event", (uintptr_t)event, "execution_status",
                   TRACE_STRING(cl_command_execution_status_to_string(
                       command_exec_callback_type)));
    LOG_API_CALL(
        "event = %p, callback_type = %d (%s), pfn_event_notify = %p, user_data "
        "= %p",
        event, command_exec_callback_type,
        cl_command_execution_status_to_string(command_exec_callback_type),
        pfn_event_notify, user_data);

    if (!is_valid_event(event)) {
        return CL_INVALID_EVENT;
    }

    if (pfn_event_notify == nullptr) {
        return CL_INVALID_VALUE;
    }

    icd_downcast(event)->register_callback(command_exec_callback_type,
                                           pfn_event_notify, user_data);

    return CL_SUCCESS;
}

cl_int cvk_enqueue_marker_with_wait_list(cvk_command_queue* command_queue,
                                         cl_uint num_events_in_wait_list,
                                         const cl_event* event_wait_list,
                                         cl_event* event) {
    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    if (!is_valid_event_wait_list(num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    auto cmd = new cvk_command_dep(command_queue, CL_COMMAND_MARKER);

    return command_queue->enqueue_command_with_deps(
        cmd, num_events_in_wait_list, event_wait_list, event);
}

cl_int CLVK_API_CALL clEnqueueMarkerWithWaitList(
    cl_command_queue command_queue, cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list, cl_event* event) {
    TRACE_FUNCTION("command_queue", (uintptr_t)command_queue,
                   "num_events_in_wait_list", num_events_in_wait_list);
    LOG_API_CALL("command_queue = %p, num_events_in_wait_list = %u, "
                 "event_wait_list = %p, event = %p",
                 command_queue, num_events_in_wait_list, event_wait_list,
                 event);

    return cvk_enqueue_marker_with_wait_list(icd_downcast(command_queue),
                                             num_events_in_wait_list,
                                             event_wait_list, event);
}

cl_int CLVK_API_CALL clEnqueueMarker(cl_command_queue command_queue,
                                     cl_event* event) {
    TRACE_FUNCTION("command_queue", (uintptr_t)command_queue);
    LOG_API_CALL("command_queue = %p, event = %p", command_queue, event);

    return cvk_enqueue_marker_with_wait_list(icd_downcast(command_queue), 0,
                                             nullptr, event);
}

cl_int cvk_enqueue_barrier_with_wait_list(cvk_command_queue* command_queue,
                                          cl_uint num_events_in_wait_list,
                                          const cl_event* event_wait_list,
                                          cl_event* event) {
    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    if (!is_valid_event_wait_list(num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    auto cmd = new cvk_command_dep(command_queue, CL_COMMAND_BARRIER);

    return command_queue->enqueue_command_with_deps(
        cmd, num_events_in_wait_list, event_wait_list, event);
}

cl_int CLVK_API_CALL clEnqueueBarrierWithWaitList(
    cl_command_queue command_queue, cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list, cl_event* event) {
    TRACE_FUNCTION("command_queue", (uintptr_t)command_queue,
                   "num_events_in_wait_list", num_events_in_wait_list);
    LOG_API_CALL("command_queue = %p, num_events_in_wait_list = %u, "
                 "event_wait_list = %p, event = %p",
                 command_queue, num_events_in_wait_list, event_wait_list,
                 event);

    return cvk_enqueue_barrier_with_wait_list(icd_downcast(command_queue),
                                              num_events_in_wait_list,
                                              event_wait_list, event);
}

cl_int CLVK_API_CALL clEnqueueBarrier(cl_command_queue command_queue) {
    TRACE_FUNCTION("command_queue", (uintptr_t)command_queue);
    LOG_API_CALL("command_queue = %p", command_queue);

    return cvk_enqueue_barrier_with_wait_list(icd_downcast(command_queue), 0,
                                              nullptr, nullptr);
}

cl_int CLVK_API_CALL clGetEventInfo(cl_event evt, cl_event_info param_name,
                                    size_t param_value_size, void* param_value,
                                    size_t* param_value_size_ret) {
    TRACE_FUNCTION("event", (uintptr_t)evt, "param_name", param_name);
    LOG_API_CALL("event = %p, param_name = %x, param_value_size = %zu, "
                 "param_value = %p, param_value_size_ret = %p",
                 evt, param_name, param_value_size, param_value,
                 param_value_size_ret);

    cl_int ret = CL_SUCCESS;
    size_t ret_size = 0;
    const void* copy_ptr = nullptr;
    cl_uint val_uint;
    cl_int val_int;
    cl_context val_context;
    cl_command_type val_command_type;
    cl_command_queue val_command_queue;

    auto event = icd_downcast(evt);

    if (!is_valid_event(event)) {
        return CL_INVALID_EVENT;
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
cvk_create_command_queue(cl_context context, cl_device_id device,
                         cl_command_queue_properties properties,
                         std::vector<cl_queue_properties>&& properties_array,
                         cl_int* errcode_ret) {

    if (!is_valid_context(context)) {
        *errcode_ret = CL_INVALID_CONTEXT;
        return nullptr;
    }

    if (!is_valid_device(device) ||
        icd_downcast(context)->device() != icd_downcast(device)) {
        *errcode_ret = CL_INVALID_DEVICE;
        return nullptr;
    }

    if (!config.ignore_out_of_order_execution()) {
        // We do not support out of order command queues so this must fail
        if (properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
            *errcode_ret = CL_INVALID_QUEUE_PROPERTIES;
            return nullptr;
        }
    }
    auto queue = std::make_unique<cvk_command_queue>(
        icd_downcast(context), icd_downcast(device), properties,
        std::move(properties_array));

    cl_int err = queue->init();

    *errcode_ret = err;

    if (err != CL_SUCCESS) {
        return nullptr;
    } else {
        return queue.release();
    }
}

cl_command_queue CLVK_API_CALL clCreateCommandQueue(
    cl_context context, cl_device_id device,
    cl_command_queue_properties properties, cl_int* errcode_ret) {
    TRACE_FUNCTION("context", (uintptr_t)context, "device", (uintptr_t)device,
                   "properties", properties);
    LOG_API_CALL(
        "context = %p, device = %p, properties = %lu, errcode_ret = %p",
        context, device, properties, errcode_ret);

    cl_int err;
    std::vector<cl_queue_properties> properties_array;
    auto ret = cvk_create_command_queue(context, device, properties,
                                        std::move(properties_array), &err);

    if (errcode_ret != nullptr) {
        *errcode_ret = err;
    }

    return ret;
}

cl_command_queue cvk_create_command_queue_with_properties(
    cl_context context, cl_device_id device,
    const cl_queue_properties* properties, cl_int* errcode_ret) {
    cl_command_queue_properties props = 0;

    std::vector<cl_queue_properties> properties_array;

    if (properties) {
        while (*properties) {
            auto key = *properties;
            auto value = *(properties + 1);

            properties_array.push_back(key);
            properties_array.push_back(value);

            if (key == CL_QUEUE_PROPERTIES) {
                props = value;
            } else {
                if (errcode_ret != nullptr) {
                    *errcode_ret = CL_INVALID_VALUE;
                }
                return nullptr;
            }

            properties += 2;
        }

        properties_array.push_back(0);
    }

    cl_int err;
    auto ret = cvk_create_command_queue(context, device, props,
                                        std::move(properties_array), &err);

    if (errcode_ret != nullptr) {
        *errcode_ret = err;
    }

    return ret;
}

cl_command_queue CLVK_API_CALL clCreateCommandQueueWithProperties(
    cl_context context, cl_device_id device,
    const cl_queue_properties* properties, cl_int* errcode_ret) {
    TRACE_FUNCTION("context", (uintptr_t)context, "device", (uintptr_t)device);
    LOG_API_CALL("context = %p, device = %p, properties = %p, errcode_ret = %p",
                 context, device, properties, errcode_ret);

    return cvk_create_command_queue_with_properties(context, device, properties,
                                                    errcode_ret);
}

cl_command_queue CLVK_API_CALL clCreateCommandQueueWithPropertiesKHR(
    cl_context context, cl_device_id device,
    const cl_queue_properties* properties, cl_int* errcode_ret) {
    TRACE_FUNCTION("context", (uintptr_t)context, "device", (uintptr_t)device);
    LOG_API_CALL("context = %p, device = %p, properties = %p, errcode_ret = %p",
                 context, device, properties, errcode_ret);

    return cvk_create_command_queue_with_properties(context, device, properties,
                                                    errcode_ret);
}

cl_int CLVK_API_CALL clReleaseCommandQueue(cl_command_queue command_queue) {
    TRACE_FUNCTION("command_queue", (uintptr_t)command_queue);
    LOG_API_CALL("command_queue = %p", command_queue);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    cl_int err = icd_downcast(command_queue)->flush();

    icd_downcast(command_queue)->release();

    return err;
}

cl_int CLVK_API_CALL clRetainCommandQueue(cl_command_queue command_queue) {
    TRACE_FUNCTION("command_queue", (uintptr_t)command_queue);
    LOG_API_CALL("command_queue = %p", command_queue);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    icd_downcast(command_queue)->retain();
    return CL_SUCCESS;
}

cl_int CLVK_API_CALL clGetCommandQueueInfo(cl_command_queue cq,
                                           cl_command_queue_info param_name,
                                           size_t param_value_size,
                                           void* param_value,
                                           size_t* param_value_size_ret) {
    TRACE_FUNCTION("command_queue", (uintptr_t)cq, "param_name", param_name);
    LOG_API_CALL("command_queue = %p, param_name = %x, param_value_size = %zu, "
                 "param_value = %p, param_value_size_ret = %p",
                 cq, param_name, param_value_size, param_value,
                 param_value_size_ret);

    cl_int ret = CL_SUCCESS;
    size_t ret_size = 0;
    const void* copy_ptr = nullptr;
    cl_uint val_uint;
    cl_context val_context;
    cl_device_id val_device;
    cl_command_queue_properties val_properties;
    cl_command_queue val_queue;

    auto command_queue = icd_downcast(cq);

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
    case CL_QUEUE_SIZE:
        val_uint = 0;
        copy_ptr = &val_uint;
        ret_size = sizeof(val_uint);
        ret = CL_INVALID_COMMAND_QUEUE;
        break;
    case CL_QUEUE_DEVICE_DEFAULT:
        val_queue = nullptr;
        copy_ptr = &val_queue;
        ret_size = sizeof(val_queue);
        break;
    case CL_QUEUE_PROPERTIES_ARRAY:
        copy_ptr = command_queue->properties_array().data();
        ret_size = command_queue->properties_array().size() *
                   sizeof(cl_queue_properties);
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

cl_int CLVK_API_CALL clSetDefaultDeviceCommandQueue(
    cl_context context, cl_device_id device, cl_command_queue command_queue) {
    TRACE_FUNCTION("context", (uintptr_t)context, "device", (uintptr_t)device,
                   "command_queue", (uintptr_t)command_queue);
    LOG_API_CALL("context = %p, device = %p, command_queue = %p", context,
                 device, command_queue);
    return CL_INVALID_OPERATION;
}

cl_int CLVK_API_CALL clSetCommandQueueProperty(
    cl_command_queue command_queue, cl_command_queue_properties properties,
    cl_bool enable, cl_command_queue_properties* old_properties) {
    TRACE_FUNCTION("command_queue", (uintptr_t)command_queue, "properties",
                   properties);
    LOG_API_CALL("command_queue = %p, properties = %lx, enable = %d, "
                 "old_properties = %p",
                 command_queue, properties, enable, old_properties);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    // TODO validate properties
    // TODO support

    return CL_INVALID_OPERATION;
}

// Memory Object APIs
static cl_mem CLVK_API_CALL cvk_create_buffer_with_properties(
    cl_context context, const cl_mem_properties* properties, cl_mem_flags flags,
    size_t size, void* host_ptr, cl_int* errcode_ret) {
    CVK_ASSERT(errcode_ret != nullptr);

    // Validate context
    if (!is_valid_context(context)) {
        *errcode_ret = CL_INVALID_CONTEXT;
        return nullptr;
    }

    // Validate properties
    std::vector<cl_mem_properties> props;

    if (properties != nullptr) {
        while (*properties) {
            // We dont't currently support any properties so return an error
            *errcode_ret = CL_INVALID_PROPERTY;
            return nullptr;
            props.push_back(*properties);
            properties++;
        }
        props.push_back(0);
    }

    // Validate flags
    if ((flags & CL_MEM_READ_WRITE) && (flags & CL_MEM_WRITE_ONLY)) {
        *errcode_ret = CL_INVALID_VALUE;
        return nullptr;
    }
    if ((flags & CL_MEM_READ_ONLY) &&
        (flags & (CL_MEM_WRITE_ONLY | CL_MEM_READ_WRITE))) {
        *errcode_ret = CL_INVALID_VALUE;
        return nullptr;
    }
    if ((flags & (CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR)) &&
        (flags & CL_MEM_USE_HOST_PTR)) {
        *errcode_ret = CL_INVALID_VALUE;
        return nullptr;
    }
    if ((flags & CL_MEM_HOST_READ_ONLY) && (flags & CL_MEM_HOST_WRITE_ONLY)) {
        *errcode_ret = CL_INVALID_VALUE;
        return nullptr;
    }
    if ((flags & CL_MEM_HOST_NO_ACCESS) &&
        (flags & (CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_WRITE_ONLY))) {
        *errcode_ret = CL_INVALID_VALUE;
        return nullptr;
    }

    // Validate size
    // TODO CL_INVALID_BUFFER_SIZE if CL_MEM_USE_HOST_PTR is set in flags and
    // host_ptr is a pointer returned by clSVMAlloc and size is greater than the
    // size passed to clSVMAlloc.
    if ((size == 0) ||
        (!icd_downcast(context)->is_mem_alloc_size_valid(size))) {
        *errcode_ret = CL_INVALID_BUFFER_SIZE;
        return nullptr;
    }

    // Validate host_ptr
    if ((host_ptr == nullptr) &&
        (flags & (CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR))) {
        *errcode_ret = CL_INVALID_HOST_PTR;
        return nullptr;
    }
    if ((host_ptr != nullptr) &&
        !(flags & (CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR))) {
        *errcode_ret = CL_INVALID_HOST_PTR;
        return nullptr;
    }

    auto buffer = cvk_buffer::create(icd_downcast(context), flags, size,
                                     host_ptr, std::move(props), errcode_ret);

    if (*errcode_ret != CL_SUCCESS) {
        return nullptr;
    } else {
        return buffer.release();
    }
}

cl_mem CLVK_API_CALL clCreateBuffer(cl_context context, cl_mem_flags flags,
                                    size_t size, void* host_ptr,
                                    cl_int* errcode_ret) {
    TRACE_FUNCTION("context", (uintptr_t)context, "flags", flags);
    LOG_API_CALL("context = %p, flags = %lu, size = %zu, host_ptr = %p, "
                 "errcode_ret = %p",
                 context, flags, size, host_ptr, errcode_ret);

    cl_int err;
    auto buffer = cvk_create_buffer_with_properties(context, nullptr, flags,
                                                    size, host_ptr, &err);

    if (errcode_ret != nullptr) {
        *errcode_ret = err;
    }

    return buffer;
}

cl_mem CLVK_API_CALL clCreateBufferWithProperties(
    cl_context context, const cl_mem_properties* properties, cl_mem_flags flags,
    size_t size, void* host_ptr, cl_int* errcode_ret) {

    TRACE_FUNCTION("context", (uintptr_t)context, "flags", flags);
    LOG_API_CALL("context = %p, properties = %p, flags = %lx, size = %zu, "
                 "host_ptr = %p, errcode_ret = %p",
                 context, properties, flags, size, host_ptr, errcode_ret);

    cl_int err;
    auto buffer = cvk_create_buffer_with_properties(context, properties, flags,
                                                    size, host_ptr, &err);

    if (errcode_ret != nullptr) {
        *errcode_ret = err;
    }

    return buffer;
}

cl_mem CLVK_API_CALL clCreateSubBuffer(cl_mem buf, cl_mem_flags flags,
                                       cl_buffer_create_type buffer_create_type,
                                       const void* buffer_create_info,
                                       cl_int* errcode_ret) {
    TRACE_FUNCTION("buffer", (uintptr_t)buf, "flags", flags,
                   "buffer_create_type", buffer_create_type);
    LOG_API_CALL("buffer = %p, flags = %lu, buffer_create_type = %u, "
                 "buffer_create_info = %p, errcode_ret = %p",
                 buf, flags, buffer_create_type, buffer_create_info,
                 errcode_ret);

    auto buffer = static_cast<cvk_buffer*>(buf);

    if (!is_valid_buffer(buffer) || buffer->is_sub_buffer()) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_MEM_OBJECT;
        }
        return nullptr;
    }

    // TODO CL_INVALID_VALUE if buffer was created with CL_MEM_WRITE_ONLY and
    // flags specifies CL_MEM_READ_WRITE or CL_MEM_READ_ONLY, or if buffer was
    // created with CL_MEM_READ_ONLY and flags specifies CL_MEM_READ_WRITE or
    // CL_MEM_WRITE_ONLY, or if flags specifies CL_MEM_USE_HOST_PTR or
    // CL_MEM_ALLOC_HOST_PTR or CL_MEM_COPY_HOST_PTR.
    // TODO CL_INVALID_VALUE if buffer was created with CL_MEM_HOST_WRITE_ONLY
    // and flags specifies CL_MEM_HOST_READ_ONLY or if buffer was created with
    // CL_MEM_HOST_READ_ONLY and flags specifies CL_MEM_HOST_WRITE_ONLY, or if
    // buffer was created with CL_MEM_HOST_NO_ACCESS and flags specifies
    // CL_MEM_HOST_READ_ONLY or CL_MEM_HOST_WRITE_ONLY.
    // TODO CL_INVALID_VALUE if value specified in buffer_create_type is not
    // valid.
    // TODO CL_INVALID_VALUE if value(s) specified in buffer_create_info (for a
    // given buffer_create_type) is not valid or if buffer_create_info is NULL.
    // TODO CL_INVALID_BUFFER_SIZE if size is 0.
    // TODO CL_MEM_OBJECT_ALLOCATION_FAILURE if there is a failure to allocate
    // memory for sub-buffer object.
    // TODO CL_OUT_OF_RESOURCES if there is a failure to allocate resources
    // required by the OpenCL implementation on the device.
    // TODO CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources
    // required by the OpenCL implementation on the host.

    if (buffer_create_type != CL_BUFFER_CREATE_TYPE_REGION) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_VALUE;
        }
        return nullptr;
    }

    auto region = static_cast<const cl_buffer_region*>(buffer_create_info);
    LOG_API_CALL("CL_BUFFER_CREATE_TYPE_REGION, origin = %zu, size = %zu",
                 region->origin, region->size);

    cl_int err = CL_SUCCESS;
    auto sub = buffer->create_subbuffer(flags, region->origin, region->size);

    if (sub == nullptr) {
        err = CL_OUT_OF_RESOURCES;
    }

    if (errcode_ret != nullptr) {
        *errcode_ret = err;
    }

    return sub;
}

cl_int CLVK_API_CALL clRetainMemObject(cl_mem memobj) {
    TRACE_FUNCTION("memobj", (uintptr_t)memobj);
    LOG_API_CALL("memobj = %p", memobj);

    if (!is_valid_mem_object(memobj)) {
        return CL_INVALID_MEM_OBJECT;
    }

    icd_downcast(memobj)->retain();

    return CL_SUCCESS;
}

cl_int CLVK_API_CALL clReleaseMemObject(cl_mem memobj) {
    TRACE_FUNCTION("memobj", (uintptr_t)memobj);
    LOG_API_CALL("memobj = %p", memobj);

    if (!is_valid_mem_object(memobj)) {
        return CL_INVALID_MEM_OBJECT;
    }

    icd_downcast(memobj)->release();

    return CL_SUCCESS;
}

cl_int CLVK_API_CALL clSetMemObjectDestructorCallback(
    cl_mem memobj,
    void(CL_CALLBACK* pfn_notify)(cl_mem memobj, void* user_data),
    void* user_data) {
    TRACE_FUNCTION("memobj", (uintptr_t)memobj);
    LOG_API_CALL("memobj = %p, pfn_notify = %p, user_data = %p", memobj,
                 pfn_notify, user_data);

    if (!is_valid_mem_object(memobj)) {
        return CL_INVALID_MEM_OBJECT;
    }

    if (pfn_notify == nullptr) {
        return CL_INVALID_VALUE;
    }

    icd_downcast(memobj)->add_destructor_callback(pfn_notify, user_data);

    return CL_SUCCESS;
}

cl_int CLVK_API_CALL clEnqueueMigrateMemObjects(
    cl_command_queue cq, cl_uint num_mem_objects, const cl_mem* mem_objects,
    cl_mem_migration_flags flags, cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list, cl_event* event) {
    TRACE_FUNCTION("command_queue", (uintptr_t)cq, "num_mem_objects",
                   num_mem_objects, "flags", flags, "num_events_in_wait_list",
                   num_events_in_wait_list);
    LOG_API_CALL("command_queue = %p, num_mem_objects = %u, mem_objects = %p, "
                 "flags = %lx, num_events_in_wait_list = %u, "
                 "event_wait_list = %p, event = %p",
                 cq, num_mem_objects, mem_objects, flags,
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

    auto command_queue = icd_downcast(cq);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    if (!is_valid_event_wait_list(num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    if (!is_same_context(command_queue, num_events_in_wait_list,
                         event_wait_list)) {
        return CL_INVALID_CONTEXT;
    }

    for (cl_uint i = 0; i < num_mem_objects; i++) {
        if (!is_same_context(command_queue, mem_objects[i])) {
            return CL_INVALID_CONTEXT;
        }
    }

    auto cmd =
        new cvk_command_dep(command_queue, CL_COMMAND_MIGRATE_MEM_OBJECTS);

    return command_queue->enqueue_command_with_deps(
        cmd, num_events_in_wait_list, event_wait_list, event);
}

cl_int CLVK_API_CALL clGetMemObjectInfo(cl_mem mem, cl_mem_info param_name,
                                        size_t param_value_size,
                                        void* param_value,
                                        size_t* param_value_size_ret) {
    TRACE_FUNCTION("memobj", (uintptr_t)mem, "param_name", param_name);
    LOG_API_CALL("memobj = %p, param_name = %x, param_value_size = %zu, "
                 "param_value = %p, param_value_size_ret = %p",
                 mem, param_name, param_value_size, param_value,
                 param_value_size_ret);

    cl_int ret = CL_SUCCESS;
    size_t ret_size = 0;
    const void* copy_ptr = nullptr;
    cl_uint val_uint;
    cl_context val_context;
    cl_mem_object_type val_object_type;
    cl_mem_flags val_flags;
    size_t val_sizet;
    cl_mem val_memobj;
    void* val_ptr;
    cl_bool val_bool;

    auto memobj = icd_downcast(mem);

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
        if (memobj->is_image_type()) {
            auto img = static_cast<cvk_image*>(memobj);
            val_memobj = img->buffer();
        } else {
            val_memobj = memobj->parent();
        }
        copy_ptr = &val_memobj;
        ret_size = sizeof(val_memobj);
        break;
    case CL_MEM_OFFSET:
        val_sizet = memobj->parent_offset();
        copy_ptr = &val_sizet;
        ret_size = sizeof(val_sizet);
        break;
    case CL_MEM_HOST_PTR:
        val_ptr = nullptr;
        if (memobj->has_any_flag(CL_MEM_USE_HOST_PTR)) {
            val_ptr = memobj->host_ptr();
        }
        copy_ptr = &val_ptr;
        ret_size = sizeof(val_ptr);
        break;
    case CL_MEM_USES_SVM_POINTER:
        val_bool = CL_FALSE;
        copy_ptr = &val_bool;
        ret_size = sizeof(val_bool);
        break;
    case CL_MEM_PROPERTIES:
        copy_ptr = memobj->properties().data();
        ret_size = memobj->properties().size() * sizeof(cl_mem_properties);
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
cl_program CLVK_API_CALL clCreateProgramWithSource(cl_context context,
                                                   cl_uint count,
                                                   const char** strings,
                                                   const size_t* lengths,
                                                   cl_int* errcode_ret) {
    TRACE_FUNCTION("context", (uintptr_t)context, "count", count);
    LOG_API_CALL("context = %p, count = %u, lengths = %p", context, count,
                 lengths);

    if (!is_valid_context(context)) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_CONTEXT;
        }
        return nullptr;
    }
    if (count == 0 || strings == nullptr) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_VALUE;
        }
        return nullptr;
    }

    cvk_program* prog = new cvk_program(icd_downcast(context));

    for (cl_uint i = 0; i < count; i++) {
        if (strings[i] == nullptr) {
            if (errcode_ret != nullptr) {
                *errcode_ret = CL_INVALID_VALUE;
            }
            return nullptr;
        }
        size_t len = (lengths != nullptr) ? lengths[i] : 0;
        prog->append_source(strings[i], len);
    }

    if (errcode_ret != nullptr) {
        *errcode_ret = CL_SUCCESS;
    }

    cvk_debug_fn("\n%s", prog->source().c_str());

    return prog;
}

cl_program CLVK_API_CALL clCreateProgramWithBinary(
    cl_context ctx, cl_uint num_devices, const cl_device_id* device_list,
    const size_t* lengths, const unsigned char** binaries,
    cl_int* binary_status, cl_int* errcode_ret) {
    TRACE_FUNCTION("context", (uintptr_t)ctx, "num_devices", num_devices);
    LOG_API_CALL("context = %p, num_devices = %u, device_list = %p, lengths = "
                 "%p, binaries = %p, binary_status = %p, errcode_ret = %p",
                 ctx, num_devices, device_list, lengths, binaries,
                 binary_status, errcode_ret);

    auto context = icd_downcast(ctx);

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

    if (icd_downcast(device_list[0]) != context->device()) {
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

    cvk_program* prog = new cvk_program(icd_downcast(context));

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

cl_program CLVK_API_CALL clCreateProgramWithBuiltInKernels(
    cl_context context, cl_uint num_devices, const cl_device_id* device_list,
    const char* kernel_names, cl_int* errcode_ret) {
    TRACE_FUNCTION("context", (uintptr_t)context, "num_devices", num_devices);
    LOG_API_CALL("context = %p, num_devices = %u, device_list = %p, "
                 "kernel_names = \"%s\", errcode_ret = %p",
                 context, num_devices, device_list, kernel_names, errcode_ret);

    if (!is_valid_context(context)) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_CONTEXT;
        }
        return nullptr;
    }

    if ((device_list == nullptr) || (num_devices == 0)) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_VALUE;
        }
        return nullptr;
    }

    if (kernel_names == nullptr) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_VALUE;
        }
        return nullptr;
    }

    // TODO CL_INVALID_DEVICE if any device in device_list is not in the list of
    // devices associated with context.

    if (errcode_ret != nullptr) {
        *errcode_ret =
            CL_INVALID_VALUE; // Since no built-in kernels are supported
    }

    return nullptr;
}

cl_int CLVK_API_CALL
clBuildProgram(cl_program prog, cl_uint num_devices,
               const cl_device_id* device_list, const char* options,
               void(CL_CALLBACK* pfn_notify)(cl_program /* program */,
                                             void* /* user_data */),
               void* user_data) {
    TRACE_FUNCTION("program", (uintptr_t)prog, "num_devices", num_devices);
    LOG_API_CALL("program = %p, num_device = %d, device_list = %p, options = "
                 "%s, pfn_notify = %p, user_data = %p",
                 prog, num_devices, device_list, options, pfn_notify,
                 user_data);

    auto program = icd_downcast(prog);

    if (!is_valid_program(program)) {
        return CL_INVALID_PROGRAM;
    }

    if (((num_devices > 0) && (device_list == nullptr)) ||
        ((num_devices == 0) && (device_list != nullptr))) {
        return CL_INVALID_VALUE;
    }

    if ((pfn_notify == nullptr) && (user_data != nullptr)) {
        return CL_INVALID_VALUE;
    }

    // TODO CL_INVALID_DEVICE if OpenCL devices listed in device_list are not in
    // the list of devices associated with program.
    // TODO CL_INVALID_BINARY if program is created with
    // clCreateProgramWithBinary and devices listed in device_list do not have a
    // valid program binary loaded.
    // TODO CL_INVALID_BUILD_OPTIONS if the build options specified by options
    // are invalid.
    build_operation build_op = build_operation::build;
    if (program->loaded_from_binary()) {
        build_op = build_operation::build_binary;
    } else if (!is_compiler_available(num_devices, device_list)) {
        return CL_COMPILER_NOT_AVAILABLE;
    }
    // TODO CL_BUILD_PROGRAM_FAILURE if there is a failure to build the program
    // executable. This error will be returned if clBuildProgram does not return
    // until the build has completed.
    // TODO CL_INVALID_OPERATION if there are kernel objects attached to
    // program.
    // TODO CL_INVALID_OPERATION if program was not created with
    // clCreateProgramWithSource or clCreateProgramWithBinary or
    // clCreateProgramWithILKHR.

    return program->build(build_op, num_devices, device_list, options, 0,
                          nullptr, nullptr, pfn_notify, user_data);
}

cl_int CLVK_API_CALL clCompileProgram(
    cl_program prog, cl_uint num_devices, const cl_device_id* device_list,
    const char* options, cl_uint num_input_headers,
    const cl_program* input_headers, const char** header_include_names,
    void(CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data) {
    TRACE_FUNCTION("program", (uintptr_t)prog, "num_devices", num_devices,
                   "num_input_headers", num_input_headers);
    LOG_API_CALL("program = %p, num_devices = %u, device_list = %p, options = "
                 "%p, num_input_headers = %u, input_headers = %p, "
                 "header_include_names = %p, pfn_notify = %p, user_data = %p",
                 prog, num_devices, device_list, options, num_input_headers,
                 input_headers, header_include_names, pfn_notify, user_data);

    auto program = icd_downcast(prog);

    if (!is_valid_program(program)) {
        return CL_INVALID_PROGRAM;
    }

    if (((device_list == nullptr) && (num_devices > 0)) ||
        ((device_list != nullptr) && (num_devices == 0))) {
        return CL_INVALID_VALUE;
    }

    if (((num_input_headers == 0) &&
         ((header_include_names != nullptr) || (input_headers != nullptr))) ||
        ((num_input_headers != 0) &&
         ((header_include_names == nullptr) || (input_headers == nullptr)))) {
        return CL_INVALID_VALUE;
    }

    if ((pfn_notify == nullptr) && (user_data != nullptr)) {
        return CL_INVALID_VALUE;
    }

    // TODO CL_INVALID_DEVICE if OpenCL devices listed in device_list are not in
    // the list of devices associated with program.
    // TODO CL_INVALID_COMPILER_OPTIONS if the compiler options specified by
    // options are invalid.

    if (!is_compiler_available(num_devices, device_list)) {
        return CL_COMPILER_NOT_AVAILABLE;
    }
    // TODO CL_COMPILE_PROGRAM_FAILURE if there is a failure to compile the
    // program source. This error will be returned if clCompileProgram does not
    // return until the compile has completed.
    // TODO CL_INVALID_OPERATION if there are kernel objects attached to
    // program.
    if (program->loaded_from_binary()) {
        return CL_INVALID_OPERATION;
    }

    // TODO Validate program
    return program->build(build_operation::compile, num_devices, device_list,
                          options, num_input_headers, input_headers,
                          header_include_names, pfn_notify, user_data);
}

cl_program CLVK_API_CALL clLinkProgram(
    cl_context context, cl_uint num_devices, const cl_device_id* device_list,
    const char* options, cl_uint num_input_programs,
    const cl_program* input_programs,
    void(CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data, cl_int* errcode_ret) {
    TRACE_FUNCTION("context", (uintptr_t)context, "num_devices", num_devices,
                   "num_input_programs", num_input_programs);
    LOG_API_CALL("context = %p, num_devices = %d, device_list = %p, options = "
                 "%p, num_input_programs = %d, input_programs = %p, pfn_notify "
                 "= %p, user_data = %p, errcode_ret = %p",
                 context, num_devices, device_list, options, num_input_programs,
                 input_programs, pfn_notify, user_data, errcode_ret);

    if (!is_valid_context(context)) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_CONTEXT;
        }
        return nullptr;
    }

    if (((device_list == nullptr) && (num_devices > 0)) ||
        ((device_list != nullptr) && (num_devices == 0))) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_VALUE;
        }
        return nullptr;
    }

    if (((input_programs == nullptr) && (num_input_programs == 0)) ||
        ((num_input_programs == 0) && (input_programs != nullptr)) ||
        ((num_input_programs != 0) && (input_programs == nullptr))) {
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

    // TODO CL_INVALID_DEVICE if OpenCL devices listed in device_list are not in
    // the list of devices associated with context.
    if (!is_compiler_available(num_devices, device_list)) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_LINKER_NOT_AVAILABLE;
        }
        return nullptr;
    }
    // TODO CL_INVALID_LINKER_OPTIONS if the linker options specified by options
    // are invalid
    // TODO CL_INVALID_OPERATION if the rules for devices containing compiled
    // binaries or libraries as described in input_programs argument above are
    // not followed.
    for (cl_uint i = 0; i < num_input_programs; i++) {
        if (!icd_downcast(input_programs[i])->can_be_linked()) {
            if (errcode_ret != nullptr) {
                *errcode_ret = CL_INVALID_OPERATION;
            }
            return nullptr;
        }
    }

    cvk_program* prog_ret = new cvk_program(icd_downcast(context));

    cl_int ret = prog_ret->build(
        build_operation::link, num_devices, device_list, options,
        num_input_programs, input_programs, nullptr, pfn_notify, user_data);

    if (errcode_ret != nullptr) {
        *errcode_ret = ret;
    }

    return prog_ret;
}

cl_int CLVK_API_CALL clUnloadPlatformCompiler(cl_platform_id platform) {
    TRACE_FUNCTION("platform", (uintptr_t)platform);
    LOG_API_CALL("platform = %p", platform);

    if (!is_valid_platform(platform)) {
        return CL_INVALID_PLATFORM;
    }

    return CL_SUCCESS;
}

cl_int CLVK_API_CALL clUnloadCompiler() {
    TRACE_FUNCTION();
    LOG_API_CALL("%s", "");

    return CL_SUCCESS;
}

cl_int CLVK_API_CALL clGetProgramInfo(cl_program prog,
                                      cl_program_info param_name,
                                      size_t param_value_size,
                                      void* param_value,
                                      size_t* param_value_size_ret) {
    TRACE_FUNCTION("program", (uintptr_t)prog, "param_name", param_name);
    LOG_API_CALL("program = %p, param_name = %x, param_value_size = %zu, "
                 "param_value = %p, param_value_size_ret = %p",
                 prog, param_name, param_value_size, param_value,
                 param_value_size_ret);

    cl_int ret = CL_SUCCESS;
    size_t ret_size = 0;
    const void* copy_ptr = nullptr;
    cl_uint val_uint;
    cl_context val_context;
    size_t val_sizet;
    cl_bool val_bool;
    api_query_string val_string;
    std::vector<size_t> val_sizet_vec;
    std::vector<cl_device_id> val_devices;

    auto program = icd_downcast(prog);

    if (!is_valid_program(program)) {
        return CL_INVALID_PROGRAM;
    }

    // TODO CL_INVALID_PROGRAM_EXECUTABLE if param_name is
    // CL_PROGRAM_NUM_KERNELS or CL_PROGRAM_KERNEL_NAMES and a successful
    // program executable has not been built for at least one device in the list
    // of devices associated with program.

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
        for (auto dev : program->devices()) {
            val_devices.push_back(const_cast<cvk_device*>(dev));
        }
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
    case CL_PROGRAM_IL:
        copy_ptr = program->il().data();
        ret_size = program->il().size();
        break;
    case CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT:
    case CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT:
        val_bool = CL_FALSE;
        copy_ptr = &val_bool;
        ret_size = sizeof(val_bool);
        break;
    default:
        ret = CL_INVALID_VALUE;
    }

    if ((param_value != nullptr) && (copy_ptr != nullptr) &&
        (param_name != CL_PROGRAM_BINARIES)) {
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

cl_int CLVK_API_CALL clGetProgramBuildInfo(cl_program prog, cl_device_id dev,
                                           cl_program_build_info param_name,
                                           size_t param_value_size,
                                           void* param_value,
                                           size_t* param_value_size_ret) {
    TRACE_FUNCTION("program", (uintptr_t)prog, "device", (uintptr_t)dev,
                   "param_name", param_name);
    LOG_API_CALL("program = %p, device = %p, param_name = %x, param_value_size "
                 "= %zu, param_value = %p, param_value_size_ret = %p",
                 prog, dev, param_name, param_value_size, param_value,
                 param_value_size_ret);
    cl_int ret = CL_SUCCESS;
    size_t ret_size = 0;
    const void* copy_ptr = nullptr;
    cl_build_status val_status;
    api_query_string val_string;
    cl_program_binary_type val_binarytype;
    size_t val_sizet;

    auto device = icd_downcast(dev);
    auto program = icd_downcast(prog);

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
    case CL_PROGRAM_BUILD_LOG:
        copy_ptr = program->build_log(device).c_str();
        ret_size = program->build_log(device).size() + 1;
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
    case CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE:
        val_sizet = 0;
        copy_ptr = &val_sizet;
        ret_size = sizeof(val_sizet);
        break;
    default:
        ret = CL_INVALID_VALUE;
    }

    if ((param_value != nullptr) && (ret_size > param_value_size)) {
        ret = CL_INVALID_VALUE;
    } else if ((param_value != nullptr) && (copy_ptr != nullptr)) {
        memcpy(param_value, copy_ptr, std::min(param_value_size, ret_size));
    }

    if (param_value_size_ret != nullptr) {
        *param_value_size_ret = ret_size;
    }

    return ret;
}

cl_int CLVK_API_CALL clRetainProgram(cl_program program) {
    TRACE_FUNCTION("program", (uintptr_t)program);
    LOG_API_CALL("program = %p", program);

    if (!is_valid_program(program)) {
        return CL_INVALID_PROGRAM;
    }

    icd_downcast(program)->retain();
    return CL_SUCCESS;
}

cl_int CLVK_API_CALL clReleaseProgram(cl_program program) {
    TRACE_FUNCTION("program", (uintptr_t)program);
    LOG_API_CALL("program = %p", program);

    if (!is_valid_program(program)) {
        return CL_INVALID_PROGRAM;
    }

    icd_downcast(program)->release();
    return CL_SUCCESS;
}

cl_int CLVK_API_CALL clSetProgramReleaseCallback(
    cl_program program,
    void(CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data) {
    TRACE_FUNCTION("program", (uintptr_t)program);
    LOG_API_CALL("program = %p, pfn_notify = %p, user_data = %p", program,
                 pfn_notify, user_data);
    return CL_INVALID_OPERATION;
}

// Kernel Object APIs
cl_kernel cvk_create_kernel(cl_program program, const char* kernel_name,
                            cl_int* errcode_ret) {
    if (kernel_name == nullptr) {
        *errcode_ret = CL_INVALID_VALUE;
        return nullptr;
    }

    auto kernel =
        std::make_unique<cvk_kernel>(icd_downcast(program), kernel_name);

    *errcode_ret = kernel->init();

    if (*errcode_ret != CL_SUCCESS) {
        return nullptr;
    } else {
        return kernel.release();
    }
}

cl_kernel CLVK_API_CALL clCreateKernel(cl_program prog, const char* kernel_name,
                                       cl_int* errcode_ret) {
    TRACE_FUNCTION("program", (uintptr_t)prog, "kernel_name",
                   TRACE_STRING(kernel_name));
    LOG_API_CALL("program = %p, kernel_name = %s", prog, kernel_name);

    auto program = icd_downcast(prog);

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
    cl_kernel ret = cvk_create_kernel(prog, kernel_name, &err);

    if (errcode_ret != nullptr) {
        *errcode_ret = err;
    }

    return ret;
}

cl_int CLVK_API_CALL clCreateKernelsInProgram(cl_program prog,
                                              cl_uint num_kernels,
                                              cl_kernel* kernels,
                                              cl_uint* num_kernels_ret) {
    TRACE_FUNCTION("program", (uintptr_t)prog, "num_kernels", num_kernels);
    LOG_API_CALL(
        "program = %p, num_kernels = %u, kernels = %p, num_kernels_ret = %p",
        prog, num_kernels, kernels, num_kernels_ret);

    auto program = icd_downcast(prog);

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
        for (auto& kname : program->kernel_names()) {
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

cl_kernel CLVK_API_CALL clCloneKernel(cl_kernel source_kernel,
                                      cl_int* errcode_ret) {
    TRACE_FUNCTION("source_kernel", (uintptr_t)source_kernel);
    LOG_API_CALL("kernel = %p, errcode_ret = %p", source_kernel, errcode_ret);

    if (!is_valid_kernel(source_kernel)) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_KERNEL;
        }
        return nullptr;
    }

    cl_int err;
    auto kernel = icd_downcast(source_kernel)->clone(&err);

    if (errcode_ret != nullptr) {
        *errcode_ret = err;
    }
    if (err != CL_SUCCESS) {
        return nullptr;
    } else {
        return kernel.release();
    }
}

cl_int CLVK_API_CALL clSetKernelArg(cl_kernel kern, cl_uint arg_index,
                                    size_t arg_size, const void* arg_value) {

    TRACE_FUNCTION("kernel", (uintptr_t)kern, "arg_index", arg_index);
    LOG_API_CALL("kernel = %p, arg_index = %u, arg_size = %zu, arg_value = %p",
                 kern, arg_index, arg_size, arg_value);

    auto kernel = icd_downcast(kern);

    if (!is_valid_kernel(kernel)) {
        return CL_INVALID_KERNEL;
    }

    // TODO CL_INVALID_ARG_VALUE if arg_value specified is not a valid value.
    // TODO CL_INVALID_ARG_SIZE if arg_size does not match the size of the data
    // type for an argument that is not a memory object or if the argument is a
    // memory object and arg_size != sizeof(cl_mem) or if arg_size is zero and
    // the argument is declared with the __local qualifier or if the argument is
    // a sampler and arg_size != sizeof(cl_sampler).
    // TODO CL_INVALID_ARG_VALUE if the argument is an image declared with the
    // read_only qualifier and arg_value refers to an image object created with
    // cl_mem_flags of CL_MEM_WRITE or if the image argument is declared with
    // the write_only qualifier and arg_value refers to an image object created
    // with cl_mem_flags of CL_MEM_READ.

    if (arg_index >= kernel->num_args()) {
        cvk_error_fn("the program has only %u arguments", kernel->num_args());
        return CL_INVALID_ARG_INDEX;
    }

    // CL_INVALID_MEM_OBJECT and CL_INVALID_SAMPLER are handled in
    // cvk_kernel_argument_values::set_arg.

    // With opaque pointers, clspv is unable to infer the type of an unused
    // kernel argument so allow nullptr for its value. It will not have an
    // affect on the kernel's operation.
    if ((arg_value == nullptr) &&
        !((kernel->arg_kind(arg_index) == kernel_argument_kind::local) ||
          (kernel->arg_kind(arg_index) == kernel_argument_kind::unused))) {
        cvk_error_fn("passing a null pointer to clSetKernelArg is only "
                     "supported for local arguments");
        return CL_INVALID_ARG_VALUE;
    }

    return kernel->set_arg(arg_index, arg_size, arg_value);
}

cl_int CLVK_API_CALL clSetKernelExecInfo(cl_kernel kernel,
                                         cl_kernel_exec_info param_name,
                                         size_t param_value_size,
                                         const void* param_value) {
    TRACE_FUNCTION("kernel", (uintptr_t)kernel);
    LOG_API_CALL("kernel = %p, param_name = %x, param_value_size = %zu, "
                 "param_value = %p",
                 kernel, param_name, param_value_size, param_value);
    return CL_INVALID_OPERATION;
}

cl_int CLVK_API_CALL clGetKernelInfo(cl_kernel kern, cl_kernel_info param_name,
                                     size_t param_value_size, void* param_value,
                                     size_t* param_value_size_ret) {
    TRACE_FUNCTION("kernel", (uintptr_t)kern, "param_name", param_name);
    LOG_API_CALL("kernel = %p, param_name = %x, param_value_size = %zu, "
                 "param_value = %p, param_value_size_ret = %p",
                 kern, param_name, param_value_size, param_value,
                 param_value_size_ret);

    cl_int ret = CL_SUCCESS;
    size_t ret_size = 0;
    const void* copy_ptr = nullptr;
    cl_uint val_uint;
    cl_context val_context;
    cl_program val_program;
    api_query_string val_string;

    auto kernel = icd_downcast(kern);

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
    case CL_KERNEL_ATTRIBUTES: {
        val_string = kernel->attributes();
        copy_ptr = val_string.c_str();
        ret_size = val_string.size_with_null();
        break;
    }
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

cl_int CLVK_API_CALL clGetKernelArgInfo(cl_kernel kern, cl_uint arg_index,
                                        cl_kernel_arg_info param_name,
                                        size_t param_value_size,
                                        void* param_value,
                                        size_t* param_value_size_ret) {
    TRACE_FUNCTION("kernel", (uintptr_t)kern, "arg_index", arg_index,
                   "param_name", param_name);
    LOG_API_CALL("kernel = %p, arg_index = %u, param_name = %x, "
                 "param_value_size = %zu, param_value = %p, "
                 "param_value_size_ret = %p",
                 kern, arg_index, param_name, param_value_size, param_value,
                 param_value_size_ret);

    cl_int ret = CL_SUCCESS;
    const void* copy_ptr = nullptr;
    size_t ret_size = 0;
    api_query_string val_string;
    cl_kernel_arg_address_qualifier val_address_qualifier;
    cl_kernel_arg_access_qualifier val_access_qualifier;
    cl_kernel_arg_type_qualifier val_type_qualifier;

    auto kernel = icd_downcast(kern);

    if (!is_valid_kernel(kernel)) {
        return CL_INVALID_KERNEL;
    }

    if (arg_index >= kernel->num_args()) {
        return CL_INVALID_ARG_INDEX;
    }

    if ((param_name != CL_KERNEL_ARG_NAME) &&
        !kernel->has_extended_arg_info(arg_index)) {
        return CL_KERNEL_ARG_INFO_NOT_AVAILABLE;
    }

    switch (param_name) {
    case CL_KERNEL_ARG_NAME:
        val_string = kernel->arg_name(arg_index);
        copy_ptr = val_string.data();
        ret_size = val_string.size_with_null();
        break;
    case CL_KERNEL_ARG_ADDRESS_QUALIFIER:
        val_address_qualifier = kernel->arg_address_qualifier(arg_index);
        copy_ptr = &val_address_qualifier;
        ret_size = sizeof(val_address_qualifier);
        break;
    case CL_KERNEL_ARG_TYPE_NAME:
        val_string = kernel->arg_type_name(arg_index);
        copy_ptr = val_string.data();
        ret_size = val_string.size_with_null();
        break;
    case CL_KERNEL_ARG_ACCESS_QUALIFIER:
        val_access_qualifier = kernel->arg_access_qualifier(arg_index);
        copy_ptr = &val_access_qualifier;
        ret_size = sizeof(val_access_qualifier);
        break;
    case CL_KERNEL_ARG_TYPE_QUALIFIER:
        val_type_qualifier = kernel->arg_type_qualifier(arg_index);
        copy_ptr = &val_type_qualifier;
        ret_size = sizeof(val_type_qualifier);
        break;
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

cl_int CLVK_API_CALL clGetKernelWorkGroupInfo(
    cl_kernel kern, cl_device_id dev, cl_kernel_work_group_info param_name,
    size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    TRACE_FUNCTION("kernel", (uintptr_t)kern, "device", (uintptr_t)dev,
                   "param_name", param_name);
    LOG_API_CALL(
        "kernel = %p, device = %p, param_name = %x, param_value_size = %zu, "
        "param_value = %p, param_value_size_ret = %p",
        kern, dev, param_name, param_value_size, param_value,
        param_value_size_ret);

    cl_int ret = CL_SUCCESS;
    const void* copy_ptr = nullptr;
    size_t val_sizet, ret_size = 0;
    cl_ulong val_ulong;
    std::array<size_t, 3> val_wgs;

    auto device = icd_downcast(dev);
    auto kernel = icd_downcast(kern);

    if (!is_valid_kernel(kernel)) {
        return CL_INVALID_KERNEL;
    }

    switch (param_name) {
    case CL_KERNEL_WORK_GROUP_SIZE:
        val_sizet = kernel->max_work_group_size(device);
        copy_ptr = &val_sizet;
        ret_size = sizeof(val_sizet);
        break;
    case CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE:
        val_sizet = device->preferred_work_group_size_multiple();
        copy_ptr = &val_sizet;
        ret_size = sizeof(val_sizet);
        break;
    case CL_KERNEL_LOCAL_MEM_SIZE:
        val_ulong = kernel->local_mem_size();
        copy_ptr = &val_ulong;
        ret_size = sizeof(val_ulong);
        break;
    case CL_KERNEL_COMPILE_WORK_GROUP_SIZE: {
        auto const& val_wgs_uint = kernel->required_work_group_size();
        val_wgs[0] = val_wgs_uint[0];
        val_wgs[1] = val_wgs_uint[1];
        val_wgs[2] = val_wgs_uint[2];
        copy_ptr = val_wgs.data();
        ret_size = sizeof(val_wgs);
        break;
    }
    case CL_KERNEL_PRIVATE_MEM_SIZE: // TODO
        // Return 0 as it is a lower bound of the private memory size needed by
        // a kernel.
        val_ulong = 0;
        copy_ptr = &val_ulong;
        ret_size = sizeof(val_ulong);
        break;
    case CL_KERNEL_GLOBAL_WORK_SIZE: // TODO
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

cl_int CLVK_API_CALL clGetKernelSubGroupInfo(
    cl_kernel kern, cl_device_id dev, cl_kernel_sub_group_info param_name,
    size_t input_value_size, const void* input_value, size_t param_value_size,
    void* param_value, size_t* param_value_size_ret) {
    TRACE_FUNCTION("kernel", (uintptr_t)kern, "device", (uintptr_t)dev,
                   "param_name", param_name);
    LOG_API_CALL("kernel = %p, device = %p, param_name = %x, input_value_size "
                 "= %zu, input_value = %p, param_value_size = %zu, param_value "
                 "= %p, param_value_size_ret = %p",
                 kern, dev, param_name, input_value_size, input_value,
                 param_value_size, param_value, param_value_size_ret);

    cl_int ret = CL_SUCCESS;
    const void* copy_ptr = nullptr;
    size_t val_sizet, ret_size = 0;
    std::array<size_t, 3> val_lws;

    auto device = icd_downcast(dev);
    auto kernel = icd_downcast(kern);

    if (!is_valid_kernel(kernel)) {
        return CL_INVALID_KERNEL;
    }

    if (!is_valid_device(device)) {
        return CL_INVALID_DEVICE;
    }

    switch (param_name) {
    case CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE:
        val_sizet = kernel->max_sub_group_size_for_ndrange(device);
        copy_ptr = &val_sizet;
        ret_size = sizeof(val_sizet);
        break;
    case CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE: {
        std::array<uint32_t, 3> lws = {1, 1, 1};
        unsigned num_dims = input_value_size / sizeof(size_t);
        if (input_value_size % sizeof(size_t) != 0) {
            ret = CL_INVALID_VALUE;
            break;
        }
        for (unsigned dim = 0; dim < num_dims; dim++) {
            lws[dim] = static_cast<const size_t*>(input_value)[dim];
        }
        val_sizet = kernel->sub_group_count_for_ndrange(device, lws);
        copy_ptr = &val_sizet;
        ret_size = sizeof(val_sizet);
        break;
    }
    case CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT: {
        if (input_value_size % sizeof(size_t) != 0) {
            ret = CL_INVALID_VALUE;
            break;
        }
        auto num_sub_groups = *static_cast<const size_t*>(input_value);
        val_lws =
            kernel->local_size_for_sub_group_count(device, num_sub_groups);
        copy_ptr = &val_lws;
        ret_size = param_value_size;
        break;
    }
    case CL_KERNEL_MAX_NUM_SUB_GROUPS:
        val_sizet = kernel->max_num_sub_groups(device);
        copy_ptr = &val_sizet;
        ret_size = sizeof(val_sizet);
        break;
    case CL_KERNEL_COMPILE_NUM_SUB_GROUPS: // TODO
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

cl_int CLVK_API_CALL clRetainKernel(cl_kernel kernel) {
    TRACE_FUNCTION("kernel", (uintptr_t)kernel);
    LOG_API_CALL("kernel = %p", kernel);

    if (!is_valid_kernel(kernel)) {
        return CL_INVALID_KERNEL;
    }

    icd_downcast(kernel)->retain();

    return CL_SUCCESS;
}

cl_int CLVK_API_CALL clReleaseKernel(cl_kernel kernel) {
    TRACE_FUNCTION("kernel", (uintptr_t)kernel);
    LOG_API_CALL("kernel = %p", kernel);

    if (!is_valid_kernel(kernel)) {
        return CL_INVALID_KERNEL;
    }

    icd_downcast(kernel)->release();

    return CL_SUCCESS;
}

/* Profiling APIs  */
cl_int CLVK_API_CALL clGetEventProfilingInfo(cl_event evt,
                                             cl_profiling_info param_name,
                                             size_t param_value_size,
                                             void* param_value,
                                             size_t* param_value_size_ret) {
    TRACE_FUNCTION("event", (uintptr_t)evt, "param_name", param_name);
    LOG_API_CALL("event = %p, param_name = %x, param_value_size = %zu, "
                 "param_value = %p, param_value_size_ret = %p",
                 evt, param_name, param_value_size, param_value,
                 param_value_size_ret);

    auto event = icd_downcast(evt);

    if (!is_valid_event(event)) {
        return CL_INVALID_EVENT;
    }

    switch (param_name) {
    case CL_PROFILING_COMMAND_QUEUED:
    case CL_PROFILING_COMMAND_SUBMIT:
    case CL_PROFILING_COMMAND_START:
    case CL_PROFILING_COMMAND_END:
    case CL_PROFILING_COMMAND_COMPLETE:
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
        if (param_name == CL_PROFILING_COMMAND_COMPLETE) {
            param_name = CL_PROFILING_COMMAND_END;
        }
        cl_ulong value = event->get_profiling_info(param_name);
        memcpy(param_value, &value, sizeof(cl_ulong));
    }

    return CL_SUCCESS;
}

/* Flush and Finish APIs */
cl_int CLVK_API_CALL clFlush(cl_command_queue command_queue) {
    TRACE_FUNCTION("command_queue", (uintptr_t)command_queue);
    LOG_API_CALL("command_queue = %p", command_queue);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    return icd_downcast(command_queue)->flush();
}

cl_int CLVK_API_CALL clFinish(cl_command_queue command_queue) {
    TRACE_FUNCTION("command_queue", (uintptr_t)command_queue);
    LOG_API_CALL("command_queue = %p", command_queue);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    return icd_downcast(command_queue)->finish();
}

/* Enqueued Commands APIs */

cl_int CLVK_API_CALL clEnqueueReadBuffer(cl_command_queue cq, cl_mem buf,
                                         cl_bool blocking_read, size_t offset,
                                         size_t size, void* ptr,
                                         cl_uint num_events_in_wait_list,
                                         const cl_event* event_wait_list,
                                         cl_event* event) {
    TRACE_FUNCTION("command_queue", (uintptr_t)cq, "buffer", (uintptr_t)buf,
                   "blocking_read", blocking_read, "offset", offset, "size",
                   size, "num_events_in_wait_list", num_events_in_wait_list);
    LOG_API_CALL("command_queue = %p, buffer = %p, blocking = %d, offset = "
                 "%zu, size = %zu, ptr = %p",
                 cq, buf, blocking_read, offset, size, ptr);

    auto buffer = icd_downcast(buf);
    auto command_queue = icd_downcast(cq);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    if (!is_valid_buffer(buffer)) {
        return CL_INVALID_MEM_OBJECT;
    }

    if (buffer->has_any_flag(CL_MEM_HOST_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS)) {
        return CL_INVALID_OPERATION;
    }

    if (!is_valid_event_wait_list(num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    auto cmd = new cvk_command_buffer_host_copy(
        command_queue, CL_COMMAND_READ_BUFFER, static_cast<cvk_buffer*>(buffer),
        ptr, offset, size);

    auto err = command_queue->enqueue_command_with_deps(
        cmd, blocking_read, num_events_in_wait_list, event_wait_list, event);

    return err;
}

cl_int CLVK_API_CALL clEnqueueWriteBuffer(cl_command_queue cq, cl_mem buf,
                                          cl_bool blocking_write, size_t offset,
                                          size_t size, const void* ptr,
                                          cl_uint num_events_in_wait_list,
                                          const cl_event* event_wait_list,
                                          cl_event* event) {
    TRACE_FUNCTION("command_queue", (uintptr_t)cq, "buffer", (uintptr_t)buf,
                   "blocking_write", blocking_write, "offset", offset, "size",
                   size, "num_events_in_wait_list", num_events_in_wait_list);
    LOG_API_CALL("command_queue = %p, buffer = %p, blocking = %d, offset = "
                 "%zu, size = %zu, ptr = %p",
                 cq, buf, blocking_write, offset, size, ptr);

    auto buffer = icd_downcast(buf);
    auto command_queue = icd_downcast(cq);

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

    if (!is_valid_event_wait_list(num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    auto cmd = new cvk_command_buffer_host_copy(
        command_queue, CL_COMMAND_WRITE_BUFFER,
        static_cast<cvk_buffer*>(buffer), ptr, offset, size);

    auto err = command_queue->enqueue_command_with_deps(
        cmd, blocking_write, num_events_in_wait_list, event_wait_list, event);

    return err;
}

cl_int CLVK_API_CALL clEnqueueReadBufferRect(
    cl_command_queue cq, cl_mem buf, cl_bool blocking_read,
    const size_t* buffer_origin, const size_t* host_origin,
    const size_t* region, size_t buffer_row_pitch, size_t buffer_slice_pitch,
    size_t host_row_pitch, size_t host_slice_pitch, void* ptr,
    cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
    cl_event* event) {
    TRACE_FUNCTION("command_queue", (uintptr_t)cq, "buffer", (uintptr_t)buf,
                   "blocking_read", blocking_read, "num_events_in_wait_list",
                   num_events_in_wait_list);
    LOG_API_CALL("command_queue = %p, buffer = %p, blocking = %d", cq, buf,
                 blocking_read);
    LOG_API_CALL("buffer_origin = {%zu,%zu,%zu}, host_origin = {%zu,%zu,%zu}, "
                 "region = {%zu,%zu,%zu}",
                 buffer_origin[0], buffer_origin[1], buffer_origin[2],
                 host_origin[0], host_origin[1], host_origin[2], region[0],
                 region[1], region[2]);
    LOG_API_CALL("buffer_row_pitch = %zu, buffer_slice_pitch = %zu,"
                 "host_row_pitch = %zu, host_slice_pitch = %zu",
                 buffer_row_pitch, buffer_slice_pitch, host_row_pitch,
                 host_slice_pitch);
    LOG_API_CALL("ptr = %p, num_events = %u, event_wait_list = %p, event = %p",
                 ptr, num_events_in_wait_list, event_wait_list, event);

    auto command_queue = icd_downcast(cq);
    auto buffer = static_cast<cvk_buffer*>(buf);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    if (!is_valid_buffer(buffer)) {
        return CL_INVALID_MEM_OBJECT;
    }

    if (buffer->has_any_flag(CL_MEM_HOST_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS)) {
        return CL_INVALID_OPERATION;
    }

    if (!is_valid_event_wait_list(num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    auto cmd = new cvk_command_copy_host_buffer_rect(
        command_queue, CL_COMMAND_READ_BUFFER_RECT, buffer, ptr, host_origin,
        buffer_origin, region, host_row_pitch, host_slice_pitch,
        buffer_row_pitch, buffer_slice_pitch);

    auto err = command_queue->enqueue_command_with_deps(
        cmd, blocking_read, num_events_in_wait_list, event_wait_list, event);

    return err;
}

cl_int CLVK_API_CALL clEnqueueWriteBufferRect(
    cl_command_queue cq, cl_mem buf, cl_bool blocking_write,
    const size_t* buffer_origin, const size_t* host_origin,
    const size_t* region, size_t buffer_row_pitch, size_t buffer_slice_pitch,
    size_t host_row_pitch, size_t host_slice_pitch, const void* ptr,
    cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
    cl_event* event) {
    TRACE_FUNCTION("command_queue", (uintptr_t)cq, "buffer", (uintptr_t)buf,
                   "blocking_write", blocking_write, "num_events_in_wait_list",
                   num_events_in_wait_list);
    LOG_API_CALL("command_queue = %p, buffer = %p, blocking = %d", cq, buf,
                 blocking_write);
    LOG_API_CALL("buffer_origin = {%zu,%zu,%zu}, host_origin = {%zu,%zu,%zu}, "
                 "region = {%zu,%zu,%zu}",
                 buffer_origin[0], buffer_origin[1], buffer_origin[2],
                 host_origin[0], host_origin[1], host_origin[2], region[0],
                 region[1], region[2]);
    LOG_API_CALL("buffer_row_pitch = %zu, buffer_slice_pitch = %zu, "
                 "host_row_pitch = %zu, host_slice_pitch = %zu",
                 buffer_row_pitch, buffer_slice_pitch, host_row_pitch,
                 host_slice_pitch);
    LOG_API_CALL("ptr = %p, num_events = %u, event_wait_list = %p, event = %p",
                 ptr, num_events_in_wait_list, event_wait_list, event);

    auto command_queue = icd_downcast(cq);
    auto buffer = static_cast<cvk_buffer*>(buf);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    if (!is_valid_buffer(buffer)) {
        return CL_INVALID_MEM_OBJECT;
    }

    if (buffer->has_any_flag(CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_NO_ACCESS)) {
        return CL_INVALID_OPERATION;
    }

    if (!is_valid_event_wait_list(num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    auto cmd = new cvk_command_copy_host_buffer_rect(
        command_queue, CL_COMMAND_WRITE_BUFFER_RECT, buffer,
        const_cast<void*>(ptr), host_origin, buffer_origin, region,
        host_row_pitch, host_slice_pitch, buffer_row_pitch, buffer_slice_pitch);

    auto err = command_queue->enqueue_command_with_deps(
        cmd, blocking_write, num_events_in_wait_list, event_wait_list, event);

    return err;
}

cl_int CLVK_API_CALL clEnqueueFillBuffer(
    cl_command_queue cq, cl_mem buf, const void* pattern, size_t pattern_size,
    size_t offset, size_t size, cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list, cl_event* event) {
    TRACE_FUNCTION("command_queue", (uintptr_t)cq, "buffer", (uintptr_t)buf,
                   "offset", offset, "size", size, "num_events_in_wait_list",
                   num_events_in_wait_list);
    LOG_API_CALL(
        "command_queue = %p, buffer = %p, pattern = %p, pattern_size = %zu,"
        "offset = %zu, size = %zu, num_events = %u, event_wait_list = %p, "
        "event = %p",
        cq, buf, pattern, pattern_size, offset, size, num_events_in_wait_list,
        event_wait_list, event);

    auto buffer = icd_downcast(buf);
    auto command_queue = icd_downcast(cq);

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
    size_t valid_pattern_sizes[] = {1, 2, 4, 8, 16, 32, 64, 128};
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

    if (!is_valid_event_wait_list(num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    // TODO check sub-buffer alignment

    auto cmd = new cvk_command_fill_buffer(
        command_queue, static_cast<cvk_buffer*>(buffer), offset, size, pattern,
        pattern_size, CL_COMMAND_FILL_BUFFER);

    return command_queue->enqueue_command_with_deps(
        cmd, num_events_in_wait_list, event_wait_list, event);
}

cl_int CLVK_API_CALL clEnqueueCopyBuffer(cl_command_queue cq, cl_mem srcbuf,
                                         cl_mem dstbuf, size_t src_offset,
                                         size_t dst_offset, size_t size,
                                         cl_uint num_events_in_wait_list,
                                         const cl_event* event_wait_list,
                                         cl_event* event) {
    TRACE_FUNCTION("command_queue", (uintptr_t)cq, "src_buffer",
                   (uintptr_t)srcbuf, "dst_buffer", (uintptr_t)dstbuf,
                   "src_offset", src_offset, "dst_offset", dst_offset, "size",
                   size, "num_events_in_wait_list", num_events_in_wait_list);
    LOG_API_CALL("command_queue = %p, src_buffer = %p, dst_buffer = %p, "
                 "src_offset = %zu,"
                 "dst_offset = %zu, size = %zu, num_events = %u, "
                 "event_wait_list = %p, event = %p",
                 cq, srcbuf, dstbuf, src_offset, dst_offset, size,
                 num_events_in_wait_list, event_wait_list, event);

    auto command_queue = icd_downcast(cq);
    auto src_buffer = icd_downcast(srcbuf);
    auto dst_buffer = icd_downcast(dstbuf);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    if (!is_same_context(cq, srcbuf) || !is_same_context(cq, dstbuf)) {
        return CL_INVALID_CONTEXT;
    }

    if (!is_same_context(cq, num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_CONTEXT;
    }

    if (!is_valid_buffer(src_buffer) || !is_valid_buffer(dst_buffer)) {
        return CL_INVALID_MEM_OBJECT;
    }

    if (!is_valid_event_wait_list(num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    auto cmd = new cvk_command_copy_buffer(
        command_queue, CL_COMMAND_COPY_BUFFER,
        static_cast<cvk_buffer*>(src_buffer),
        static_cast<cvk_buffer*>(dst_buffer), src_offset, dst_offset, size);

    return command_queue->enqueue_command_with_deps(
        cmd, num_events_in_wait_list, event_wait_list, event);
}

cl_int CLVK_API_CALL clEnqueueCopyBufferRect(
    cl_command_queue cq, cl_mem src_buffer, cl_mem dst_buffer,
    const size_t* src_origin, const size_t* dst_origin, const size_t* region,
    size_t src_row_pitch, size_t src_slice_pitch, size_t dst_row_pitch,
    size_t dst_slice_pitch, cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list, cl_event* event) {
    TRACE_FUNCTION("command_queue", (uintptr_t)cq, "src_buffer",
                   (uintptr_t)src_buffer, "dst_buffer", (uintptr_t)dst_buffer,
                   "num_events_in_wait_list", num_events_in_wait_list);
    LOG_API_CALL("command_queue = %p, src_buffer = %p, dst_buffer = %p, "
                 "src_origin = {%zu,%zu,%zu}, dst_origin = {%zu,%zu,%zu}, "
                 "region = {%zu,%zu,%zu}, src_row_pitch = %zu, "
                 "src_slice_pitch = %zu, dst_row_pitch = %zu, "
                 "dst_slice_pitch = %zu, num_events_in_wait_list = %u, "
                 "event_wait_list = %p, event = %p",
                 cq, src_buffer, dst_buffer, src_origin[0], src_origin[1],
                 src_origin[2], dst_origin[0], dst_origin[1], dst_origin[2],
                 region[0], region[1], region[2], src_row_pitch,
                 src_slice_pitch, dst_row_pitch, dst_slice_pitch,
                 num_events_in_wait_list, event_wait_list, event);

    auto command_queue = icd_downcast(cq);

    // TODO CL_INVALID_COMMAND_QUEUE if command_queue is not a valid
    // command-queue.

    if (!is_valid_buffer(src_buffer) || !is_valid_buffer(dst_buffer)) {
        return CL_INVALID_MEM_OBJECT;
    }
    // TODO CL_INVALID_VALUE if (src_offset, region) or (dst_offset, region)
    // require accessing elements outside the src_buffer and dst_buffer objects
    // respectively.
    // TODO CL_INVALID_VALUE if any region array element is 0.
    // TODO CL_INVALID_VALUE if src_row_pitch is not 0 and is less than
    // region[0].
    // TODO CL_INVALID_VALUE if dst_row_pitch is not 0 and is less than
    // region[0].
    // TODO CL_INVALID_VALUE if src_slice_pitch is not 0 and is less than
    // region[1] * src_row_pitch or if src_slice_pitch is not 0 and is not a
    // multiple of src_row_pitch.
    // TODO CL_INVALID_VALUE if dst_slice_pitch is not 0 and is less than
    // region[1] * dst_row_pitch or if dst_slice_pitch is not 0 and is not a
    // multiple of dst_row_pitch.
    // TODO CL_INVALID_VALUE if src_buffer and dst_buffer are the same buffer
    // object and src_slice_pitch is not equal to dst_slice_pitch and
    // src_row_pitch is not equal to dst_row_pitch.
    //
    if (!is_valid_event_wait_list(num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    if (!is_same_context(command_queue, src_buffer) ||
        !is_same_context(command_queue, dst_buffer) ||
        !is_same_context(command_queue, num_events_in_wait_list,
                         event_wait_list)) {
        return CL_INVALID_CONTEXT;
    }
    // TODO CL_MEM_COPY_OVERLAP if src_buffer and dst_buffer are the same buffer
    // object and the source and destination regions overlap or if src_buffer
    // and dst_buffer are different sub-buffers of the same associated buffer
    // object and they overlap. Refer to Appendix E in the OpenCL specification
    // for details on how to determine if source and destination regions
    // overlap.
    // TODO CL_MISALIGNED_SUB_BUFFER_OFFSET if src_buffer is a sub-buffer object
    // and offset specified when the sub-buffer object is created is not aligned
    // to CL_DEVICE_MEM_BASE_ADDR_ALIGN value for device associated with queue.
    // TODO CL_MISALIGNED_SUB_BUFFER_OFFSET if dst_buffer is a sub-buffer object
    // and offset specified when the sub-buffer object is created is not aligned
    // to CL_DEVICE_MEM_BASE_ADDR_ALIGN value for device associated with queue.
    // TODO CL_MEM_OBJECT_ALLOCATION_FAILURE if there is a failure to allocate
    // memory for data store associated with src_buffer or dst_buffer.
    // TODO CL_OUT_OF_RESOURCES if there is a failure to allocate resources
    // required by the OpenCL implementation on the device.
    // TODO CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources
    // required by the OpenCL implementation on the host.
    //
    auto srcbuf = static_cast<cvk_buffer*>(src_buffer);
    auto dstbuf = static_cast<cvk_buffer*>(dst_buffer);
    auto cmd = new cvk_command_copy_buffer_rect(
        command_queue, srcbuf, dstbuf, src_origin, dst_origin, region,
        src_row_pitch, src_slice_pitch, dst_row_pitch, dst_slice_pitch);
    return command_queue->enqueue_command_with_deps(
        cmd, num_events_in_wait_list, event_wait_list, event);
}

void* cvk_enqueue_map_buffer(cvk_command_queue* cq, cvk_buffer* buffer,
                             cl_bool blocking_map, size_t offset, size_t size,
                             cl_map_flags map_flags,
                             cl_uint num_events_in_wait_list,
                             const cl_event* event_wait_list, cl_event* event,
                             cl_int* errcode_ret, cl_command_type type,
                             cvk_image* image = nullptr) {
    auto cmd = new cvk_command_map_buffer(cq, buffer, offset, size, map_flags,
                                          type, image);

    void* map_ptr;
    cl_int err = cmd->build(&map_ptr);

    // FIXME This error cannot occur for objects created with
    // CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR.
    if (err != CL_SUCCESS) {
        *errcode_ret = CL_MAP_FAILURE;
        return nullptr;
    }

    err = cq->enqueue_command_with_deps(
        cmd, blocking_map, num_events_in_wait_list, event_wait_list, event);

    *errcode_ret = err;

    return map_ptr;
}

void* CLVK_API_CALL clEnqueueMapBuffer(cl_command_queue cq, cl_mem buf,
                                       cl_bool blocking_map,
                                       cl_map_flags map_flags, size_t offset,
                                       size_t size,
                                       cl_uint num_events_in_wait_list,
                                       const cl_event* event_wait_list,
                                       cl_event* event, cl_int* errcode_ret) {
    TRACE_FUNCTION("command_queue", (uintptr_t)cq, "buffer", (uintptr_t)buf,
                   "blocking_map", blocking_map, "map_flags", map_flags,
                   "offset", offset, "size", size, "num_events_in_wait_list",
                   num_events_in_wait_list);
    LOG_API_CALL("command_queue = %p, buffer = %p, offset = %zu, size = %zu",
                 cq, buf, offset, size);

    auto command_queue = icd_downcast(cq);
    auto buffer = static_cast<cvk_buffer*>(buf);

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

    if (!is_valid_event_wait_list(num_events_in_wait_list, event_wait_list)) {
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

    // TODO CL_MISALIGNED_SUB_BUFFER_OFFSET if buffer is a sub-buffer object and
    // offset specified when the sub-buffer object is created is not aligned to
    // CL_DEVICE_MEM_BASE_ADDR_ALIGN value for device associated with queue.

    if ((map_flags & CL_MAP_READ) &&
        (buffer->has_any_flag(CL_MEM_HOST_WRITE_ONLY |
                              CL_MEM_HOST_NO_ACCESS))) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_OPERATION;
        }
        return nullptr;
    }

    if (((map_flags & CL_MAP_WRITE) ||
         (map_flags & CL_MAP_WRITE_INVALIDATE_REGION)) &&
        (buffer->has_any_flag(CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_NO_ACCESS))) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_OPERATION;
        }
        return nullptr;
    }

    cl_int err;
    auto map_ptr = cvk_enqueue_map_buffer(
        command_queue, buffer, blocking_map, offset, size, map_flags,
        num_events_in_wait_list, event_wait_list, event, &err,
        CL_COMMAND_MAP_BUFFER);

    if (errcode_ret != nullptr) {
        *errcode_ret = err;
    }

    return map_ptr;
}

cl_int CLVK_API_CALL clEnqueueUnmapMemObject(cl_command_queue cq, cl_mem mem,
                                             void* mapped_ptr,
                                             cl_uint num_events_in_wait_list,
                                             const cl_event* event_wait_list,
                                             cl_event* event) {
    TRACE_FUNCTION("command_queue", (uintptr_t)cq, "memobj", (uintptr_t)mem,
                   "num_events_in_wait_list", num_events_in_wait_list);
    LOG_API_CALL("command_queue = %p, memobj = %p, mapped_ptr = %p", cq, mem,
                 mapped_ptr);

    auto command_queue = icd_downcast(cq);
    auto memobj = icd_downcast(mem);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    if (!is_valid_mem_object(memobj)) {
        return CL_INVALID_MEM_OBJECT;
    }

    cvk_command* cmd;

    if (memobj->is_image_type()) {
        auto image = static_cast<cvk_image*>(memobj);
        if (image->is_backed_by_buffer_view()) {
            auto buffer = static_cast<cvk_buffer*>(image->buffer());
            cmd =
                new cvk_command_unmap_buffer(command_queue, buffer, mapped_ptr);
        } else {
            auto cmd_unmap = std::make_unique<cvk_command_unmap_image>(
                command_queue, image, mapped_ptr, true);

            auto err = cmd_unmap->build();
            if (err != CL_SUCCESS) {
                return err;
            }
            cmd = cmd_unmap.release();
        }
    } else {
        auto buffer = static_cast<cvk_buffer*>(memobj);
        cmd = new cvk_command_unmap_buffer(command_queue, buffer, mapped_ptr);
    }

    return command_queue->enqueue_command_with_deps(
        cmd, num_events_in_wait_list, event_wait_list, event);
}

cl_int cvk_enqueue_ndrange_kernel(cvk_command_queue* command_queue,
                                  cvk_kernel* kernel, uint32_t dims,
                                  const cvk_ndrange& ndrange,
                                  cl_uint num_events_in_wait_list,
                                  const cl_event* event_wait_list,
                                  cl_event* event) {

    // TODO check that it's a host command queue
    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    if (!is_valid_kernel(kernel)) {
        return CL_INVALID_KERNEL;
    }

    if (!is_valid_event_wait_list(num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    if (!is_same_context(command_queue, kernel)) {
        return CL_INVALID_CONTEXT;
    }

    if (!is_same_context(command_queue, num_events_in_wait_list,
                         event_wait_list)) {
        return CL_INVALID_CONTEXT;
    }

    auto device = command_queue->device();

    if ((dims < 1) || (dims > device->max_work_item_dimensions())) {
        return CL_INVALID_WORK_DIMENSION;
    }

    auto program = kernel->program();

    if (program->binary_type(device) != CL_PROGRAM_BINARY_TYPE_EXECUTABLE) {
        return CL_INVALID_PROGRAM_EXECUTABLE;
    }

    if (!kernel->args_valid()) {
        return CL_INVALID_KERNEL_ARGS;
    }

    // TODO CL_INVALID_GLOBAL_WORK_SIZE if any of the values specified in
    // global_work_size[0], … global_work_size[work_dim - 1] exceed the maximum
    // value representable by size_t on the device on which the kernel-instance
    // will be enqueued.
    // TODO CL_INVALID_GLOBAL_OFFSET if the value specified in global_work_size
    // + the corresponding values in global_work_offset for any dimensions is
    // greater than the maximum value representable by size t on the device on
    // which the kernel-instance will be enqueued.
    // TODO CL_INVALID_WORK_GROUP_SIZE if local_work_size is specified and is
    // not consistent with the required number of sub-groups for kernel in the
    // program source.
    // TODO CL_INVALID_WORK_GROUP_SIZE if local_work_size is specified and the
    // total number of work-items in the work-group computed as
    // local_work_size[0] × … local_work_size[work_dim - 1] is greater than the
    // value specified by CL_KERNEL_WORK_GROUP_SIZE in the Kernel Object Device
    // Queries table.
    // TODO CL_INVALID_WORK_GROUP_SIZE if the program was compiled with
    // cl-uniform-work-group-size and the number of work-items specified by
    // global_work_size is not evenly divisible by size of work-group given by
    // local_work_size or by the required work-group size specified in the
    // kernel source.
    // TODO CL_INVALID_WORK_ITEM_SIZE if the number of work-items specified in
    // any of local_work_size[0], … local_work_size[work_dim - 1] is greater
    // than the corresponding values specified by
    // CL_DEVICE_MAX_WORK_ITEM_SIZES[0], …,
    // CL_DEVICE_MAX_WORK_ITEM_SIZES[work_dim - 1].
    // TODO CL_MISALIGNED_SUB_BUFFER_OFFSET if a sub-buffer object is specified
    // as the value for an argument that is a buffer object and the offset
    // specified when the sub-buffer object is created is not aligned to
    // CL_DEVICE_MEM_BASE_ADDR_ALIGN value for device associated with queue.
    // TODO CL_INVALID_IMAGE_SIZE if an image object is specified as an argument
    // value and the image dimensions (image width, height, specified or compute
    // row and/or slice pitch) are not supported by device associated with
    // queue.
    // TODO CL_IMAGE_FORMAT_NOT_SUPPORTED if an image object is specified as an
    // argument value and the image format (image channel order and data type)
    // is not supported by device associated with queue.
    // TODO CL_OUT_OF_RESOURCES if there is a failure to queue the execution
    // instance of kernel on the command-queue because of insufficient resources
    // needed to execute the kernel. For example, the explicitly specified
    // local_work_size causes a failure to execute the kernel because of
    // insufficient resources such as registers or local memory. Another example
    // would be the number of read-only image args used in kernel exceed the
    // CL_DEVICE_MAX_READ_IMAGE_ARGS value for device or the number of
    // write-only and read-write image args used in kernel exceed the
    // CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS value for device or the number of
    // samplers used in kernel exceed CL_DEVICE_MAX_SAMPLERS for device.
    // TODO CL_MEM_OBJECT_ALLOCATION_FAILURE if there is a failure to allocate
    // memory for data store associated with image or buffer objects specified
    // as arguments to kernel.
    // TODO CL_INVALID_OPERATION if SVM pointers are passed as arguments to a
    // kernel and the device does not support SVM or if system pointers are
    // passed as arguments to a kernel and/or stored inside SVM allocations
    // passed as kernel arguments and the device does not support fine grain
    // system SVM allocations.

    // Check work-group size matches the required size if specified
    auto reqd_work_group_size = kernel->required_work_group_size();
    if (reqd_work_group_size[0] != 0) {
        if (reqd_work_group_size != ndrange.lws) {
            return CL_INVALID_WORK_GROUP_SIZE;
        }
    }

    // Check uniformity of the NDRange if needed
    if (!command_queue->device()->supports_non_uniform_workgroup()) {
        if (!ndrange.is_uniform()) {
            return CL_INVALID_WORK_GROUP_SIZE;
        }
    }

    auto cmd = new cvk_command_kernel(command_queue, kernel, dims, ndrange);

    return command_queue->enqueue_command_with_deps(
        cmd, num_events_in_wait_list, event_wait_list, event);
}

cl_int CLVK_API_CALL clEnqueueTask(cl_command_queue command_queue,
                                   cl_kernel kernel,
                                   cl_uint num_events_in_wait_list,
                                   const cl_event* event_wait_list,
                                   cl_event* event) {
    TRACE_FUNCTION("command_queue", (uintptr_t)command_queue, "kernel",
                   (uintptr_t)kernel, "num_events_in_wait_list",
                   (uintptr_t)num_events_in_wait_list);
    LOG_API_CALL(
        "command_queue = %p, kernel = %p, num_events_in_wait_list = %d,"
        " event_wait_list = %p, event = %p",
        command_queue, kernel, num_events_in_wait_list, event_wait_list, event);

    cvk_ndrange ndrange;
    ndrange.gws = {1, 1, 1};

    return cvk_enqueue_ndrange_kernel(
        icd_downcast(command_queue), icd_downcast(kernel), 1, ndrange,
        num_events_in_wait_list, event_wait_list, event);
}

cl_int CLVK_API_CALL clEnqueueNDRangeKernel(
    cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
    const size_t* global_work_offset, const size_t* global_work_size,
    const size_t* local_work_size, cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list, cl_event* event) {

    TRACE_FUNCTION("command_queue", (uintptr_t)command_queue, "kernel",
                   (uintptr_t)kernel, "num_events_in_wait_list",
                   (uintptr_t)num_events_in_wait_list);
    LOG_API_CALL(
        "command_queue = %p, kernel = %p, work_dim = %u, "
        "num_events_in_wait_list = %u, event_wait_list = %p, event = %p",
        command_queue, kernel, work_dim, num_events_in_wait_list,
        event_wait_list, event);

    cvk_ndrange ndrange(work_dim, global_work_offset, global_work_size,
                        local_work_size);

    // Try to pick a sensible work-group size if the user didn't specify one.
    if (local_work_size == nullptr) {
        icd_downcast(command_queue)
            ->device()
            ->select_work_group_size(ndrange.gws, ndrange.lws);
        cvk_info_fn("selected local work size: {%u,%u,%u}", ndrange.lws[0],
                    ndrange.lws[1], ndrange.lws[2]);
    }

    LOG_API_CALL("goff = {%u,%u,%u}", ndrange.offset[0], ndrange.offset[1],
                 ndrange.offset[2]);
    LOG_API_CALL("gws = {%u,%u,%u}", ndrange.gws[0], ndrange.gws[1],
                 ndrange.gws[2]);
    LOG_API_CALL("lws = {%u,%u,%u}", ndrange.lws[0], ndrange.lws[1],
                 ndrange.lws[2]);

    return cvk_enqueue_ndrange_kernel(
        icd_downcast(command_queue), icd_downcast(kernel), work_dim, ndrange,
        num_events_in_wait_list, event_wait_list, event);
}

cl_int CLVK_API_CALL clEnqueueNativeKernel(
    cl_command_queue command_queue, void(CL_CALLBACK* user_func)(void*),
    void* args, size_t cb_args, cl_uint num_mem_objects, const cl_mem* mem_list,
    const void** args_mem_loc, cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list, cl_event* event) {
    TRACE_FUNCTION("command_queue", (uintptr_t)command_queue,
                   "num_events_in_wait_list",
                   (uintptr_t)num_events_in_wait_list);
    LOG_API_CALL(
        "command_queue = %p, user_func = %p, args = %p, cb_args = %zu, "
        "num_mem_objects = %u, mem_list = %p, args_mem_loc = %p, "
        "num_events_in_wait_list = %u, event_wait_list = %p, event = %p",
        command_queue, user_func, args, cb_args, num_mem_objects, mem_list,
        args_mem_loc, num_events_in_wait_list, event_wait_list, event);

    return CL_INVALID_OPERATION;
}

cl_sampler cvk_create_sampler(cl_context context, cl_bool normalized_coords,
                              cl_addressing_mode addressing_mode,
                              cl_filter_mode filter_mode,
                              std::vector<cl_sampler_properties>&& properties,
                              cl_int* errcode_ret) {

    if (!is_valid_context(context)) {
        *errcode_ret = CL_INVALID_CONTEXT;
        return nullptr;
    }

    auto ctx = icd_downcast(context);

    if (!ctx->device()->supports_images()) {
        *errcode_ret = CL_INVALID_OPERATION;
        return nullptr;
    }

    auto sampler = cvk_sampler::create(ctx, normalized_coords, addressing_mode,
                                       filter_mode, std::move(properties));

    if (sampler == nullptr) {
        *errcode_ret = CL_OUT_OF_RESOURCES;
    } else {
        *errcode_ret = CL_SUCCESS;
    }

    return sampler;
}

cl_sampler CLVK_API_CALL clCreateSampler(cl_context context,
                                         cl_bool normalized_coords,
                                         cl_addressing_mode addressing_mode,
                                         cl_filter_mode filter_mode,
                                         cl_int* errcode_ret) {
    TRACE_FUNCTION("context", (uintptr_t)context, "normalized_coords",
                   normalized_coords, "addressing_mode", addressing_mode,
                   "filter_mode", filter_mode);
    LOG_API_CALL("context = %p, normalized_coords = %d, addressing_mode = %d, "
                 "filter_mode = %d, errcode_ret = %p",
                 context, normalized_coords, addressing_mode, filter_mode,
                 errcode_ret);

    std::vector<cl_sampler_properties> properties;

    cl_int err;
    auto sampler =
        cvk_create_sampler(context, normalized_coords, addressing_mode,
                           filter_mode, std::move(properties), &err);

    if (errcode_ret != nullptr) {
        *errcode_ret = err;
    }

    return sampler;
}

cl_sampler CLVK_API_CALL clCreateSamplerWithProperties(
    cl_context context, const cl_sampler_properties* sampler_properties,
    cl_int* errcode_ret) {

    TRACE_FUNCTION("context", (uintptr_t)context);
    LOG_API_CALL("context = %p, sampler_properties = %p, errcode_ret = %p",
                 context, sampler_properties, errcode_ret);

    cl_bool normalized_coords = CL_TRUE;
    cl_addressing_mode addressing_mode = CL_ADDRESS_CLAMP;
    cl_filter_mode filter_mode = CL_FILTER_NEAREST;

    std::vector<cl_sampler_properties> properties;

    if (sampler_properties) {
        while (*sampler_properties) {
            auto key = *sampler_properties;
            auto value = *(sampler_properties + 1);

            switch (key) {
            case CL_SAMPLER_NORMALIZED_COORDS:
                normalized_coords = value;
                break;
            case CL_SAMPLER_ADDRESSING_MODE:
                addressing_mode = value;
                break;
            case CL_SAMPLER_FILTER_MODE:
                filter_mode = value;
                break;
            default:
                if (errcode_ret != nullptr) {
                    *errcode_ret = CL_INVALID_VALUE;
                }
                return nullptr;
            }

            properties.push_back(key);
            properties.push_back(value);

            sampler_properties += 2;
        }
        properties.push_back(0);
    }

    cl_int err;
    auto sampler =
        cvk_create_sampler(context, normalized_coords, addressing_mode,
                           filter_mode, std::move(properties), &err);

    if (errcode_ret != nullptr) {
        *errcode_ret = err;
    }

    return sampler;
}

cl_int CLVK_API_CALL clRetainSampler(cl_sampler sampler) {
    TRACE_FUNCTION("sampler", (uintptr_t)sampler);
    LOG_API_CALL("sampler = %p", sampler);

    if (!is_valid_sampler(sampler)) {
        return CL_INVALID_SAMPLER;
    }

    icd_downcast(sampler)->retain();

    return CL_SUCCESS;
}

cl_int CLVK_API_CALL clReleaseSampler(cl_sampler sampler) {
    TRACE_FUNCTION("sampler", (uintptr_t)sampler);
    LOG_API_CALL("sampler = %p", sampler);

    if (!is_valid_sampler(sampler)) {
        return CL_INVALID_SAMPLER;
    }

    icd_downcast(sampler)->release();

    return CL_SUCCESS;
}

cl_int CLVK_API_CALL clGetSamplerInfo(cl_sampler samp,
                                      cl_sampler_info param_name,
                                      size_t param_value_size,
                                      void* param_value,
                                      size_t* param_value_size_ret) {
    TRACE_FUNCTION("sampler", (uintptr_t)samp, "param_name", param_name);
    LOG_API_CALL("sampler = %p, param_name = %d, param_value_size = %zu, "
                 "param_value = %p, param_value_size_ret = %p",
                 samp, param_name, param_value_size, param_value,
                 param_value_size_ret);

    cl_int ret = CL_SUCCESS;
    size_t ret_size = 0;
    const void* copy_ptr = nullptr;
    cl_uint val_uint;
    cl_context val_context;
    cl_bool val_bool;
    cl_addressing_mode val_addressing_mode;
    cl_filter_mode val_filter_mode;

    auto sampler = icd_downcast(samp);

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
    case CL_SAMPLER_PROPERTIES:
        copy_ptr = sampler->properties().data();
        ret_size = sampler->properties().size() * sizeof(cl_sampler_properties);
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

cl_mem cvk_create_image(cl_context context, cl_mem_flags flags,
                        const cl_image_format* image_format,
                        const cl_image_desc* image_desc, void* host_ptr,
                        std::vector<cl_mem_properties>&& properties,
                        cl_int* errcode_ret) {
    if (!is_valid_context(context)) {
        *errcode_ret = CL_INVALID_CONTEXT;
        return nullptr;
    }
    // TODO CL_INVALID_VALUE if values specified in flags are not valid.
    // TODO CL_INVALID_IMAGE_FORMAT_DESCRIPTOR if values specified in
    // image_format are not valid or if image_format is NULL.
    // TODO CL_INVALID_IMAGE_DESCRIPTOR if values specified in image_desc are
    // not valid or if image_desc is NULL.
    // TODO CL_INVALID_IMAGE_SIZE if image dimensions specified in image_desc
    // exceed the minimum maximum image dimensions described in the table of
    // allowed values for param_name for clGetDeviceInfo for all devices in
    // context.
    // TODO CL_INVALID_HOST_PTR if host_ptr in image_desc is NULL and
    // CL_MEM_USE_HOST_PTR or CL_MEM_COPY_HOST_PTR are set in flags or if
    // host_ptr is not NULL but CL_MEM_COPY_HOST_PTR or CL_MEM_USE_HOST_PTR are
    // not set in flags.
    // TODO CL_INVALID_VALUE if a 1D image buffer is being created and the
    // buffer object was created with CL_MEM_WRITE_ONLY and flags specifies
    // CL_MEM_READ_WRITE or CL_MEM_READ_ONLY, or if the buffer object was
    // created with CL_MEM_READ_ONLY and flags specifies CL_MEM_READ_WRITE or
    // CL_MEM_WRITE_ONLY, or if flags specifies CL_MEM_USE_HOST_PTR or
    // CL_MEM_ALLOC_HOST_PTR or CL_MEM_COPY_HOST_PTR.
    // TODO CL_INVALID_VALUE if a 1D image buffer is being created and the
    // buffer object was created with CL_MEM_HOST_WRITE_ONLY and flags specifies
    // CL_MEM_HOST_READ_ONLY, or if the buffer object was created with
    // CL_MEM_HOST_READ_ONLY and flags specifies CL_MEM_HOST_WRITE_ONLY, or if
    // the buffer object was created with CL_MEM_HOST_NO_ACCESS and flags
    // specifies CL_MEM_HOST_READ_ONLY or CL_MEM_HOST_WRITE_ONLY.
    // TODO CL_IMAGE_FORMAT_NOT_SUPPORTED if the image_format is not supported.
    // TODO CL_MEM_OBJECT_ALLOCATION_FAILURE if there is a failure to allocate
    // memory for image object.

    // TODO support creating 2D images from buffers
    if ((image_desc->image_type == CL_MEM_OBJECT_IMAGE2D) &&
        (image_desc->mem_object != nullptr)) {
        *errcode_ret = CL_INVALID_OPERATION;
        return nullptr;
    }

    auto image =
        cvk_image::create(icd_downcast(context), flags, image_desc,
                          image_format, host_ptr, std::move(properties));

    *errcode_ret = (image != nullptr)
                       ? CL_SUCCESS
                       : CL_OUT_OF_RESOURCES; // FIXME do this properly

    return image;
}

cl_mem cvk_create_image(cl_context context, cl_mem_flags flags,
                        const cl_image_format* image_format,
                        const cl_image_desc* image_desc, void* host_ptr,
                        cl_int* errcode_ret) {
    std::vector<cl_mem_properties> properties;
    return cvk_create_image(context, flags, image_format, image_desc, host_ptr,
                            std::move(properties), errcode_ret);
}

cl_mem CLVK_API_CALL clCreateImage(cl_context context, cl_mem_flags flags,
                                   const cl_image_format* image_format,
                                   const cl_image_desc* image_desc,
                                   void* host_ptr, cl_int* errcode_ret) {
    TRACE_FUNCTION("context", (uintptr_t)context, "flags", flags);
    LOG_API_CALL(
        "context = %p, flags = %lu, image_format = %p, image_desc = %p,"
        " host_ptr = %p, errcode_ret = %p",
        context, flags, image_format, image_desc, host_ptr, errcode_ret);

    cl_int err;
    auto image = cvk_create_image(context, flags, image_format, image_desc,
                                  host_ptr, &err);

    if (errcode_ret != nullptr) {
        *errcode_ret = err;
    }

    return image;
}

cl_mem CLVK_API_CALL clCreateImageWithProperties(
    cl_context context, const cl_mem_properties* properties, cl_mem_flags flags,
    const cl_image_format* image_format, const cl_image_desc* image_desc,
    void* host_ptr, cl_int* errcode_ret) {

    TRACE_FUNCTION("context", (uintptr_t)context, "flags", flags);
    LOG_API_CALL("context = %p, properties = %p, flags = %lx, image_format = "
                 "%p, image_desc = %p, host_ptr = %p, errcode_ret = %p",
                 context, properties, flags, image_format, image_desc, host_ptr,
                 errcode_ret);

    cl_int err;
    std::vector<cl_mem_properties> props;

    if (properties != nullptr) {
        while (*properties) {
            props.push_back(*properties);
            properties++;
        }
        props.push_back(0);
    }

    auto image = cvk_create_image(context, flags, image_format, image_desc,
                                  host_ptr, std::move(props), &err);

    if (errcode_ret != nullptr) {
        *errcode_ret = err;
    }

    return image;
}

cl_mem CLVK_API_CALL clCreateImage2D(cl_context context, cl_mem_flags flags,
                                     const cl_image_format* image_format,
                                     size_t image_width, size_t image_height,
                                     size_t image_row_pitch, void* host_ptr,
                                     cl_int* errcode_ret) {
    TRACE_FUNCTION("context", (uintptr_t)context, "flags", flags);
    LOG_API_CALL(
        "context = %p, flags = %lu, image_format = %p, image_width = %zu, "
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
        0,        // image_slice_pitch
        0,        // num_mip_levels
        0,        // num_samples
        {nullptr} // buffer
    };

    cl_int err;
    auto image =
        cvk_create_image(context, flags, image_format, &desc, host_ptr, &err);

    if (errcode_ret != nullptr) {
        *errcode_ret = err;
    }

    return image;
}

cl_mem CLVK_API_CALL clCreateImage3D(cl_context context, cl_mem_flags flags,
                                     const cl_image_format* image_format,
                                     size_t image_width, size_t image_height,
                                     size_t image_depth, size_t image_row_pitch,
                                     size_t image_slice_pitch, void* host_ptr,
                                     cl_int* errcode_ret) {
    TRACE_FUNCTION("context", (uintptr_t)context, "flags", flags);
    LOG_API_CALL(
        "context = %p, flags = %lu, image_format = %p, image_width = %zu, "
        "image_height = %zu, image_depth = %zu, image_row_pitch = %zu, "
        "image_slice_pitch = %zu, host_ptr = %p, errcode_ret = %p",
        context, flags, image_format, image_width, image_height, image_depth,
        image_row_pitch, image_slice_pitch, host_ptr, errcode_ret);

    cl_image_desc desc = {
        CL_MEM_OBJECT_IMAGE3D,
        image_width,
        image_height,
        image_depth,
        0, // image_array_size
        image_row_pitch,
        image_slice_pitch,
        0,        // num_mip_levels
        0,        // num_samples
        {nullptr} // buffer
    };

    cl_int err;
    auto image =
        cvk_create_image(context, flags, image_format, &desc, host_ptr, &err);

    if (errcode_ret != nullptr) {
        *errcode_ret = err;
    }

    return image;
}

cl_int CLVK_API_CALL clGetImageInfo(cl_mem image, cl_image_info param_name,
                                    size_t param_value_size, void* param_value,
                                    size_t* param_value_size_ret) {
    TRACE_FUNCTION("image", (uintptr_t)image, "param_name", param_name);
    LOG_API_CALL("image = %p, param_name = %x, param_value_size = %zu, "
                 "param_value = %p, param_value_size_ret = %p",
                 image, param_name, param_value_size, param_value,
                 param_value_size_ret);

    cl_int ret = CL_SUCCESS;
    size_t ret_size = 0;
    const void* copy_ptr = nullptr;
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

bool operator!=(const VkComponentMapping& lhs, const VkComponentMapping& rhs) {
    return lhs.r != rhs.r || lhs.g != rhs.g || lhs.b != rhs.b || lhs.a != rhs.a;
}

static bool is_image_format_supported(
    VkPhysicalDevice& pdev, VkFormat format, cl_mem_object_type image_type,
    const VkFormatFeatureFlags& required_format_feature_flags,
    cl_channel_order image_channel_order) {
    VkFormatProperties properties;
    vkGetPhysicalDeviceFormatProperties(pdev, format, &properties);

    cvk_debug("Vulkan format %d:", format);
    cvk_debug(
        "  linear : %s",
        vulkan_format_features_string(properties.linearTilingFeatures).c_str());
    cvk_debug("  optimal: %s",
              vulkan_format_features_string(properties.optimalTilingFeatures)
                  .c_str());
    cvk_debug("  buffer : %s",
              vulkan_format_features_string(properties.bufferFeatures).c_str());

    cvk_debug(
        "Required format features %s",
        vulkan_format_features_string(required_format_feature_flags).c_str());
    VkFormatFeatureFlags features;
    if (image_type == CL_MEM_OBJECT_IMAGE1D_BUFFER) {
        features = properties.bufferFeatures;
    } else {
        // TODO support linear tiling
        features = properties.optimalTilingFeatures;
    }
    if ((features & required_format_feature_flags) ==
        required_format_feature_flags) {
        if ((image_channel_order == CL_LUMINANCE ||
             image_channel_order == CL_INTENSITY) &&
            (image_type == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
            return false;
        }
        return true;
    }
    return false;
}

cl_int CLVK_API_CALL clGetSupportedImageFormats(cl_context context,
                                                cl_mem_flags flags,
                                                cl_mem_object_type image_type,
                                                cl_uint num_entries,
                                                cl_image_format* image_formats,
                                                cl_uint* num_image_formats) {
    TRACE_FUNCTION("context", (uintptr_t)context, "flags", flags, "image_type",
                   image_type, "num_entries", num_entries);
    LOG_API_CALL(
        "context = %p, flags = %lu, image_type = %d, num_entries = %u, "
        "image_formats = %p, num_image_formats = %p",
        context, flags, image_type, num_entries, image_formats,
        num_image_formats);

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

    auto dev = icd_downcast(context)->device();
    auto pdev = dev->vulkan_physical_device();

    if (!dev->supports_read_write_images() &&
        (flags & CL_MEM_KERNEL_READ_AND_WRITE)) {
        if (num_image_formats != nullptr) {
            *num_image_formats = 0;
        }
        return CL_SUCCESS;
    }

    const VkFormatFeatureFlags required_format_feature_flags =
        cvk_image::required_format_feature_flags_for(image_type, flags);

    // TODO tiling selection
    //  No host access => OPTIMAL
    //  Host ACCESS => LINEAR if supported, OPTIMAL otherwise?

    // Iterate over all known CL/VK format associations and report the CL
    // formats for which the Vulkan format is supported
    for (auto mapping : get_format_maps()) {
        VkComponentMapping components_sampled, components_storage;
        image_format_support fmt_support;
        cl_image_format clfmt = mapping.first;
        if (!cl_image_format_to_vulkan_format(clfmt, image_type, dev,
                                              &fmt_support, &components_sampled,
                                              &components_storage)) {
            continue;
        }
        if ((fmt_support.flags & flags) != flags) {
            continue;
        }
        if (!is_image_format_supported(pdev, fmt_support.vkfmt, image_type,
                                       required_format_feature_flags,
                                       clfmt.image_channel_order)) {
            continue;
        }

        // image format is supported
        if ((image_formats != nullptr) && (num_formats_found < num_entries)) {
            image_formats[num_formats_found] = clfmt;
            cvk_debug_fn(
                "reporting image format {%s, %s}",
                cl_channel_order_to_string(clfmt.image_channel_order).c_str(),
                cl_channel_type_to_string(clfmt.image_channel_data_type)
                    .c_str());
        }
        num_formats_found++;
    }

    cvk_debug_fn("reporting %u formats", num_formats_found);

    if (num_image_formats != nullptr) {
        *num_image_formats = num_formats_found;
    }

    return CL_SUCCESS;
}

cl_int cvk_enqueue_image_copy(
    cvk_command_queue* queue, cl_command_type command_type, cvk_mem* image,
    bool blocking, const size_t* origin, const size_t* region, size_t row_pitch,
    size_t slice_pitch, void* ptr, cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list, cl_event* event) {

    auto img = static_cast<cvk_image*>(image);
    if (img->is_backed_by_buffer_view()) {
        auto cmd = new cvk_command_buffer_host_copy(
            queue, command_type, static_cast<cvk_buffer*>(img->buffer()), ptr,
            origin[0] * img->element_size(), region[0] * img->element_size());
        auto err = queue->enqueue_command_with_deps(
            cmd, blocking, num_events_in_wait_list, event_wait_list, event);
        return err;
    }

    // Create image map command
    std::array<size_t, 3> orig = {origin[0], origin[1], origin[2]};
    std::array<size_t, 3> reg = {region[0], region[1], region[2]};

    cl_map_flags map_flags;
    if (command_type == CL_COMMAND_WRITE_IMAGE) {
        map_flags = CL_MAP_WRITE_INVALIDATE_REGION;
    } else {
        map_flags = CL_MAP_READ;
    }

    auto cmd_map = std::make_unique<cvk_command_map_image>(queue, img, orig,
                                                           reg, map_flags);
    void* map_ptr;
    cl_int err = cmd_map->build(&map_ptr);
    if (err != CL_SUCCESS) {
        return err;
    }
    auto map_buffer = cmd_map->map_buffer();

    // Create copy command
    auto rpitch = row_pitch;
    if (rpitch == 0) {
        rpitch = region[0] * img->element_size();
    }

    auto spitch = slice_pitch;
    if (spitch == 0) {
        spitch = region[1] * rpitch;
    }
    const size_t zero_origin[3] = {0, 0, 0};
    auto cmd_copy = std::make_unique<cvk_command_copy_host_buffer_rect>(
        queue, command_type, map_buffer, ptr, zero_origin, zero_origin, region,
        rpitch, spitch, img->map_buffer_row_pitch(reg),
        img->map_buffer_slice_pitch(reg), img->element_size());

    // Create unmap command
    auto cmd_unmap =
        std::make_unique<cvk_command_unmap_image>(queue, img, map_ptr);
    err = cmd_unmap->build();
    if (err != CL_SUCCESS) {
        return err;
    }

    // Create combine command
    std::vector<std::unique_ptr<cvk_command>> commands;
    commands.emplace_back(std::move(cmd_map));
    commands.emplace_back(std::move(cmd_copy));
    commands.emplace_back(std::move(cmd_unmap));

    auto cmd =
        new cvk_command_combine(queue, command_type, std::move(commands));

    // Enqueue combined command
    return queue->enqueue_command_with_deps(
        cmd, blocking, num_events_in_wait_list, event_wait_list, event);
}

cl_int CLVK_API_CALL clEnqueueReadImage(
    cl_command_queue cq, cl_mem img, cl_bool blocking_read,
    const size_t* origin, const size_t* region, size_t row_pitch,
    size_t slice_pitch, void* ptr, cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list, cl_event* event) {
    TRACE_FUNCTION("command_queue", (uintptr_t)cq, "image", (uintptr_t)img,
                   "blocking_read", blocking_read, "num_events_in_wait_list",
                   num_events_in_wait_list);
    LOG_API_CALL("command_queue = %p, image = %p, blocking_read = %d, "
                 "origin = {%zu,%zu,%zu}, region = {%zu, %zu, %zu}, "
                 "row_pitch = %zu, slice_pitch = %zu, ptr = %p, "
                 "num_events_in_wait_list = %u, event_wait_list = %p, "
                 "event = %p",
                 cq, img, blocking_read, origin[0], origin[1], origin[2],
                 region[0], region[1], region[2], row_pitch, slice_pitch, ptr,
                 num_events_in_wait_list, event_wait_list, event);

    auto command_queue = icd_downcast(cq);
    auto image = icd_downcast(img);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    if (!is_valid_image(image)) {
        return CL_INVALID_MEM_OBJECT;
    }

    if (!is_valid_event_wait_list(num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    if (!is_same_context(command_queue, image) ||
        !is_same_context(command_queue, num_events_in_wait_list,
                         event_wait_list)) {
        return CL_INVALID_CONTEXT;
    }
    // TODO CL_INVALID_VALUE if the region being read specified by origin and
    // region is out of bounds or if ptr is a NULL value.
    // TODO CL_INVALID_VALUE if values in origin and region do not follow rules
    // described in the argument description for origin and region.
    // TODO CL_INVALID_IMAGE_SIZE if image dimensions (image width, height,
    // specified or compute row and/or slice pitch) for image are not supported
    // by device associated with queue.
    // TODO CL_INVALID_IMAGE_FORMAT if image format (image channel order and
    // data type) for image are not supported by device associated with queue.
    // TODO CL_MEM_OBJECT_ALLOCATION_FAILURE if there is a failure to allocate
    // memory for data store associated with image.
    if (!command_queue->device()->supports_images()) {
        return CL_INVALID_OPERATION;
    }

    if (image->has_any_flag(CL_MEM_HOST_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS)) {
        return CL_INVALID_OPERATION;
    }

    return cvk_enqueue_image_copy(command_queue, CL_COMMAND_READ_IMAGE, image,
                                  blocking_read, origin, region, row_pitch,
                                  slice_pitch, ptr, num_events_in_wait_list,
                                  event_wait_list, event);
}

cl_int CLVK_API_CALL clEnqueueWriteImage(
    cl_command_queue cq, cl_mem img, cl_bool blocking_write,
    const size_t* origin, const size_t* region, size_t input_row_pitch,
    size_t input_slice_pitch, const void* ptr, cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list, cl_event* event) {
    TRACE_FUNCTION("command_queue", (uintptr_t)cq, "image", (uintptr_t)img,
                   "blocking_write", blocking_write, "num_events_in_wait_list",
                   num_events_in_wait_list);
    LOG_API_CALL("command_queue = %p, image = %p, blocking_write = %d, "
                 "origin = {%zu,%zu,%zu}, region = {%zu, %zu, %zu}, "
                 "input_row_pitch = %zu, input_slice_pitch = %zu, ptr = %p, "
                 "num_events_in_wait_list = %u, event_wait_list = %p, "
                 "event = %p",
                 cq, img, blocking_write, origin[0], origin[1], origin[2],
                 region[0], region[1], region[2], input_row_pitch,
                 input_slice_pitch, ptr, num_events_in_wait_list,
                 event_wait_list, event);

    auto command_queue = icd_downcast(cq);
    auto image = icd_downcast(img);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    if (!is_valid_image(image)) {
        return CL_INVALID_MEM_OBJECT;
    }

    if (!is_valid_event_wait_list(num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    if (!is_same_context(command_queue, image) ||
        !is_same_context(command_queue, num_events_in_wait_list,
                         event_wait_list)) {
        return CL_INVALID_CONTEXT;
    }

    // TODO CL_INVALID_VALUE if the region being written specified by origin and
    // region is out of bounds or if ptr is a NULL value.
    // TODO CL_INVALID_VALUE if values in origin and region do not follow rules
    // described in the argument description for origin and region.
    // TODO CL_INVALID_IMAGE_SIZE if image dimensions (image width, height,
    // specified or compute row and/or slice pitch) for image are not supported
    // by device associated with queue.
    // TODO CL_INVALID_IMAGE_FORMAT if image format (image channel order and
    // data type) for image are not supported by device associated with queue.
    // TODO CL_MEM_OBJECT_ALLOCATION_FAILURE if there is a failure to allocate
    // memory for data store associated with image.
    if (!command_queue->device()->supports_images()) {
        return CL_INVALID_OPERATION;
    }

    if (image->has_any_flag(CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_NO_ACCESS)) {
        return CL_INVALID_OPERATION;
    }

    return cvk_enqueue_image_copy(
        command_queue, CL_COMMAND_WRITE_IMAGE, image, blocking_write, origin,
        region, input_row_pitch, input_slice_pitch, const_cast<void*>(ptr),
        num_events_in_wait_list, event_wait_list, event);
}

cl_int CLVK_API_CALL
clEnqueueCopyImage(cl_command_queue cq, cl_mem src_image, cl_mem dst_image,
                   const size_t* src_origin, const size_t* dst_origin,
                   const size_t* region, cl_uint num_events_in_wait_list,
                   const cl_event* event_wait_list, cl_event* event) {
    TRACE_FUNCTION("command_queue", (uintptr_t)cq, "src_image",
                   (uintptr_t)src_image, "dst_image", (uintptr_t)dst_image,
                   "num_events_in_wait_list", num_events_in_wait_list);
    LOG_API_CALL("command_queue = %p, src_image = %p, dst_image = %p, "
                 "src_origin = {%zu,%zu,%zu}, dst_origin = {%zu, %zu, %zu}, "
                 "region = {%zu, %zu, %zu}, "
                 "num_events_in_wait_list = %u, event_wait_list = %p, "
                 "event = %p",
                 cq, src_image, dst_image, src_origin[0], src_origin[1],
                 src_origin[2], dst_origin[0], dst_origin[1], dst_origin[2],
                 region[0], region[1], region[2], num_events_in_wait_list,
                 event_wait_list, event);

    auto command_queue = icd_downcast(cq);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    if (!is_valid_image(src_image) || !is_valid_image(dst_image)) {
        return CL_INVALID_MEM_OBJECT;
    }

    if (!is_same_context(command_queue, src_image) ||
        !is_same_context(command_queue, dst_image) ||
        !is_same_context(command_queue, num_events_in_wait_list,
                         event_wait_list)) {
        return CL_INVALID_CONTEXT;
    }

    if (!is_valid_event_wait_list(num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    auto src_img = static_cast<cvk_image*>(src_image);
    auto dst_img = static_cast<cvk_image*>(dst_image);

    // TODO CL_INVALID_VALUE if the 2D or 3D rectangular region specified by
    // src_origin and src_origin + region refers to a region outside src_image,
    // or if the 2D or 3D rectangular region specified by dst_origin and
    // dst_origin + region refers to a region outside dst_image.
    // TODO CL_INVALID_VALUE if values in src_origin, dst_origin and region do
    // not follow rules described in the argument description for src_origin,
    // dst_origin and region.
    // TODO CL_INVALID_IMAGE_SIZE if image dimensions (image width, height,
    // specified or compute row and/or slice pitch) for src_image or dst_image
    // are not supported by device associated with queue.
    // TODO CL_INVALID_IMAGE_FORMAT if image format (image channel order and
    // data type) for src_image or dst_image are not supported by device
    // associated with queue.
    // TODO CL_MEM_OBJECT_ALLOCATION_FAILURE if there is a failure to allocate
    // memory for data store associated with src_image or dst_image.
    if (!command_queue->device()->supports_images()) {
        return CL_INVALID_OPERATION;
    }

    // TODO CL_MEM_COPY_OVERLAP if src_image and dst_image are the same image
    // object and the source and destination regions overlap.

    if (!src_img->has_same_format(dst_img)) {
        return CL_IMAGE_FORMAT_MISMATCH;
    }

    std::array<size_t, 3> src_orig = {src_origin[0], src_origin[1],
                                      src_origin[2]};
    std::array<size_t, 3> dst_orig = {dst_origin[0], dst_origin[1],
                                      dst_origin[2]};
    std::array<size_t, 3> reg = {region[0], region[1], region[2]};

    if (src_img->is_backed_by_buffer_view() &&
        dst_img->is_backed_by_buffer_view()) {
        auto cmd = new cvk_command_copy_buffer(
            command_queue, CL_COMMAND_COPY_IMAGE,
            static_cast<cvk_buffer*>(src_img->buffer()),
            static_cast<cvk_buffer*>(dst_img->buffer()),
            src_origin[0] * src_img->element_size(),
            dst_origin[0] * dst_img->element_size(),
            region[0] * src_img->element_size());

        return command_queue->enqueue_command_with_deps(
            cmd, num_events_in_wait_list, event_wait_list, event);
    } else if (src_img->is_backed_by_buffer_view()) {
        auto cmd = std::make_unique<cvk_command_buffer_image_copy>(
            CL_COMMAND_COPY_IMAGE, CL_COMMAND_COPY_BUFFER_TO_IMAGE,
            command_queue, static_cast<cvk_buffer*>(src_img->buffer()), dst_img,
            src_origin[0] * src_img->element_size(), dst_orig, reg);

        return command_queue->enqueue_command_with_deps(
            cmd.release(), num_events_in_wait_list, event_wait_list, event);
    } else if (dst_img->is_backed_by_buffer_view()) {
        auto cmd = std::make_unique<cvk_command_buffer_image_copy>(
            CL_COMMAND_COPY_IMAGE, CL_COMMAND_COPY_IMAGE_TO_BUFFER,
            command_queue, static_cast<cvk_buffer*>(dst_img->buffer()), src_img,
            dst_origin[0] * dst_img->element_size(), src_orig, reg);

        return command_queue->enqueue_command_with_deps(
            cmd.release(), num_events_in_wait_list, event_wait_list, event);
    } else {
        auto cmd = std::make_unique<cvk_command_image_image_copy>(
            command_queue, src_img, dst_img, src_orig, dst_orig, reg);

        return command_queue->enqueue_command_with_deps(
            cmd.release(), num_events_in_wait_list, event_wait_list, event);
    }
}

cl_int CLVK_API_CALL clEnqueueFillImage(
    cl_command_queue cq, cl_mem image, const void* fill_color,
    const size_t* origin, const size_t* region, cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list, cl_event* event) {
    TRACE_FUNCTION("command_queue", (uintptr_t)cq, "image", (uintptr_t)image,
                   "num_events_in_wait_list", num_events_in_wait_list);
    LOG_API_CALL("command_queue = %p, image = %p, fill_color = %p, "
                 "origin = {%zu,%zu,%zu}, region = {%zu, %zu, %zu}, "
                 "num_events_in_wait_list = %u, event_wait_list = %p, "
                 "event = %p",
                 cq, image, fill_color, origin[0], origin[1], origin[2],
                 region[0], region[1], region[2], num_events_in_wait_list,
                 event_wait_list, event);

    auto command_queue = icd_downcast(cq);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    if (!is_valid_image(image)) {
        return CL_INVALID_MEM_OBJECT;
    }

    if (!is_valid_event_wait_list(num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    if (!is_same_context(command_queue, image) ||
        !is_same_context(command_queue, num_events_in_wait_list,
                         event_wait_list)) {
        return CL_INVALID_CONTEXT;
    }

    if (fill_color == nullptr) {
        return CL_INVALID_VALUE;
    }
    // TODO CL_INVALID_VALUE if the region being written specified by origin and
    // region is out of bounds or if ptr is a NULL value.
    // TODO CL_INVALID_VALUE if values in origin and region do not follow rules
    // described in the argument description for origin and region.
    // TODO CL_INVALID_IMAGE_SIZE if image dimensions (image width, height,
    // specified or compute row and/or slice pitch) for image are not supported
    // by device associated with queue.
    // TODO CL_INVALID_IMAGE_FORMAT if image format (image channel order and
    // data type) for image are not supported by device associated with queue.
    // TODO CL_MEM_OBJECT_ALLOCATION_FAILURE if there is a failure to allocate
    // memory for data store associated with image.

    // TODO use Vulkan clear commands when possible
    // TODO use a shader when better

    auto img = static_cast<cvk_image*>(image);

    // Fill
    cvk_image::fill_pattern_array pattern;
    size_t pattern_size;
    img->prepare_fill_pattern(fill_color, pattern, &pattern_size);

    if (img->is_backed_by_buffer_view()) {
        auto cmd = new cvk_command_fill_buffer(
            command_queue, static_cast<cvk_buffer*>(img->buffer()),
            origin[0] * img->element_size(), region[0] * img->element_size(),
            pattern.data(), pattern_size, CL_COMMAND_FILL_IMAGE);

        return command_queue->enqueue_command_with_deps(
            cmd, num_events_in_wait_list, event_wait_list, event);
    }

    // Create image map command
    std::array<size_t, 3> orig = {origin[0], origin[1], origin[2]};
    std::array<size_t, 3> reg = {region[0], region[1], region[2]};

    auto cmd_map = std::make_unique<cvk_command_map_image>(
        command_queue, img, orig, reg, CL_MAP_WRITE_INVALIDATE_REGION);
    void* map_ptr;
    cl_int err = cmd_map->build(&map_ptr);
    if (err != CL_SUCCESS) {
        return err;
    }

    auto cmd_fill = std::make_unique<cvk_command_fill_image>(
        command_queue, map_ptr, pattern, pattern_size, reg);

    // Create unmap command
    auto cmd_unmap =
        std::make_unique<cvk_command_unmap_image>(command_queue, img, map_ptr);
    err = cmd_unmap->build();
    if (err != CL_SUCCESS) {
        return err;
    }

    // Create combine command
    std::vector<std::unique_ptr<cvk_command>> commands;
    commands.emplace_back(std::move(cmd_map));
    commands.emplace_back(std::move(cmd_fill));
    commands.emplace_back(std::move(cmd_unmap));

    auto cmd = new cvk_command_combine(command_queue, CL_COMMAND_FILL_IMAGE,
                                       std::move(commands));

    // Enqueue combined command
    return command_queue->enqueue_command_with_deps(
        cmd, num_events_in_wait_list, event_wait_list, event);
}

cl_int CLVK_API_CALL clEnqueueCopyImageToBuffer(
    cl_command_queue cq, cl_mem src_image, cl_mem dst_buffer,
    const size_t* src_origin, const size_t* region, size_t dst_offset,
    cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
    cl_event* event) {
    TRACE_FUNCTION("command_queue", (uintptr_t)cq, "src_image",
                   (uintptr_t)src_image, "dst_buffer", (uintptr_t)dst_buffer,
                   "num_events_in_wait_list", num_events_in_wait_list);
    LOG_API_CALL("command_queue = %p, src_image = %p, dst_buffer = %p, "
                 "src_origin = {%zu,%zu,%zu}, region = {%zu, %zu, %zu}, "
                 "dst_offset = %zu, num_events_in_wait_list = %u, "
                 "event_wait_list = %p, event = %p",
                 cq, src_image, dst_buffer, src_origin[0], src_origin[1],
                 src_origin[2], region[0], region[1], region[2], dst_offset,
                 num_events_in_wait_list, event_wait_list, event);

    auto command_queue = icd_downcast(cq);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    if (!is_valid_image(src_image) || !is_valid_buffer(dst_buffer)) {
        return CL_INVALID_MEM_OBJECT;
    }

    if (!is_valid_event_wait_list(num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    if (!is_same_context(command_queue, src_image) ||
        !is_same_context(command_queue, dst_buffer) ||
        !is_same_context(command_queue, num_events_in_wait_list,
                         event_wait_list)) {
        return CL_INVALID_CONTEXT;
    }
    // TODO CL_INVALID_MEM_OBJECT if src_image is a 1D image buffer object
    // created from dst_buffer.
    // TODO CL_INVALID_VALUE if the 1D, 2D, or 3D rectangular region specified
    // by src_origin and src_origin + region refers to a region outside
    // src_image, or if the region specified by dst_offset and dst_offset +
    // dst_cb refers to a region outside dst_buffer.
    // TODO CL_INVALID_VALUE if values in src_origin and region do not follow
    // rules described in the argument description for src_origin and region.

    // TODO CL_MISALIGNED_SUB_BUFFER_OFFSET if dst_buffer is a sub-buffer object
    // and offset specified when the sub-buffer object is created is not aligned
    // to CL_DEVICE_MEM_BASE_ADDR_ALIGN value for device associated with queue.
    // TODO CL_INVALID_IMAGE_SIZE if image dimensions (image width, height,
    // specified or compute row and/or slice pitch) for src_image are not
    // supported by device associated with queue.
    // TODO CL_INVALID_IMAGE_FORMAT if image format (image channel order and
    // data type) for src_image are not supported by device associated with
    // queue.
    // TODO CL_MEM_OBJECT_ALLOCATION_FAILURE if there is a failure to allocate
    // memory for data store associated with src_image or dst_buffer.

    auto image = static_cast<cvk_image*>(src_image);
    auto buffer = static_cast<cvk_buffer*>(dst_buffer);

    if (!command_queue->device()->supports_images()) {
        return CL_INVALID_OPERATION;
    }

    std::array<size_t, 3> origin = {src_origin[0], src_origin[1],
                                    src_origin[2]};
    std::array<size_t, 3> reg = {region[0], region[1], region[2]};

    if (image->is_backed_by_buffer_view()) {
        auto cmd = new cvk_command_copy_buffer(
            command_queue, CL_COMMAND_COPY_IMAGE_TO_BUFFER,
            static_cast<cvk_buffer*>(image->buffer()), buffer,
            src_origin[0] * image->element_size(), dst_offset,
            region[0] * image->element_size());

        return command_queue->enqueue_command_with_deps(
            cmd, num_events_in_wait_list, event_wait_list, event);
    } else {
        auto cmd = std::make_unique<cvk_command_buffer_image_copy>(
            CL_COMMAND_COPY_IMAGE_TO_BUFFER, command_queue, buffer, image,
            dst_offset, origin, reg);

        return command_queue->enqueue_command_with_deps(
            cmd.release(), num_events_in_wait_list, event_wait_list, event);
    }
}

cl_int CLVK_API_CALL clEnqueueCopyBufferToImage(
    cl_command_queue cq, cl_mem src_buffer, cl_mem dst_image, size_t src_offset,
    const size_t* dst_origin, const size_t* region,
    cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
    cl_event* event) {
    TRACE_FUNCTION("command_queue", (uintptr_t)cq, "src_buffer",
                   (uintptr_t)src_buffer, "dst_image", (uintptr_t)dst_image,
                   "num_events_in_wait_list", num_events_in_wait_list);
    LOG_API_CALL("command_queue = %p, src_buffer = %p, dst_image = %p, "
                 "src_offset = %zu, "
                 "dst_origin = {%zu,%zu,%zu}, region = {%zu, %zu, %zu}, "
                 "num_events_in_wait_list = %u, event_wait_list = %p, "
                 "event = %p",
                 cq, src_buffer, dst_image, src_offset, dst_origin[0],
                 dst_origin[1], dst_origin[2], region[0], region[1], region[2],
                 num_events_in_wait_list, event_wait_list, event);

    auto command_queue = icd_downcast(cq);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    if (!is_valid_image(dst_image) || !is_valid_buffer(src_buffer)) {
        return CL_INVALID_MEM_OBJECT;
    }

    if (!is_valid_event_wait_list(num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    if (!is_same_context(command_queue, src_buffer) ||
        !is_same_context(command_queue, dst_image) ||
        !is_same_context(command_queue, num_events_in_wait_list,
                         event_wait_list)) {
        return CL_INVALID_CONTEXT;
    }
    // TODO CL_INVALID_MEM_OBJECT if dst_image is a 1D image buffer object
    // created from src_buffer.
    // TODO CL_INVALID_VALUE if the 1D, 2D, or 3D rectangular region specified
    // by dst_origin and dst_origin + region refers to a region outside
    // dst_origin, or if the region specified by src_offset and src_offset +
    // src_cb refers to a region outside src_buffer.
    // TODO CL_INVALID_VALUE if values in dst_origin and region do not follow
    // rules described in the argument description for dst_origin and region.
    // TODO CL_MISALIGNED_SUB_BUFFER_OFFSET if src_buffer is a sub-buffer object
    // and offset specified when the sub-buffer object is created is not aligned
    // to CL_DEVICE_MEM_BASE_ADDR_ALIGN value for device associated with queue.
    // TODO CL_INVALID_IMAGE_SIZE if image dimensions (image width, height,
    // specified or compute row and/or slice pitch) for dst_image are not
    // supported by device associated with queue.
    // TODO CL_INVALID_IMAGE_FORMAT if image format (image channel order and
    // data type) for dst_image are not supported by device associated with
    // queue.
    // TODO CL_MEM_OBJECT_ALLOCATION_FAILURE if there is a failure to allocate
    // memory for data store associated with src_buffer or dst_image.

    auto image = static_cast<cvk_image*>(dst_image);
    auto buffer = static_cast<cvk_buffer*>(src_buffer);

    if (!command_queue->device()->supports_images()) {
        return CL_INVALID_OPERATION;
    }

    std::array<size_t, 3> origin = {dst_origin[0], dst_origin[1],
                                    dst_origin[2]};
    std::array<size_t, 3> reg = {region[0], region[1], region[2]};

    if (image->is_backed_by_buffer_view()) {
        auto cmd = new cvk_command_copy_buffer(
            command_queue, CL_COMMAND_COPY_BUFFER_TO_IMAGE, buffer,
            static_cast<cvk_buffer*>(image->buffer()), src_offset,
            dst_origin[0] * image->element_size(),
            region[0] * image->element_size());

        return command_queue->enqueue_command_with_deps(
            cmd, num_events_in_wait_list, event_wait_list, event);
    } else {
        auto cmd = std::make_unique<cvk_command_buffer_image_copy>(
            CL_COMMAND_COPY_BUFFER_TO_IMAGE, command_queue, buffer, image,
            src_offset, origin, reg);

        return command_queue->enqueue_command_with_deps(
            cmd.release(), num_events_in_wait_list, event_wait_list, event);
    }
}

void* cvk_enqueue_map_image(cl_command_queue cq, cl_mem img,
                            cl_bool blocking_map, cl_map_flags map_flags,
                            const size_t* origin, const size_t* region,
                            size_t* image_row_pitch, size_t* image_slice_pitch,
                            cl_uint num_events_in_wait_list,
                            const cl_event* event_wait_list, cl_event* event,
                            cl_int* errcode_ret) {
    auto command_queue = icd_downcast(cq);
    auto image = static_cast<cvk_image*>(img);

    if (!is_valid_command_queue(command_queue)) {
        *errcode_ret = CL_INVALID_COMMAND_QUEUE;
        return nullptr;
    }

    if (!is_valid_image(image)) {
        *errcode_ret = CL_INVALID_MEM_OBJECT;
        return nullptr;
    }

    if (!is_valid_event_wait_list(num_events_in_wait_list, event_wait_list)) {
        *errcode_ret = CL_INVALID_EVENT_WAIT_LIST;
        return nullptr;
    }

    if (!is_same_context(command_queue, image) ||
        !is_same_context(command_queue, num_events_in_wait_list,
                         event_wait_list)) {
        *errcode_ret = CL_INVALID_CONTEXT;
        return nullptr;
    }

    if (!map_flags_are_valid(map_flags)) {
        *errcode_ret = CL_INVALID_VALUE;
        return nullptr;
    }
    // TODO CL_INVALID_VALUE if region being mapped given by (origin,
    // origin+region) is out of bounds
    // TODO CL_INVALID_VALUE if values in origin and region do not follow rules
    // described in the argument description for origin and region.

    if (image_row_pitch == nullptr) {
        *errcode_ret = CL_INVALID_VALUE;
        return nullptr;
    }

    switch (image->type()) {
    case CL_MEM_OBJECT_IMAGE3D:
    case CL_MEM_OBJECT_IMAGE1D_ARRAY:
    case CL_MEM_OBJECT_IMAGE2D_ARRAY:
        if (image_slice_pitch == nullptr) {
            *errcode_ret = CL_INVALID_VALUE;
            return nullptr;
        }
        break;
    default:
        break;
    }
    // TODO CL_INVALID_IMAGE_SIZE if image dimensions (image width, height,
    // specified or compute row and/or slice pitch) for image are not supported
    // by device associated with queue.
    // TODO CL_INVALID_IMAGE_FORMAT if image format (image channel order and
    // data type) for image are not supported by device associated with queue.
    // TODO CL_MAP_FAILURE if there is a failure to map the requested region
    // into the host address space. This error cannot occur for image objects
    // created with CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR.
    // TODO CL_MEM_OBJECT_ALLOCATION_FAILURE if there is a failure to allocate
    // memory for data store associated with buffer.
    if (!command_queue->device()->supports_images()) {
        *errcode_ret = CL_INVALID_OPERATION;
        return nullptr;
    }

    if ((map_flags & CL_MAP_READ) &&
        (image->has_any_flag(CL_MEM_HOST_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS))) {
        *errcode_ret = CL_INVALID_OPERATION;
        return nullptr;
    }

    if (((map_flags & CL_MAP_WRITE) ||
         (map_flags & CL_MAP_WRITE_INVALIDATE_REGION)) &&
        (image->has_any_flag(CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_NO_ACCESS))) {
        *errcode_ret = CL_INVALID_OPERATION;
        return nullptr;
    }

    std::array<size_t, 3> orig = {origin[0], origin[1], origin[2]};
    std::array<size_t, 3> reg = {region[0], region[1], region[2]};
    auto cmd = std::make_unique<cvk_command_map_image>(
        command_queue, image, orig, reg, map_flags, true);

    void* map_ptr;
    cl_int err = cmd->build(&map_ptr);

    if (err != CL_SUCCESS) {
        *errcode_ret = err;
        return nullptr;
    }

    *image_row_pitch = cmd->row_pitch();
    if (image_slice_pitch != nullptr) {
        *image_slice_pitch = cmd->slice_pitch();
    }

    err = command_queue->enqueue_command_with_deps(cmd.release(), blocking_map,
                                                   num_events_in_wait_list,
                                                   event_wait_list, event);

    *errcode_ret = err;

    return map_ptr;
}

void* CLVK_API_CALL clEnqueueMapImage(
    cl_command_queue cq, cl_mem image, cl_bool blocking_map,
    cl_map_flags map_flags, const size_t* origin, const size_t* region,
    size_t* image_row_pitch, size_t* image_slice_pitch,
    cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
    cl_event* event, cl_int* errcode_ret) {
    TRACE_FUNCTION("command_queue", (uintptr_t)cq, "image", (uintptr_t)image,
                   "blocking_map", blocking_map, "map_flags", map_flags,
                   "num_events_in_wait_list", num_events_in_wait_list);
    LOG_API_CALL("command_queue = %p, image = %p, blocking_map = %d, "
                 "map_flags = %lx, "
                 "origin = {%zu,%zu,%zu}, region = {%zu, %zu, %zu}, "
                 "image_row_pitch = %p, image_slice_pitch = %p, "
                 "num_events_in_wait_list = %u, event_wait_list = %p, "
                 "event = %p, errcode_ret = %p",
                 cq, image, blocking_map, map_flags, origin[0], origin[1],
                 origin[2], region[0], region[1], region[2], image_row_pitch,
                 image_slice_pitch, num_events_in_wait_list, event_wait_list,
                 event, errcode_ret);

    auto command_queue = icd_downcast(cq);

    cl_int err;
    void* ret;
    auto img = static_cast<cvk_image*>(image);
    if (img->is_backed_by_buffer_view()) {
        ret = cvk_enqueue_map_buffer(
            command_queue, static_cast<cvk_buffer*>(img->buffer()),
            blocking_map, origin[0] * img->element_size(),
            region[0] * img->element_size(), map_flags, num_events_in_wait_list,
            event_wait_list, event, &err, CL_COMMAND_MAP_IMAGE, img);
    } else {
        ret = cvk_enqueue_map_image(command_queue, image, blocking_map,
                                    map_flags, origin, region, image_row_pitch,
                                    image_slice_pitch, num_events_in_wait_list,
                                    event_wait_list, event, &err);
    }

    if (errcode_ret != nullptr) {
        *errcode_ret = err;
    }

    return ret;
}

cl_program cvk_create_program_with_il(cl_context context, const void* il,
                                      size_t length, cl_int* errcode_ret) {
    if (!is_valid_context(context)) {
        *errcode_ret = CL_INVALID_CONTEXT;
        return nullptr;
    }

    if ((il == nullptr) || (length == 0)) {
        *errcode_ret = CL_INVALID_VALUE;
        return nullptr;
    }

    // TODO CL_INVALID_VALUE if the length-byte block of memory pointed to by il
    // does not contain well-formed intermediate language.

    auto program = new cvk_program(icd_downcast(context), il, length);

    *errcode_ret = program->parse_user_spec_constants();

    return program;
}

cl_program CLVK_API_CALL clCreateProgramWithILKHR(cl_context context,
                                                  const void* il, size_t length,
                                                  cl_int* errcode_ret) {
    TRACE_FUNCTION("context", (uintptr_t)context);
    LOG_API_CALL("context = %p, il = %p, length = %zu, errcode_ret = %p",
                 context, il, length, errcode_ret);

    cl_int errcode;
    auto program = cvk_create_program_with_il(context, il, length, &errcode);

    if (errcode_ret != nullptr) {
        *errcode_ret = errcode;
    }

    return program;
}

cl_program CLVK_API_CALL clCreateProgramWithIL(cl_context context,
                                               const void* il, size_t length,
                                               cl_int* errcode_ret) {
    TRACE_FUNCTION("context", (uintptr_t)context);
    LOG_API_CALL("context = %p, il = %p, length = %zu, errcode_ret = %p",
                 context, il, length, errcode_ret);

    cl_int errcode;
    auto program = cvk_create_program_with_il(context, il, length, &errcode);

    if (errcode_ret != nullptr) {
        *errcode_ret = errcode;
    }

    return program;
}

cl_int CLVK_API_CALL
clSetProgramSpecializationConstant(cl_program prog, cl_uint spec_id,
                                   size_t spec_size, const void* spec_value) {
    TRACE_FUNCTION("program", (uintptr_t)prog, "spec_id", spec_id, "spec_size",
                   spec_size);
    LOG_API_CALL("program = %p, spec_id = %u, spec_size = %zu, spec_value = %p",
                 prog, spec_id, spec_size, spec_value);

#if !ENABLE_SPIRV_IL
    // If the SPIRV Intermediate Language is not enabled, the OpenCL
    // specification requires clSetProgramSpecializationConstant to return
    // CL_INVALID_OPERATION.
    return CL_INVALID_OPERATION;
#endif

    auto program = icd_downcast(prog);

    if (!is_valid_program(program)) {
        return CL_INVALID_PROGRAM;
    }

    // TODO CL_INVALID_OPERATION if no devices associated with program support
    // intermediate language programs.
    // TODO CL_COMPILER_NOT_AVAILABLE if program is created with
    // clCreateProgramWithIL and a compiler is not available, i.e.
    // CL_DEVICE_COMPILER_AVAILABLE specified in the Device Queries table is set
    // to CL_FALSE.

    if (spec_value == nullptr) {
        return CL_INVALID_VALUE;
    }

    return program->set_user_spec_constant(spec_id, spec_size, spec_value);
}

// Shared Virtual Memory
void* CLVK_API_CALL clSVMAlloc(cl_context context, cl_svm_mem_flags flags,
                               size_t size, cl_uint alignment) {

    TRACE_FUNCTION("context", (uintptr_t)context, "flags", flags, "size", size,
                   "alignment", alignment);
    LOG_API_CALL("context = %p, flags = %lu, size = %zu, alignment = %u",
                 context, flags, size, alignment);

    return nullptr;
}

void CLVK_API_CALL clSVMFree(cl_context context, void* svm_pointer) {
    TRACE_FUNCTION("context", (uintptr_t)context);
    LOG_API_CALL("context = %p, svm_pointer = %p", context, svm_pointer);
}

cl_int CLVK_API_CALL clEnqueueSVMFree(
    cl_command_queue command_queue, cl_uint num_svm_pointers,
    void* svm_pointers[],
    void(CL_CALLBACK* pfn_free_func)(cl_command_queue queue,
                                     cl_uint num_svm_pointers,
                                     void* svm_pointers[], void* user_data),
    void* user_data, cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list, cl_event* event) {
    TRACE_FUNCTION("command_queue", (uintptr_t)command_queue,
                   "num_svm_pointers", num_svm_pointers,
                   "num_events_in_wait_list", num_events_in_wait_list);
    LOG_API_CALL("command_queue = %p", command_queue);
    UNUSED(num_svm_pointers);
    UNUSED(svm_pointers);
    UNUSED(pfn_free_func);
    UNUSED(user_data);
    UNUSED(num_events_in_wait_list);
    UNUSED(event_wait_list);
    UNUSED(event);
    return CL_INVALID_OPERATION;
}

cl_int CLVK_API_CALL clEnqueueSVMMap(cl_command_queue command_queue,
                                     cl_bool blocking_map, cl_map_flags flags,
                                     void* svm_ptr, size_t size,
                                     cl_uint num_events_in_wait_list,
                                     const cl_event* event_wait_list,
                                     cl_event* event) {
    TRACE_FUNCTION("command_queue", (uintptr_t)command_queue);
    LOG_API_CALL("command_queue = %p", command_queue);
    UNUSED(blocking_map);
    UNUSED(flags);
    UNUSED(svm_ptr);
    UNUSED(size);
    UNUSED(num_events_in_wait_list);
    UNUSED(event_wait_list);
    UNUSED(event);
    return CL_INVALID_OPERATION;
}

cl_int CLVK_API_CALL clEnqueueSVMMemcpy(cl_command_queue command_queue,
                                        cl_bool blocking_copy, void* dst_ptr,
                                        const void* src_ptr, size_t size,
                                        cl_uint num_events_in_wait_list,
                                        const cl_event* event_wait_list,
                                        cl_event* event) {
    TRACE_FUNCTION("command_queue", (uintptr_t)command_queue);
    LOG_API_CALL("command_queue = %p", command_queue);
    UNUSED(blocking_copy);
    UNUSED(dst_ptr);
    UNUSED(src_ptr);
    UNUSED(size);
    UNUSED(num_events_in_wait_list);
    UNUSED(event_wait_list);
    UNUSED(event);
    return CL_INVALID_OPERATION;
}

cl_int CLVK_API_CALL clEnqueueSVMMemFill(cl_command_queue command_queue,
                                         void* svm_ptr, const void* pattern,
                                         size_t pattern_size, size_t size,
                                         cl_uint num_events_in_wait_list,
                                         const cl_event* event_wait_list,
                                         cl_event* event) {
    TRACE_FUNCTION("command_queue", (uintptr_t)command_queue);
    LOG_API_CALL("command_queue = %p", command_queue);
    UNUSED(svm_ptr);
    UNUSED(pattern);
    UNUSED(pattern_size);
    UNUSED(size);
    UNUSED(num_events_in_wait_list);
    UNUSED(event_wait_list);
    UNUSED(event);
    return CL_INVALID_OPERATION;
}

cl_int CLVK_API_CALL clEnqueueSVMMigrateMem(
    cl_command_queue command_queue, cl_uint num_svm_pointers,
    const void** svm_pointers, const size_t* sizes,
    cl_mem_migration_flags flags, cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list, cl_event* event) {
    TRACE_FUNCTION("command_queue", (uintptr_t)command_queue);
    LOG_API_CALL("command_queue = %p", command_queue);
    UNUSED(num_svm_pointers);
    UNUSED(svm_pointers);
    UNUSED(sizes);
    UNUSED(flags);
    UNUSED(num_events_in_wait_list);
    UNUSED(event_wait_list);
    UNUSED(event);
    return CL_INVALID_OPERATION;
}

cl_int CLVK_API_CALL clEnqueueSVMUnmap(cl_command_queue command_queue,
                                       void* svm_ptr,
                                       cl_uint num_events_in_wait_list,
                                       const cl_event* event_wait_list,
                                       cl_event* event) {
    TRACE_FUNCTION("command_queue", (uintptr_t)command_queue);
    LOG_API_CALL("command_queue = %p", command_queue);
    UNUSED(svm_ptr);
    UNUSED(num_events_in_wait_list);
    UNUSED(event_wait_list);
    UNUSED(event);
    return CL_INVALID_OPERATION;
}

cl_int CLVK_API_CALL clSetKernelArgSVMPointer(cl_kernel kernel,
                                              cl_uint arg_index,
                                              const void* arg_value) {
    TRACE_FUNCTION("kernel", (uintptr_t)kernel, "arg_index", arg_index);
    LOG_API_CALL("kernel = %p, arg_index = %u, arg_value = %p", kernel,
                 arg_index, arg_value);
    UNUSED(kernel);
    UNUSED(arg_index);
    UNUSED(arg_value);
    return CL_INVALID_OPERATION;
}

// Pipes
cl_mem CLVK_API_CALL clCreatePipe(cl_context context, cl_mem_flags flags,
                                  cl_uint pipe_packet_size,
                                  cl_uint pipe_max_packets,
                                  const cl_pipe_properties* properties,
                                  cl_int* errcode_ret) {
    TRACE_FUNCTION("context", (uintptr_t)context, "flags", flags,
                   "pipe_packet_size", pipe_packet_size, "pipe_max_packets",
                   pipe_max_packets);
    LOG_API_CALL("context = %p, flags = %lx, packet_size = %u, max_packets = "
                 "%u, properties = %p, errcode_ret = %p",
                 context, flags, pipe_packet_size, pipe_max_packets, properties,
                 errcode_ret);
    if (errcode_ret != nullptr) {
        *errcode_ret = CL_INVALID_OPERATION;
    }
    return nullptr;
}

cl_int CLVK_API_CALL clGetPipeInfo(cl_mem pipe, cl_pipe_info param_name,
                                   size_t param_value_size, void* param_value,
                                   size_t* param_value_size_ret) {
    TRACE_FUNCTION("pipe", (uintptr_t)pipe, "param_name", param_name);
    LOG_API_CALL("pipe = %p, param_name = %x, param_value_size = %zu, "
                 "param_value = %p, param_value_size_ret = %p",
                 pipe, param_name, param_value_size, param_value,
                 param_value_size_ret);
    return CL_INVALID_MEM_OBJECT;
}

// Timer functions
cl_int CLVK_API_CALL clGetHostTimer(cl_device_id device,
                                    cl_ulong* host_timestamp) {

    TRACE_FUNCTION("device", (uintptr_t)device);
    LOG_API_CALL("device = %p, host_timestamp = %p", device, host_timestamp);

    if (!is_valid_device(device)) {
        return CL_INVALID_DEVICE;
    }

    auto dev = icd_downcast(device);

    if (!dev->has_timer_support()) {
        return CL_INVALID_OPERATION;
    }

    if (host_timestamp == nullptr) {
        return CL_INVALID_VALUE;
    }

    return dev->get_device_host_timer(nullptr, host_timestamp);
}

cl_int CLVK_API_CALL clGetDeviceAndHostTimer(cl_device_id device,
                                             cl_ulong* device_timestamp,
                                             cl_ulong* host_timestamp) {

    TRACE_FUNCTION("device", (uintptr_t)device);
    LOG_API_CALL("device = %p, device_timestamp = %p, host_timestamp = %p",
                 device, device_timestamp, host_timestamp);

    if (!is_valid_device(device)) {
        return CL_INVALID_DEVICE;
    }

    auto dev = icd_downcast(device);

    if (!dev->has_timer_support()) {
        return CL_INVALID_OPERATION;
    }

    if ((device_timestamp == nullptr) || (host_timestamp == nullptr)) {
        return CL_INVALID_VALUE;
    }

    cl_ulong host;

    cl_int err = dev->get_device_host_timer(nullptr, &host);

    *device_timestamp = host;
    *host_timestamp = host;

    return err;
}

// cl_khr_semaphore
cl_semaphore_khr cvk_create_semaphore_with_properties_khr(
    cl_context context, const cl_semaphore_properties_khr* sema_props,
    cl_int* errcode_ret) {

    if (!is_valid_context(context)) {
        *errcode_ret = CL_INVALID_CONTEXT;
        return nullptr;
    }

    if (sema_props == nullptr) {
        *errcode_ret = CL_INVALID_VALUE;
        return nullptr;
    }

    cl_semaphore_type_khr type = 0;
    std::vector<cl_semaphore_properties_khr> properties;
    std::vector<cl_device_id> devices;

    if (sema_props) {
        bool has_type = false;
        while (*sema_props) {
            auto key = *sema_props;
            auto value = *(sema_props + 1);

            if (key == CL_SEMAPHORE_TYPE_KHR) {
                properties.push_back(key);
                properties.push_back(value);
                type = value;
                sema_props += 2;
                has_type = true;
            } else if (key == CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR) {
                properties.push_back(key);
                sema_props++;
                while (*sema_props != CL_SEMAPHORE_DEVICE_HANDLE_LIST_END_KHR) {
                    auto devapi = reinterpret_cast<cl_device_id>(*sema_props);
                    if (!is_valid_device(devapi)) {
                        *errcode_ret = CL_INVALID_DEVICE;
                        return nullptr;
                    }
                    auto dev = static_cast<cvk_device*>(devapi);
                    if (!icd_downcast(context)->has_device(dev)) {
                        *errcode_ret = CL_INVALID_DEVICE;
                        return nullptr;
                    }
                    devices.push_back(devapi);
                    properties.push_back(*sema_props);
                    sema_props++;
                }
                properties.push_back(*sema_props);
                sema_props++;
            } else {
                *errcode_ret = CL_INVALID_PROPERTY;
                return nullptr;
            }
        }

        if (!has_type) {
            *errcode_ret = CL_INVALID_VALUE;
            return nullptr;
        }

        properties.push_back(0);
    }

    if (type == 0) {
        *errcode_ret = CL_INVALID_VALUE;
        return nullptr;
    }

    auto sem = std::make_unique<cvk_semaphore>(
        icd_downcast(context), type, std::move(devices), std::move(properties));

    auto err = sem->init();
    if (err != CL_SUCCESS) {
        *errcode_ret = err;
        return nullptr;
    }

    *errcode_ret = CL_SUCCESS;

    return sem.release();
}

cl_semaphore_khr clCreateSemaphoreWithPropertiesKHR(
    cl_context context, const cl_semaphore_properties_khr* sema_props,
    cl_int* errcode_ret) {

    TRACE_FUNCTION("context", (uintptr_t)context);
    LOG_API_CALL("context = %p, sema_props = %p, errcode_ret = %p", context,
                 sema_props, errcode_ret);

    cl_int err;
    auto sem =
        cvk_create_semaphore_with_properties_khr(context, sema_props, &err);

    if (errcode_ret != nullptr) {
        *errcode_ret = err;
    }

    return sem;
}

cl_int
clEnqueueWaitSemaphoresKHR(cl_command_queue command_queue,
                           cl_uint num_sema_objects,
                           const cl_semaphore_khr* sema_objects,
                           const cl_semaphore_payload_khr* sema_payload_list,
                           cl_uint num_events_in_wait_list,
                           const cl_event* event_wait_list, cl_event* event) {
    TRACE_FUNCTION("command_queue", (uintptr_t)command_queue);
    LOG_API_CALL("command_queue = %p, num_sema_objects = %u, sema_objects = "
                 "%p, sema_payload_list = %p, num_events_in_wait_list = %u, "
                 "event_wait_list = %p, event = %p",
                 command_queue, num_sema_objects, sema_objects,
                 sema_payload_list, num_events_in_wait_list, event_wait_list,
                 event);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    if (num_sema_objects == 0) {
        return CL_INVALID_VALUE;
    }

    for (cl_uint i = 0; i < num_sema_objects; i++) {
        if (!is_valid_semaphore(sema_objects[i])) {
            return CL_INVALID_SEMAPHORE_KHR;
        }
        auto sem = icd_downcast(sema_objects[i]);
        if (sem->requires_payload() && sema_payload_list == nullptr) {
            return CL_INVALID_VALUE;
        }
        auto queue = icd_downcast(command_queue);
        if (!sem->can_be_used_with_device(queue->device())) {
            return CL_INVALID_COMMAND_QUEUE;
        }
    }

    if (!is_same_context(command_queue, num_events_in_wait_list,
                         event_wait_list)) {
        return CL_INVALID_CONTEXT;
    }

    if (!is_same_context(command_queue, num_sema_objects, sema_objects)) {
        return CL_INVALID_CONTEXT;
    }

    if (!is_valid_event_wait_list(num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    for (cl_uint i = 0; i < num_events_in_wait_list; i++) {
        if (icd_downcast(event_wait_list[i])->terminated()) {
            return CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST;
        }
    }

    return CL_INVALID_OPERATION;
}

cl_int
clEnqueueSignalSemaphoresKHR(cl_command_queue command_queue,
                             cl_uint num_sema_objects,
                             const cl_semaphore_khr* sema_objects,
                             const cl_semaphore_payload_khr* sema_payload_list,
                             cl_uint num_events_in_wait_list,
                             const cl_event* event_wait_list, cl_event* event) {
    TRACE_FUNCTION("command_queue", (uintptr_t)command_queue);
    LOG_API_CALL("command_queue = %p, num_sema_objects = %u, sema_objects = "
                 "%p, sema_payload_list = %p, num_events_in_wait_list = %u, "
                 "event_wait_list = %p, event = %p",
                 command_queue, num_sema_objects, sema_objects,
                 sema_payload_list, num_events_in_wait_list, event_wait_list,
                 event);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    if (num_sema_objects == 0) {
        return CL_INVALID_VALUE;
    }

    for (cl_uint i = 0; i < num_sema_objects; i++) {
        if (!is_valid_semaphore(sema_objects[i])) {
            return CL_INVALID_SEMAPHORE_KHR;
        }
        auto sem = icd_downcast(sema_objects[i]);
        if (sem->requires_payload() && sema_payload_list == nullptr) {
            return CL_INVALID_VALUE;
        }
        auto queue = icd_downcast(command_queue);
        if (!sem->can_be_used_with_device(queue->device())) {
            return CL_INVALID_COMMAND_QUEUE;
        }
    }

    if (!is_same_context(command_queue, num_events_in_wait_list,
                         event_wait_list)) {
        return CL_INVALID_CONTEXT;
    }

    if (!is_same_context(command_queue, num_sema_objects, sema_objects)) {
        return CL_INVALID_CONTEXT;
    }

    if (!is_valid_event_wait_list(num_events_in_wait_list, event_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    for (cl_uint i = 0; i < num_events_in_wait_list; i++) {
        if (icd_downcast(event_wait_list[i])->terminated()) {
            return CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST;
        }
    }

    return CL_INVALID_OPERATION;
}

cl_int clGetSemaphoreInfoKHR(const cl_semaphore_khr sema_object,
                             cl_semaphore_info_khr param_name,
                             size_t param_value_size, void* param_value,
                             size_t* param_value_size_ret) {
    TRACE_FUNCTION("sema_object", (uintptr_t)sema_object);
    LOG_API_CALL("sema_object = %p, param_name = %x, param_value_size = %zu, "
                 "param_value = %p, param_value_size_ret = %p",
                 sema_object, param_name, param_value_size, param_value,
                 param_value_size_ret);

    cl_int ret = CL_SUCCESS;
    size_t ret_size = 0;
    const void* copy_ptr = nullptr;
    cl_context val_context;
    cl_uint val_uint;
    cl_semaphore_type_khr val_semaphore_type;
    cl_semaphore_payload_khr val_semaphore_payload;

    auto sem = icd_downcast(sema_object);

    if (!is_valid_semaphore(sem)) {
        return CL_INVALID_SEMAPHORE_KHR;
    }

    switch (param_name) {
    case CL_SEMAPHORE_CONTEXT_KHR:
        val_context = sem->context();
        copy_ptr = &val_context;
        ret_size = sizeof(val_context);
        break;
    case CL_SEMAPHORE_REFERENCE_COUNT_KHR:
        val_uint = sem->refcount();
        copy_ptr = &val_uint;
        ret_size = sizeof(val_uint);
        break;
    case CL_SEMAPHORE_TYPE_KHR:
        val_semaphore_type = sem->type();
        copy_ptr = &val_semaphore_type;
        ret_size = sizeof(val_semaphore_type);
        break;
    case CL_SEMAPHORE_PAYLOAD_KHR:
        val_semaphore_payload = sem->payload();
        copy_ptr = &val_semaphore_payload;
        ret_size = sizeof(val_semaphore_payload);
        break;
    case CL_SEMAPHORE_PROPERTIES_KHR:
        copy_ptr = sem->properties().data();
        ret_size =
            sem->properties().size() * sizeof(cl_semaphore_properties_khr);
        break;
    case CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR:
        copy_ptr = sem->devices().data();
        ret_size = sem->devices().size() * sizeof(cl_device_id);
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

cl_int clReleaseSemaphoreKHR(cl_semaphore_khr sema_object) {
    TRACE_FUNCTION("sema_object", (uintptr_t)sema_object);
    LOG_API_CALL("sema_object = %p", sema_object);

    if (!is_valid_semaphore(sema_object)) {
        return CL_INVALID_SEMAPHORE_KHR;
    }

    icd_downcast(sema_object)->release();

    return CL_SUCCESS;
}

cl_int clRetainSemaphoreKHR(cl_semaphore_khr sema_object) {
    TRACE_FUNCTION("sema_object", (uintptr_t)sema_object);
    LOG_API_CALL("sema_object = %p", sema_object);

    if (!is_valid_semaphore(sema_object)) {
        return CL_INVALID_SEMAPHORE_KHR;
    }

    icd_downcast(sema_object)->retain();

    return CL_SUCCESS;
}

// clang-format off
cl_icd_dispatch gDispatchTable = {
    // OpenCL 1.0
    clGetPlatformIDs,
    clGetPlatformInfo,
    clGetDeviceIDs,
    clGetDeviceInfo,
    clCreateContext,
    clCreateContextFromType,
    clRetainContext,
    clReleaseContext,
    clGetContextInfo,
    clCreateCommandQueue,
    clRetainCommandQueue,
    clReleaseCommandQueue,
    clGetCommandQueueInfo,
    clSetCommandQueueProperty,
    clCreateBuffer,
    clCreateImage2D,
    clCreateImage3D,
    clRetainMemObject,
    clReleaseMemObject,
    clGetSupportedImageFormats,
    clGetMemObjectInfo,
    clGetImageInfo,
    clCreateSampler,
    clRetainSampler,
    clReleaseSampler,
    clGetSamplerInfo,
    clCreateProgramWithSource,
    clCreateProgramWithBinary,
    clRetainProgram,
    clReleaseProgram,
    clBuildProgram,
    clUnloadCompiler,
    clGetProgramInfo,
    clGetProgramBuildInfo,
    clCreateKernel,
    clCreateKernelsInProgram,
    clRetainKernel,
    clReleaseKernel,
    clSetKernelArg,
    clGetKernelInfo,
    clGetKernelWorkGroupInfo,
    clWaitForEvents,
    clGetEventInfo,
    clRetainEvent,
    clReleaseEvent,
    clGetEventProfilingInfo,
    clFlush,
    clFinish,
    clEnqueueReadBuffer,
    clEnqueueWriteBuffer,
    clEnqueueCopyBuffer,
    clEnqueueReadImage,
    clEnqueueWriteImage,
    clEnqueueCopyImage,
    clEnqueueCopyImageToBuffer,
    clEnqueueCopyBufferToImage,
    clEnqueueMapBuffer,
    clEnqueueMapImage,
    clEnqueueUnmapMemObject,
    clEnqueueNDRangeKernel,
    clEnqueueTask,
    clEnqueueNativeKernel,
    clEnqueueMarker,
    clEnqueueWaitForEvents,
    clEnqueueBarrier,
    clGetExtensionFunctionAddress,
    nullptr, // clCreateFromGLBuffer;
    nullptr, // clCreateFromGLTexture2D;
    nullptr, // clCreateFromGLTexture3D;
    nullptr, // clCreateFromGLRenderbuffer;
    nullptr, // clGetGLObjectInfo;
    nullptr, // clGetGLTextureInfo;
    nullptr, // clEnqueueAcquireGLObjects;
    nullptr, // clEnqueueReleaseGLObjects;
    nullptr, // clGetGLContextInfoKHR;

    nullptr, // clGetDeviceIDsFromD3D10KHR;
    nullptr, // clCreateFromD3D10BufferKHR;
    nullptr, // clCreateFromD3D10Texture2DKHR;
    nullptr, // clCreateFromD3D10Texture3DKHR;
    nullptr, // clEnqueueAcquireD3D10ObjectsKHR;
    nullptr, // clEnqueueReleaseD3D10ObjectsKHR;

    // OpenCL 1.1
    clSetEventCallback,
    clCreateSubBuffer,
    clSetMemObjectDestructorCallback,
    clCreateUserEvent,
    clSetUserEventStatus,
    clEnqueueReadBufferRect,
    clEnqueueWriteBufferRect,
    clEnqueueCopyBufferRect,

    /* cl_ext_device_fission */
    nullptr, // clCreateSubDevicesEXT;
    nullptr, // clRetainDeviceEXT;
    nullptr, // clReleaseDeviceEXT;

    /* cl_khr_gl_event */
    nullptr, // clCreateEventFromGLsyncKHR;

    // OpenCL 1.2
    clCreateSubDevices,
    clRetainDevice,
    clReleaseDevice,
    clCreateImage,
    clCreateProgramWithBuiltInKernels,
    clCompileProgram,
    clLinkProgram,
    clUnloadPlatformCompiler,
    clGetKernelArgInfo,
    clEnqueueFillBuffer,
    clEnqueueFillImage,
    clEnqueueMigrateMemObjects,
    clEnqueueMarkerWithWaitList,
    clEnqueueBarrierWithWaitList,
    clGetExtensionFunctionAddressForPlatform,
    nullptr, // clCreateFromGLTexture;

    /* cl_khr_d3d11_sharing */
    nullptr, // clGetDeviceIDsFromD3D11KHR;
    nullptr, // clCreateFromD3D11BufferKHR;
    nullptr, // clCreateFromD3D11Texture2DKHR;
    nullptr, // clCreateFromD3D11Texture3DKHR;
    nullptr, // clCreateFromDX9MediaSurfaceKHR;
    nullptr, // clEnqueueAcquireD3D11ObjectsKHR;
    nullptr, // clEnqueueReleaseD3D11ObjectsKHR;

    /* cl_khr_dx9_media_sharing */
    nullptr, // clGetDeviceIDsFromDX9MediaAdapterKHR;
    nullptr, // clEnqueueAcquireDX9MediaSurfacesKHR;
    nullptr, // clEnqueueReleaseDX9MediaSurfacesKHR;

    /* cl_khr_egl_image */
    nullptr, // clCreateFromEGLImageKHR;
    nullptr, // clEnqueueAcquireEGLObjectsKHR;
    nullptr, // clEnqueueReleaseEGLObjectsKHR;

    /* cl_khr_egl_event */
    nullptr, // clCreateEventFromEGLSyncKHR;

    /* OpenCL 2.0 */
    clCreateCommandQueueWithProperties,
    clCreatePipe,
    clGetPipeInfo,
    clSVMAlloc,
    clSVMFree,
    clEnqueueSVMFree,
    clEnqueueSVMMemcpy,
    clEnqueueSVMMemFill,
    clEnqueueSVMMap,
    clEnqueueSVMUnmap,
    clCreateSamplerWithProperties,
    clSetKernelArgSVMPointer,
    clSetKernelExecInfo,

    /* cl_khr_sub_groups */
    clGetKernelSubGroupInfo,

    /* OpenCL 2.1 */
    clCloneKernel,
    clCreateProgramWithIL,
    clEnqueueSVMMigrateMem,
    clGetDeviceAndHostTimer,
    clGetHostTimer,
    clGetKernelSubGroupInfo,
    clSetDefaultDeviceCommandQueue,

    /* OpenCL 2.2 */
    clSetProgramReleaseCallback,
    clSetProgramSpecializationConstant,

    /* OpenCL 3.0 */
    clCreateBufferWithProperties,
    clCreateImageWithProperties,
    clSetContextDestructorCallback,
};
// clang-format on

cl_int CLVK_API_CALL clIcdGetPlatformIDsKHR(cl_uint num_entries,
                                            cl_platform_id* platforms,
                                            cl_uint* num_platforms) {
    auto state = get_or_init_global_state();

    TRACE_FUNCTION("num_entries", num_entries);
    LOG_API_CALL("num_entries = %u, platforms = %p, num_platforms = %p",
                 num_entries, platforms, num_platforms);

    return cvk_get_platform_ids(state, num_entries, platforms, num_platforms);
}

cl_int CLVK_API_CALL clGetKernelSuggestedLocalWorkSizeKHR(
    cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
    const size_t* global_work_offset, const size_t* global_work_size,
    size_t* suggested_local_work_size) {

    TRACE_FUNCTION("command_queue", (uintptr_t)command_queue, "kernel",
                   (uintptr_t)kernel);
    LOG_API_CALL(
        "command_queue = %p, kernel = %p, work_dim = %u, global_work_offset = "
        "%p, global_work_size = %p, suggested_local_work_size = %p",
        command_queue, kernel, work_dim, global_work_offset, global_work_size,
        suggested_local_work_size);

    if (!is_valid_command_queue(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    if (!is_valid_kernel(kernel)) {
        return CL_INVALID_KERNEL;
    }

    if (!is_same_context(command_queue, kernel)) {
        return CL_INVALID_CONTEXT;
    }

    // TODO CL_INVALID_PROGRAM_EXECUTABLE if there is no successfully built
    // program executable available for kernel for the device associated with
    // command_queue.
    // TODO CL_INVALID_KERNEL_ARGS if all argument values for kernel have not
    // been set.
    // TODO CL_MISALIGNED_SUB_BUFFER_OFFSET if a sub-buffer object is set as an
    // argument to kernel and the offset specified when the sub-buffer object
    // was created is not aligned to CL_DEVICE_MEM_BASE_ADDR_ALIGN for the
    // device associated with command_queue.
    // TODO CL_INVALID_IMAGE_SIZE if an image object is set as an argument to
    // kernel and the image dimensions are not supported by device associated
    // with command_queue.
    // TODO CL_IMAGE_FORMAT_NOT_SUPPORTED if an image object is set as an
    // argument to kernel and the image format is not supported by the device
    // associated with command_queue.
    // TODO CL_INVALID_OPERATION if an SVM pointer is set as an argument to
    // kernel and the device associated with command_queue does not support SVM
    // or the required SVM capabilities for the SVM pointer.

    if ((work_dim < 1) ||
        (work_dim >
         icd_downcast(command_queue)->device()->max_work_item_dimensions())) {
        return CL_INVALID_WORK_DIMENSION;
    }

    if (global_work_size == nullptr) {
        return CL_INVALID_GLOBAL_WORK_SIZE;
    }

    for (cl_uint i = 0; i < work_dim; i++) {
        if (global_work_size[i] == 0) {
            return CL_INVALID_GLOBAL_WORK_SIZE;
        }
    }

    // TODO CL_INVALID_GLOBAL_WORK_SIZE if any of the values specified in
    // global_work_size exceed the maximum value representable by size_t on the
    // device associated with command_queue.
    // TODO CL_INVALID_GLOBAL_OFFSET if the value specified in global_work_size
    // plus the corresponding value in global_work_offset for dimension exceeds
    // the maximum value representable by size_t on the device associated with
    // command_queue.

    cvk_ndrange ndrange(work_dim, global_work_offset, global_work_size,
                        nullptr);

    icd_downcast(command_queue)
        ->device()
        ->select_work_group_size(ndrange.gws, ndrange.lws);

    for (cl_uint i = 0; i < work_dim; i++) {
        suggested_local_work_size[i] = ndrange.lws[i];
    }

    return CL_SUCCESS;
}
