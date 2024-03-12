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

#if defined(USING_SWIFTSHADER)
#define DISABLED_SWIFTSHADER(X) DISABLED_##X
#else
#define DISABLED_SWIFTSHADER(X) X
#endif

#ifdef COMPILER_AVAILABLE
#define DISABLED_NOCOMPILER(X) X
#else
#define DISABLED_NOCOMPILER(X) DISABLED_##X
#endif

#define CL_TARGET_OPENCL_VERSION 300
#define CL_USE_DEPRECATED_OPENCL_1_0_APIS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define CL_USE_DEPRECATED_OPENCL_2_1_APIS
#define CL_USE_DEPRECATED_OPENCL_2_2_APIS
#include "CL/cl.h"

#ifdef CLVK_UNIT_TESTING_ENABLED
#include "unit.hpp"

#define CLVK_CONFIG_ASSERT_GT(opt, val)                                        \
    ASSERT_GT(clvk_get_config()->opt.value, val)
#define CLVK_CONFIG_ASSERT_EQ(opt, val)                                        \
    ASSERT_EQ(clvk_get_config()->opt.value, val)
#define CLVK_CONFIG_GET(opt) clvk_get_config()->opt.value
#else
#define CLVK_CONFIG_ASSERT_GT(opt, val)
#define CLVK_CONFIG_ASSERT_EQ(opt, val)
#define CLVK_CONFIG_GET(opt)
#endif

// clang-format off
static inline const char* cl_code_to_string(cl_int code) {
#define code_case(X)                                                           \
    case X:                                                                    \
        return #X;
    switch (code) {
    code_case(CL_SUCCESS)
    code_case(CL_DEVICE_NOT_FOUND)
    code_case(CL_DEVICE_NOT_AVAILABLE)
    code_case(CL_COMPILER_NOT_AVAILABLE)
    code_case(CL_MEM_OBJECT_ALLOCATION_FAILURE)
    code_case(CL_OUT_OF_RESOURCES)
    code_case(CL_OUT_OF_HOST_MEMORY)
    code_case(CL_PROFILING_INFO_NOT_AVAILABLE)
    code_case(CL_MEM_COPY_OVERLAP)
    code_case(CL_IMAGE_FORMAT_MISMATCH)
    code_case(CL_IMAGE_FORMAT_NOT_SUPPORTED)
    code_case(CL_BUILD_PROGRAM_FAILURE)
    code_case(CL_MAP_FAILURE)
    code_case(CL_MISALIGNED_SUB_BUFFER_OFFSET)
    code_case(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
    code_case(CL_COMPILE_PROGRAM_FAILURE)
    code_case(CL_LINKER_NOT_AVAILABLE)
    code_case(CL_LINK_PROGRAM_FAILURE)
    code_case(CL_DEVICE_PARTITION_FAILED)
    code_case(CL_KERNEL_ARG_INFO_NOT_AVAILABLE)
    code_case(CL_INVALID_VALUE)
    code_case(CL_INVALID_DEVICE_TYPE)
    code_case(CL_INVALID_PLATFORM)
    code_case(CL_INVALID_DEVICE)
    code_case(CL_INVALID_CONTEXT)
    code_case(CL_INVALID_QUEUE_PROPERTIES)
    code_case(CL_INVALID_COMMAND_QUEUE)
    code_case(CL_INVALID_HOST_PTR)
    code_case(CL_INVALID_MEM_OBJECT)
    code_case(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
    code_case(CL_INVALID_IMAGE_SIZE)
    code_case(CL_INVALID_SAMPLER)
    code_case(CL_INVALID_BINARY)
    code_case(CL_INVALID_BUILD_OPTIONS)
    code_case(CL_INVALID_PROGRAM)
    code_case(CL_INVALID_PROGRAM_EXECUTABLE)
    code_case(CL_INVALID_KERNEL_NAME)
    code_case(CL_INVALID_KERNEL_DEFINITION)
    code_case(CL_INVALID_KERNEL)
    code_case(CL_INVALID_ARG_INDEX)
    code_case(CL_INVALID_ARG_VALUE)
    code_case(CL_INVALID_ARG_SIZE)
    code_case(CL_INVALID_KERNEL_ARGS)
    code_case(CL_INVALID_WORK_DIMENSION)
    code_case(CL_INVALID_WORK_GROUP_SIZE)
    code_case(CL_INVALID_WORK_ITEM_SIZE)
    code_case(CL_INVALID_GLOBAL_OFFSET)
    code_case(CL_INVALID_EVENT_WAIT_LIST)
    code_case(CL_INVALID_EVENT)
    code_case(CL_INVALID_OPERATION)
    code_case(CL_INVALID_GL_OBJECT)
    code_case(CL_INVALID_BUFFER_SIZE)
    code_case(CL_INVALID_MIP_LEVEL)
    code_case(CL_INVALID_GLOBAL_WORK_SIZE)
    code_case(CL_INVALID_PROPERTY)
    code_case(CL_INVALID_IMAGE_DESCRIPTOR)
    code_case(CL_INVALID_COMPILER_OPTIONS)
    code_case(CL_INVALID_LINKER_OPTIONS)
    code_case(CL_INVALID_DEVICE_PARTITION_COUNT)
    }
#undef code_case
    return "Unknown";
}
// clang-format on

#include <chrono>

static inline uint64_t sampleTime() {
    auto now = std::chrono::steady_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(duration)
        .count();
}

extern cl_device_id gDevice;
extern cl_platform_id gPlatform;

#include "gtest/gtest.h"

#define ASSERT_CL_SUCCESS(X) ASSERT_EQ(X, CL_SUCCESS) << cl_code_to_string(X)
#define EXPECT_CL_SUCCESS(X) EXPECT_EQ(X, CL_SUCCESS) << cl_code_to_string(X)

template <typename T> struct holder {
    holder(T obj) : m_obj(obj) {}
    ~holder() {
        if (m_obj != nullptr) {
            deleter();
        }
    }
    void deleter() { assert(false); }
    operator T() { return m_obj; }
    T release() {
        T ret = m_obj;
        m_obj = nullptr;
        return ret;
    }

private:
    T m_obj;
};

template <> inline void holder<cl_mem>::deleter() {
    auto err = clReleaseMemObject(m_obj);
    ASSERT_CL_SUCCESS(err);
}

template <> inline void holder<cl_kernel>::deleter() {
    auto err = clReleaseKernel(m_obj);
    ASSERT_CL_SUCCESS(err);
}

template <> inline void holder<cl_sampler>::deleter() {
    auto err = clReleaseSampler(m_obj);
    ASSERT_CL_SUCCESS(err);
}

template <> inline void holder<cl_program>::deleter() {
    auto err = clReleaseProgram(m_obj);
    ASSERT_CL_SUCCESS(err);
}

template <> inline void holder<cl_event>::deleter() {
    auto err = clReleaseEvent(m_obj);
    ASSERT_CL_SUCCESS(err);
}

template <> inline void holder<cl_command_queue>::deleter() {
    auto err = clReleaseCommandQueue(m_obj);
    ASSERT_CL_SUCCESS(err);
}

template <typename T>
T GetPlatformInfo(cl_platform_id platform, cl_platform_info info) {
    T val;
    auto err = clGetPlatformInfo(platform, info, sizeof(val), &val, nullptr);
    EXPECT_CL_SUCCESS(err);
    return val;
}

static void GetDeviceAndHostTimer(cl_device_id device, cl_ulong* device_ts,
                                  cl_ulong* host_ts) {
    auto err = clGetDeviceAndHostTimer(device, device_ts, host_ts);
    ASSERT_CL_SUCCESS(err);
}

class WithContext : public ::testing::Test {
protected:
    cl_context m_context;

    cl_platform_id platform() const { return gPlatform; }

    void SetUp() override {
        cl_int err;
        m_context =
            clCreateContext(nullptr, 1, &gDevice, nullptr, nullptr, &err);
        ASSERT_CL_SUCCESS(err);
    }

    void TearDown() override {
        cl_int err = clReleaseContext(m_context);
        ASSERT_CL_SUCCESS(err);
    }

    holder<cl_program> CreateProgram(const char* source) {
        cl_int err;

        auto program =
            clCreateProgramWithSource(m_context, 1, &source, nullptr, &err);
        EXPECT_CL_SUCCESS(err);

        return program;
    }

    void BuildProgram(cl_program program, const char* options = nullptr) {
        cl_int err =
            clBuildProgram(program, 1, &gDevice, options, nullptr, nullptr);
        EXPECT_CL_SUCCESS(err);

        if (err != CL_SUCCESS) {
            std::string build_log = GetProgramBuildLog(program);
            printf("Build log:\n%s\n", build_log.c_str());
        }
    }

    holder<cl_program> CreateAndBuildProgram(const char* source,
                                             const char* options = nullptr) {
        auto program = CreateProgram(source);

        BuildProgram(program, options);

        return program;
    }

    holder<cl_program>
    CreateProgramWithBinary(const std::vector<uint8_t>& binary) {
        cl_int err;
        const size_t size = binary.size();
        const unsigned char* data = binary.data();
        cl_int binary_status;
        auto program = clCreateProgramWithBinary(m_context, 1, &gDevice, &size,
                                                 &data, &binary_status, &err);
        EXPECT_CL_SUCCESS(err);
        EXPECT_EQ(binary_status, CL_BUILD_SUCCESS);
        return program;
    }

    holder<cl_program>
    CreateAndBuildProgramWithBinary(const std::vector<uint8_t>& binary,
                                    const char* options = nullptr) {
        auto program = CreateProgramWithBinary(binary);

        BuildProgram(program, options);

        return program;
    }

    void CompileProgram(cl_program program, const char* options = nullptr) {
        cl_int err = clCompileProgram(program, 1, &gDevice, options, 0, nullptr,
                                      nullptr, nullptr, nullptr);
        EXPECT_CL_SUCCESS(err);

        if (err != CL_SUCCESS) {
            std::string build_log = GetProgramBuildLog(program);
            printf("Build log:\n%s\n", build_log.c_str());
        }
    }

    void CompileProgram(cl_program program, const char* options,
                        cl_uint num_input_headers,
                        const cl_program* input_headers,
                        const char** header_include_names) {
        cl_int err = clCompileProgram(program, 1, &gDevice, options,
                                      num_input_headers, input_headers,
                                      header_include_names, nullptr, nullptr);
        EXPECT_CL_SUCCESS(err);

        if (err != CL_SUCCESS) {
            std::string build_log = GetProgramBuildLog(program);
            printf("Build log:\n%s\n", build_log.c_str());
        }
    }

    holder<cl_program> LinkProgram(cl_uint num_input_programs,
                                   cl_program* input_programs,
                                   const char* options = nullptr) {
        cl_int err;
        auto program =
            clLinkProgram(m_context, 1, &gDevice, options, num_input_programs,
                          input_programs, nullptr, nullptr, &err);
        EXPECT_CL_SUCCESS(err);

        if (err != CL_SUCCESS) {
            std::string build_log = GetProgramBuildLog(program);
            printf("Build log:\n%s\n", build_log.c_str());
        }

        return program;
    }

    cl_program_binary_type GetProgramBinaryType(cl_program program) {
        cl_program_binary_type binary_type;
        cl_int err =
            clGetProgramBuildInfo(program, gDevice, CL_PROGRAM_BINARY_TYPE,
                                  sizeof(binary_type), &binary_type, nullptr);
        EXPECT_CL_SUCCESS(err);
        return binary_type;
    }

    std::vector<uint8_t> GetProgramBinary(cl_program program) {
        std::vector<uint8_t> binary;
        size_t binary_size;
        cl_int err =
            clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                             sizeof(binary_size), &binary_size, nullptr);
        EXPECT_CL_SUCCESS(err);
        binary.resize(binary_size);
        unsigned char* data = binary.data();
        err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, binary_size, &data,
                               nullptr);
        EXPECT_CL_SUCCESS(err);
        return binary;
    }

    std::string GetProgramBuildLog(cl_program program) {
        size_t log_size;
        cl_int err = clGetProgramBuildInfo(
            program, gDevice, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        EXPECT_CL_SUCCESS(err);
        std::string build_log;
        build_log.resize(log_size);
        auto data_ptr = const_cast<char*>(build_log.c_str());
        err = clGetProgramBuildInfo(program, gDevice, CL_PROGRAM_BUILD_LOG,
                                    log_size, data_ptr, nullptr);
        EXPECT_CL_SUCCESS(err);
        return build_log;
    }

    holder<cl_kernel> CreateKernel(const char* source, const char* name) {
        auto program = CreateAndBuildProgram(source);

        cl_int err;
        auto kernel = clCreateKernel(program, name, &err);
        EXPECT_CL_SUCCESS(err);

        return kernel;
    }

    holder<cl_kernel> CreateKernel(const char* source, const char* options,
                                   const char* name) {
        auto program = CreateAndBuildProgram(source, options);

        cl_int err;
        auto kernel = clCreateKernel(program, name, &err);
        EXPECT_CL_SUCCESS(err);

        return kernel;
    }

    holder<cl_kernel> CreateKernel(cl_program program, const char* name) {
        cl_int err;
        auto kernel = clCreateKernel(program, name, &err);
        EXPECT_CL_SUCCESS(err);
        return kernel;
    }

    holder<cl_command_queue>
    CreateCommandQueue(cl_device_id device,
                       cl_command_queue_properties properties) {
        cl_int err;
        auto queue = clCreateCommandQueue(m_context, device, properties, &err);
        EXPECT_CL_SUCCESS(err);
        return queue;
    }

    void Finish(cl_command_queue queue) {
        cl_int err = clFinish(queue);
        ASSERT_CL_SUCCESS(err);
    }

    void ReleaseCommandQueue(cl_command_queue queue) {
        cl_int err = clReleaseCommandQueue(queue);
        ASSERT_CL_SUCCESS(err);
    }

    holder<cl_event> CreateUserEvent() {
        cl_int err;
        auto event = clCreateUserEvent(m_context, &err);
        EXPECT_CL_SUCCESS(err);
        return event;
    }

    void SetUserEventStatus(cl_event event, cl_int status) {
        auto err = clSetUserEventStatus(event, status);
        ASSERT_CL_SUCCESS(err);
    }

    holder<cl_mem> CreateBuffer(cl_mem_flags flags, size_t size,
                                void* host_ptr = nullptr) {
        cl_int err;
        auto mem = clCreateBuffer(m_context, flags, size, host_ptr, &err);
        EXPECT_CL_SUCCESS(err);
        return mem;
    }

    holder<cl_mem> CreateSubBuffer(cl_mem buffer, cl_mem_flags flags,
                                   const size_t region_origin,
                                   const size_t region_size) {
        cl_int err;
        cl_buffer_region buffer_region = {region_origin, region_size};
        auto mem = clCreateSubBuffer(
            buffer, flags, CL_BUFFER_CREATE_TYPE_REGION, &buffer_region, &err);
        EXPECT_CL_SUCCESS(err);
        return mem;
    }

    holder<cl_sampler> CreateSampler(cl_bool normalized_coords,
                                     cl_addressing_mode addressing_mode,
                                     cl_filter_mode filter_mode) {
        cl_int err;
        auto sampler = clCreateSampler(m_context, normalized_coords,
                                       addressing_mode, filter_mode, &err);
        EXPECT_CL_SUCCESS(err);
        return sampler;
    }

    holder<cl_mem> CreateImage(cl_mem_flags flags,
                               const cl_image_format* image_format,
                               const cl_image_desc* image_desc,
                               void* host_ptr = nullptr) {
        cl_int err;
        auto mem = clCreateImage(m_context, flags, image_format, image_desc,
                                 host_ptr, &err);
        EXPECT_CL_SUCCESS(err);
        return mem;
    }

    void GetImageInfo(cl_mem image, cl_image_info param_name,
                      size_t param_value_size, void* param_value,
                      size_t* param_value_size_ret) {
        cl_int err = clGetImageInfo(image, param_name, param_value_size,
                                    param_value, param_value_size_ret);
        ASSERT_CL_SUCCESS(err);
    }

    void GetMemObjectInfo(cl_mem mem, cl_mem_info param_name,
                          size_t param_value_size, void* param_value,
                          size_t* param_value_size_ret) {
        cl_int err = clGetMemObjectInfo(mem, param_name, param_value_size,
                                        param_value, param_value_size_ret);
        ASSERT_CL_SUCCESS(err);
    }

    void GetSupportedImageFormats(cl_mem_object_type image_type,
                                  cl_uint num_entries,
                                  cl_image_format* image_formats,
                                  cl_uint* num_image_formats) {
        cl_int err =
            clGetSupportedImageFormats(m_context, 0, image_type, num_entries,
                                       image_formats, num_image_formats);
        ASSERT_CL_SUCCESS(err);
    }

    void SetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size,
                      const void* arg_value) {
        cl_int err = clSetKernelArg(kernel, arg_index, arg_size, arg_value);
        ASSERT_CL_SUCCESS(err);
    }

    void SetKernelArg(cl_kernel kernel, cl_uint arg_index, cl_mem memobj) {
        SetKernelArg(kernel, arg_index, sizeof(cl_mem), &memobj);
    }

    void SetKernelArg(cl_kernel kernel, cl_uint arg_index, cl_sampler sampler) {
        SetKernelArg(kernel, arg_index, sizeof(cl_sampler), &sampler);
    }

    void SetKernelArg(cl_kernel kernel, cl_uint arg_index, cl_int* val) {
        SetKernelArg(kernel, arg_index, sizeof(*val), val);
    }

    void SetKernelArg(cl_kernel kernel, cl_uint arg_index, cl_uint* val) {
        SetKernelArg(kernel, arg_index, sizeof(*val), val);
    }
};

class WithCommandQueue : public WithContext {
protected:
    cl_command_queue m_queue;

    cl_device_id device() const { return gDevice; }

    void SetUpQueue(cl_command_queue_properties properties) {
#ifndef COMPILER_AVAILABLE
        GTEST_SKIP();
#endif
        WithContext::SetUp();
        auto queue = CreateCommandQueue(device(), properties);
        m_queue = queue.release();
    }

    void SetUp() override { SetUpQueue(0); }

    void TearDown() override {
#ifdef COMPILER_AVAILABLE
        ReleaseCommandQueue(m_queue);
        WithContext::TearDown();
#endif
    }

    using WithContext::Finish;
    void Finish() { Finish(m_queue); }

    void Flush() {
        cl_int err = clFlush(m_queue);
        ASSERT_CL_SUCCESS(err);
    }

    void WaitForEvents(cl_uint num_events, const cl_event* event_list) {
        auto err = clWaitForEvents(num_events, event_list);
        ASSERT_CL_SUCCESS(err);
    }

    void WaitForEvent(cl_event event) { WaitForEvents(1, &event); }

    void GetEventProfilingInfo(cl_event event, cl_profiling_info param_name,
                               size_t param_value_size, void* param_value,
                               size_t* param_value_size_ret) {
        cl_int err =
            clGetEventProfilingInfo(event, param_name, param_value_size,
                                    param_value, param_value_size_ret);
        ASSERT_CL_SUCCESS(err);
    }

    void GetEventInfo(cl_event event, cl_event_info param_name,
                      size_t param_value_size, void* param_value,
                      size_t* param_value_size_ret) {
        cl_int err = clGetEventInfo(event, param_name, param_value_size,
                                    param_value, param_value_size_ret);
        ASSERT_CL_SUCCESS(err);
    }

    template <typename T>
    void GetEventInfo(cl_event event, cl_event_info param_name, T* out_val) {
        GetEventInfo(event, param_name, sizeof(T), out_val, nullptr);
    }

    void GetEventProfilingInfo(cl_event event, cl_profiling_info param_name,
                               cl_ulong* val_ret) {
        GetEventProfilingInfo(event, param_name, sizeof(*val_ret), val_ret,
                              nullptr);
    }

    void EnqueueNDRangeKernel(cl_kernel kernel, cl_uint work_dim,
                              const size_t* global_work_offset,
                              const size_t* global_work_size,
                              const size_t* local_work_size,
                              cl_uint num_events_in_wait_list,
                              const cl_event* event_wait_list,
                              cl_event* event) {
        auto err = clEnqueueNDRangeKernel(
            m_queue, kernel, work_dim, global_work_offset, global_work_size,
            local_work_size, num_events_in_wait_list, event_wait_list, event);
        ASSERT_CL_SUCCESS(err);
    }

    void EnqueueNDRangeKernel(cl_kernel kernel, cl_uint work_dim,
                              const size_t* global_work_offset,
                              const size_t* global_work_size,
                              const size_t* local_work_size) {
        EnqueueNDRangeKernel(kernel, work_dim, global_work_offset,
                             global_work_size, local_work_size, 0, nullptr,
                             nullptr);
    }

    template <typename T>
    T* EnqueueMapBuffer(cl_mem buffer, cl_bool blocking_map,
                        cl_map_flags map_flags, size_t offset, size_t size,
                        cl_uint num_events_in_wait_list,
                        const cl_event* event_wait_list, cl_event* event) {
        cl_int err;
        auto ptr = clEnqueueMapBuffer(m_queue, buffer, blocking_map, map_flags,
                                      offset, size, num_events_in_wait_list,
                                      event_wait_list, event, &err);
        EXPECT_CL_SUCCESS(err);
        return static_cast<T*>(ptr);
    }

    template <typename T>
    T* EnqueueMapBuffer(cl_mem buffer, cl_bool blocking_map,
                        cl_map_flags map_flags, size_t offset, size_t size) {
        return EnqueueMapBuffer<T>(buffer, blocking_map, map_flags, offset,
                                   size, 0, nullptr, nullptr);
    }

    template <typename T>
    T* EnqueueMapImage(cl_mem image, cl_bool blocking_map,
                       cl_map_flags map_flags, const size_t* origin,
                       const size_t* region, size_t* image_row_pitch,
                       size_t* image_slice_pitch,
                       cl_uint num_events_in_wait_list,
                       const cl_event* event_wait_list, cl_event* event) {
        cl_int err;
        auto ptr = clEnqueueMapImage(m_queue, image, blocking_map, map_flags,
                                     origin, region, image_row_pitch,
                                     image_slice_pitch, num_events_in_wait_list,
                                     event_wait_list, event, &err);
        EXPECT_CL_SUCCESS(err);
        return static_cast<T*>(ptr);
    }

    template <typename T>
    T* EnqueueMapImage(cl_mem image, cl_bool blocking_map,
                       cl_map_flags map_flags, const size_t* origin,
                       const size_t* region, size_t* image_row_pitch,
                       size_t* image_slice_pitch) {
        return EnqueueMapImage<T>(image, blocking_map, map_flags, origin,
                                  region, image_row_pitch, image_slice_pitch, 0,
                                  nullptr, nullptr);
    }

    void EnqueueUnmapMemObject(cl_mem memobj, void* mapped_ptr,
                               cl_uint num_events_in_wait_list,
                               const cl_event* event_wait_list,
                               cl_event* event) {
        auto err = clEnqueueUnmapMemObject(m_queue, memobj, mapped_ptr,
                                           num_events_in_wait_list,
                                           event_wait_list, event);
        ASSERT_CL_SUCCESS(err);
    }

    void EnqueueUnmapMemObject(cl_mem memobj, void* mapped_ptr) {
        EnqueueUnmapMemObject(memobj, mapped_ptr, 0, nullptr, nullptr);
    }

    void EnqueueReadBuffer(cl_mem buffer, cl_bool blocking_read, size_t offset,
                           size_t size, void* ptr,
                           cl_uint num_events_in_wait_list,
                           const cl_event* event_wait_list, cl_event* event) {
        auto err = clEnqueueReadBuffer(m_queue, buffer, blocking_read, offset,
                                       size, ptr, num_events_in_wait_list,
                                       event_wait_list, event);
        ASSERT_CL_SUCCESS(err);
    }

    void EnqueueReadBuffer(cl_mem buffer, cl_bool blocking_read, size_t offset,
                           size_t size, void* ptr) {
        EnqueueReadBuffer(buffer, blocking_read, offset, size, ptr, 0, nullptr,
                          nullptr);
    }

    void EnqueueWriteBuffer(cl_mem buffer, cl_bool blocking_write,
                            size_t offset, size_t size, const void* ptr,
                            cl_uint num_events_in_wait_list,
                            const cl_event* event_wait_list, cl_event* event) {
        auto err = clEnqueueWriteBuffer(m_queue, buffer, blocking_write, offset,
                                        size, ptr, num_events_in_wait_list,
                                        event_wait_list, event);
        ASSERT_CL_SUCCESS(err);
    }

    void EnqueueWriteBuffer(cl_mem buffer, cl_bool blocking_write,
                            size_t offset, size_t size, const void* ptr) {
        EnqueueWriteBuffer(buffer, blocking_write, offset, size, ptr, 0,
                           nullptr, nullptr);
    }

    void EnqueueFillBuffer(cl_mem buffer, const void* pattern,
                           size_t pattern_size, size_t offset, size_t size) {
        auto err = clEnqueueFillBuffer(m_queue, buffer, pattern, pattern_size,
                                       offset, size, 0, nullptr, nullptr);
        ASSERT_CL_SUCCESS(err);
    }

    void EnqueueFillBuffer(cl_mem buffer, const void* pattern,
                           size_t pattern_size, size_t size) {
        auto err = clEnqueueFillBuffer(m_queue, buffer, pattern, pattern_size,
                                       0, size, 0, nullptr, nullptr);
        ASSERT_CL_SUCCESS(err);
    }

    void EnqueueReadImage(cl_mem image, cl_bool blocking_read,
                          const size_t* origin, const size_t* region,
                          size_t row_pitch, size_t slice_pitch, void* ptr,
                          cl_uint num_events_in_wait_list,
                          const cl_event* event_wait_list, cl_event* event) {
        auto err = clEnqueueReadImage(
            m_queue, image, blocking_read, origin, region, row_pitch,
            slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event);
        ASSERT_CL_SUCCESS(err);
    }

    void EnqueueReadImage(cl_mem image, cl_bool blocking_read,
                          const size_t* origin, const size_t* region,
                          size_t row_pitch, size_t slice_pitch, void* ptr) {
        EnqueueReadImage(image, blocking_read, origin, region, row_pitch,
                         slice_pitch, ptr, 0, nullptr, nullptr);
    }

    void EnqueueWriteImage(cl_mem image, cl_bool blocking_write,
                           const size_t* origin, const size_t* region,
                           size_t input_row_pitch, size_t input_slice_pitch,
                           const void* ptr, cl_uint num_events_in_wait_list,
                           const cl_event* event_wait_list, cl_event* event) {
        auto err = clEnqueueWriteImage(
            m_queue, image, blocking_write, origin, region, input_row_pitch,
            input_slice_pitch, ptr, num_events_in_wait_list, event_wait_list,
            event);
        ASSERT_CL_SUCCESS(err);
    }

    void EnqueueWriteImage(cl_mem image, cl_bool blocking_write,
                           const size_t* origin, const size_t* region,
                           size_t input_row_pitch, size_t input_slice_pitch,
                           const void* ptr) {
        EnqueueWriteImage(image, blocking_write, origin, region,
                          input_row_pitch, input_slice_pitch, ptr, 0, nullptr,
                          nullptr);
    }

    void EnqueueFillImage(cl_mem image, const void* fill_color,
                          const size_t* origin, const size_t* region,
                          cl_uint num_events_in_wait_list,
                          const cl_event* event_wait_list, cl_event* event) {
        auto err =
            clEnqueueFillImage(m_queue, image, fill_color, origin, region,
                               num_events_in_wait_list, event_wait_list, event);
        ASSERT_CL_SUCCESS(err);
    }

    void EnqueueFillImage(cl_mem image, const void* fill_color,
                          const size_t* origin, const size_t* region) {
        EnqueueFillImage(image, fill_color, origin, region, 0, nullptr,
                         nullptr);
    }

    void GetDeviceInfo(cl_device_info param_name, size_t param_value_size,
                       void* param_value, size_t* param_value_size_ret) {
        auto err = clGetDeviceInfo(device(), param_name, param_value_size,
                                   param_value, param_value_size_ret);
        ASSERT_CL_SUCCESS(err);
    }
};

class WithProfiledCommandQueue : public WithCommandQueue {
protected:
    void SetUp() override { SetUpQueue(CL_QUEUE_PROFILING_ENABLE); }
};
