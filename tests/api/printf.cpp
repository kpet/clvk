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

#ifdef CLVK_UNIT_TESTING_ENABLED

#include "testcl.hpp"
#include "unit.hpp"
#include "utils.hpp"

#include <filesystem>

#ifdef __APPLE__
#include <unistd.h>
#endif

#ifdef WIN32
#include <Windows.h>
#include <io.h>
#endif

static std::string stdoutFileName;

#define BUFFER_SIZE 1024
static char stdoutBuffer[BUFFER_SIZE];

static void releaseStdout(int fd) {
    fflush(stdout);
    dup2(fd, fileno(stdout));
    close(fd);
}

#define RETURN_ON_FAILURE(err, ret)                                            \
    do {                                                                       \
        if (err != CL_SUCCESS) {                                               \
            printf("%s:%d Error after CL call: %d (%s)\n", __FILE__, __LINE__, \
                   err, cl_code_to_string(err));                               \
            return ret;                                                        \
        }                                                                      \
    } while (0)

static bool getStdout(int& fd) {
    fd = dup(fileno(stdout));
    if (!freopen(stdoutFileName.c_str(), "w", stdout)) {
        fprintf(stderr, "ERROR!\n");
        releaseStdout(fd);
        return false;
    }
    return true;
}

static char* getStdoutContent() {
    FILE* f;
    memset(stdoutBuffer, 0, BUFFER_SIZE);
    fflush(stdout);
    f = fopen(stdoutFileName.c_str(), "r");
    if (f == nullptr) {
        return nullptr;
    }

    char* ptr = stdoutBuffer;
    do {
        ptr += strlen(ptr);
        ptr = fgets(ptr, BUFFER_SIZE, f);
    } while (ptr != nullptr);
    fclose(f);

    return stdoutBuffer;
}

struct temp_folder_deletion {
    ~temp_folder_deletion() {
        if (!m_path.empty()) {
            std::filesystem::remove_all(m_path.c_str());
        }
    }
    void set_path(std::string path) { m_path = path; }

private:
    std::string m_path;
};

static char* mkdtemp(char* tmpl, size_t size) {
#ifdef WIN32
    if (_mktemp_s(tmpl, size + 1) != 0) {
        return nullptr;
    }

    if (!CreateDirectory(tmpl, nullptr)) {
        return nullptr;
    }

    return tmpl;
#else
    return mkdtemp(tmpl);
#endif
}

static std::string getStdoutFileName(temp_folder_deletion& temp) {
    char template_tmp_dir[] = "clvk-XXXXXX";
    std::filesystem::path prefix(
        mkdtemp(template_tmp_dir, sizeof(template_tmp_dir)));
    std::filesystem::path suffix("stdout_buffer");
    temp.set_path(prefix.string());
    return (prefix / suffix).string();
}

cl_platform_id get_platform_test() {
    cl_platform_id platform;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, nullptr);
    RETURN_ON_FAILURE(err, nullptr);

    size_t platform_name_len;
    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr,
                            &platform_name_len);
    RETURN_ON_FAILURE(err, nullptr);

    std::string platform_name(platform_name_len - 1, ' ');
    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, platform_name_len,
                            const_cast<char*>(platform_name.data()), nullptr);
    RETURN_ON_FAILURE(err, nullptr);

    std::cout << "Platform: " << platform_name << std::endl;

    return platform;
}

cl_device_id get_device_test(cl_platform_id platform) {
    cl_device_id device;
    cl_int err;

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 10, &device, nullptr);
    RETURN_ON_FAILURE(err, nullptr);

    size_t device_name_len;
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &device_name_len);
    RETURN_ON_FAILURE(err, nullptr);

    std::string device_name(device_name_len - 1, ' ');
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, device_name_len,
                          const_cast<char*>(device_name.data()), nullptr);
    RETURN_ON_FAILURE(err, nullptr);

    std::cout << "Device: " << device_name << std::endl;

    return device;
}

std::string captured_output; // Global to store the captured output

/* Define a printf callback function. */
void printf_callback(const char* buffer, size_t len, size_t complete,
                     void* user_data) {
    captured_output.append(buffer, len);
}

void get_command_queue_printf(cl_command_queue& queue, cl_context new_ctx,
                              cl_device_id device) {
    cl_int err;
    queue = clCreateCommandQueue(new_ctx, device, 0, &err);
    ASSERT_CL_SUCCESS(err);
}

void run_kernel_print_context(const char* source, const char* name) {
    cl_int err;
    auto platform = get_platform_test();
    auto device = get_device_test(platform);
    cl_context_properties properties[] = {
        /* Enable a printf callback function for this context. */
        CL_PRINTF_CALLBACK_ARM, (cl_context_properties)printf_callback,

        /* Request a minimum printf buffer size of 4MiB for devices in the
        context that support this extension. */
        CL_PRINTF_BUFFERSIZE_ARM, (cl_context_properties)0x100000,

        CL_CONTEXT_PLATFORM, (cl_context_properties)gPlatform, 0};
    auto new_ctx =
        clCreateContext(properties, 1, &device, nullptr, nullptr, &err);
    ASSERT_CL_SUCCESS(err);

    auto program =
        clCreateProgramWithSource(new_ctx, 1, &source, nullptr, &err);
    EXPECT_CL_SUCCESS(err);

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    EXPECT_CL_SUCCESS(err);

    if (err != CL_SUCCESS) {
        std::cerr << " Failed to build Program " << std::endl;
    }

    auto kernel = clCreateKernel(program, name, &err);
    EXPECT_CL_SUCCESS(err);

    cl_command_queue queue;
    get_command_queue_printf(queue, new_ctx, device);

    size_t gws = 1;
    size_t lws = 1;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &gws, &lws, 0, nullptr,
                           nullptr);
    err = clFinish(queue);
    ASSERT_CL_SUCCESS(err);
}

TEST_F(WithCommandQueue, SimplePrintf) {
    const char message[] = "Hello World!";
    char source[512];
    sprintf(source, "kernel void test_printf() { printf(\"%s\");}", message);
    run_kernel_print_context(source, "test_printf");
}

TEST_F(WithCommandQueue, TooLongPrintf) {
    const char* source = R"(
    kernel void test_printf() {
      for (unsigned i = 0; i < 3; i++){
        printf("get_global_id(%u) = %u\n", i, (unsigned)get_global_id(i));
      }
    }
    )";
    run_kernel_print_context(source, "test_printf");
}

TEST_F(WithCommandQueue, PrintfMissingLengthModifier) {
    const char message[] = "1,2,3,4\n";
    char source[512];
    sprintf(source,
            "kernel void test_printf() { printf(\"%%v4u\", (uint4)(%s));}",
            message);
    run_kernel_print_context(source, "test_printf");
}

TEST_F(WithCommandQueue, TooLongPrintf2) {
    // each print takes 12 bytes (4 for the printf_id, and 2*4 for the 2 integer
    // to print) + 4 for the byte written counter + 8 which are not enough for
    // the third print, but should not cause any issue in clvk
    auto cfg1 =
        CLVK_CONFIG_SCOPED_OVERRIDE(printf_buffer_size, uint32_t, 36, true);

    temp_folder_deletion temp;
    stdoutFileName = getStdoutFileName(temp);

    int fd;
    ASSERT_TRUE(getStdout(fd));

    const char* source = R"(
    kernel void test_printf() {
      for (unsigned i = 0; i < 3; i++){
        printf("get_global_id(%u) = %u\n", i, (unsigned)get_global_id(i));
      }
    }
    )";
    auto kernel = CreateKernel(source, "test_printf");

    size_t gws = 1;
    size_t lws = 1;
    EnqueueNDRangeKernel(kernel, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
    Finish();

    releaseStdout(fd);
    auto printf_buffer = getStdoutContent();
    ASSERT_NE(printf_buffer, nullptr);

    // We only get the first 2 prints because the buffer is too small to get the
    // last one.
    const char* message = "get_global_id(0) = 0\nget_global_id(1) = 0\n";
    ASSERT_STREQ(printf_buffer, message);
}

#endif