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
#include <cstring>
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

std::vector<char> globalBuffer;

static void releaseStdout(int fd) {
    fflush(stdout);
    dup2(fd, fileno(stdout));
    close(fd);
}

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
    if (f == nullptr)
        return nullptr;

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
        if (!m_path.empty())
            std::filesystem::remove_all(m_path.c_str());
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

void printf_callback(const char* buffer, size_t len, size_t complete,
                     void* user_data) {
    auto org_buf_size = globalBuffer.size();
    // Calculate how much data we can fit
    size_t space_available = globalBuffer.capacity() - org_buf_size;
    size_t data_to_copy = (len <= space_available) ? len : space_available;
    // Copy data into the buffer (up to the available space)
    if (space_available > len) {
        globalBuffer.resize(org_buf_size + len, '\0');
        memcpy(globalBuffer.data() + org_buf_size, buffer, data_to_copy);
    }

    if (complete || (space_available < len && space_available >= 1)) {
        globalBuffer.emplace_back('\0');
    }
}

std::vector<cl_context_properties>& setup_arm_printf_test(long int buff_size) {
    globalBuffer.clear();
    globalBuffer.shrink_to_fit();
    globalBuffer.reserve(buff_size);
    static std::vector<cl_context_properties> properties = {
        CL_PRINTF_CALLBACK_ARM,
        (cl_context_properties)printf_callback,
        CL_PRINTF_BUFFERSIZE_ARM,
        buff_size,
    };
    return properties;
}

TEST_F(WithCommandQueue, SimplePrintf) {
    temp_folder_deletion temp;
    stdoutFileName = getStdoutFileName(temp);

    int fd;
    ASSERT_TRUE(getStdout(fd));

    const char message[] = "Hello World!";
    char source[512];
    sprintf(source, "kernel void test_printf() { printf(\"%s\");}", message);
    auto kernel = CreateKernel(source, "test_printf");

    size_t gws = 1;
    size_t lws = 1;
    EnqueueNDRangeKernel(kernel, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
    Finish();

    releaseStdout(fd);
    auto printf_buffer = getStdoutContent();
    ASSERT_NE(printf_buffer, nullptr);

    ASSERT_STREQ(printf_buffer, message);
}

TEST_F(WithPrintfEnabled, TooLongPrintf) {
    auto props = setup_arm_printf_test(44);
    SetUpWithContextProperties(props.data());
    // We only get the first 2 prints because the buffer is too small to get
    // the last one.
    const char* message = "get_global_id(0) = 0\nget_global_id(1) = 0\n";

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
    // Reset the buffer if complete, otherwise keep the remaining part
    ASSERT_STREQ(globalBuffer.data(), message);
}

TEST_F(WithPrintfEnabled, TooLongPrintf2) {
    auto props = setup_arm_printf_test(46);
    SetUpWithContextProperties(props.data());
    const char* message = "get_global_id(0) = 0\nget_global_id(1) = 0\n";
    // We only get the first 2 prints because the buffer is too small to get
    // the last one.

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
    // Reset the buffer if complete, otherwise keep the remaining part
    ASSERT_STREQ(globalBuffer.data(), message);
}

TEST_F(WithPrintfEnabled, PrintfMissingLengthModifier) {
    auto props = setup_arm_printf_test(24);
    SetUpWithContextProperties(props.data());
    char source[512];
    const char message[] = "1,2,3,4";
    sprintf(source, "kernel void test_printf() { printf(\"%s\");}", message);

    sprintf(source,
            "kernel void test_printf() { printf(\"%%v4u\", (uint4)(%s));}",
            message);
    auto kernel = CreateKernel(source, "test_printf");

    size_t gws = 1;
    size_t lws = 1;
    EnqueueNDRangeKernel(kernel, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
    Finish();

    ASSERT_STREQ(globalBuffer.data(), message);
}

#endif
