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
#include "utils.hpp"
#include "unit.hpp"

#include <filesystem>

static std::string stdoutFileName;

#define BUFFER_SIZE 1024
static char stdoutBuffer[BUFFER_SIZE];

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

static std::string getStdoutFileName(temp_folder_deletion& temp) {
    char template_tmp_dir[] = "clvk-XXXXXX";
    std::filesystem::path prefix(
        cvk_mkdtemp(template_tmp_dir, sizeof(template_tmp_dir)));
    std::filesystem::path suffix("stdout_buffer");
    temp.set_path(prefix.string());
    return (prefix / suffix).string();
}

TEST_F(WithCommandQueue, SimplePrintf) {
    temp_folder_deletion temp;
    stdoutFileName = getStdoutFileName(temp);

    int fd;
    ASSERT_TRUE(getStdout(fd));

    const char message[] = "Hello World!";
    char* source = nullptr;
    asprintf(&source, "kernel void test_printf() { printf(\"%s\");}", message);
    ASSERT_NE(source, nullptr);
    auto kernel = CreateKernel(source, "test_printf");
    free(source);

    size_t gws = 1;
    size_t lws = 1;
    EnqueueNDRangeKernel(kernel, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
    Finish();

    releaseStdout(fd);
    auto printf_buffer = getStdoutContent();
    ASSERT_NE(printf_buffer, nullptr);

    ASSERT_STREQ(printf_buffer, message);
}

TEST_F(WithCommandQueue, TooLongPrintf) {
    clvk_override_printf_buffer_size(24);

    temp_folder_deletion temp;
    stdoutFileName = getStdoutFileName(temp);

    int fd;
    ASSERT_TRUE(getStdout(fd));

    const char* source = R"(
    kernel void test_printf() {
      for (unsigned i = 0; i < 3; i++){
        printf("get_global_id(%u) = %u\n", i, get_global_id(i));
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
