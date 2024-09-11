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

TEST_F(WithCommandQueueAndPrintf, SimplePrintf) {
    const char message[] = "Hello World!";
    char source[512];
    sprintf(source, "kernel void test_printf() { printf(\"%s\");}", message);
    auto kernel = CreateKernel(source, "test_printf");

    size_t gws = 1;
    size_t lws = 1;
    EnqueueNDRangeKernel(kernel, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
    Finish();

    ASSERT_STREQ(m_printf_output.c_str(), message);
}

TEST_F(WithCommandQueueAndPrintf, SimpleFormatedPrintf) {
    const char* source = "kernel void test_printf() { printf(\"%s\", \"\"); }";
    auto kernel = CreateKernel(source, "test_printf");

    size_t gws = 1;
    size_t lws = 1;
    EnqueueNDRangeKernel(kernel, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
    Finish();

    ASSERT_STREQ(m_printf_output.c_str(), "");
}

TEST_F(WithCommandQueueAndPrintf, TooLongPrintf) {
    // each print takes 12 bytes (4 for the printf_id, and 2*4 for the 2 integer
    // to print) + 4 for the byte written counter
    auto cfg1 =
        CLVK_CONFIG_SCOPED_OVERRIDE(printf_buffer_size, uint32_t, 28, true);

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

    ASSERT_STREQ(m_printf_output.c_str(), message);
}

TEST_F(WithCommandQueueAndPrintf, TooLongPrintf2) {
    // each print takes 12 bytes (4 for the printf_id, and 2*4 for the 2 integer
    // to print) + 4 for the byte written counter + 8 which are not enough for
    // the third print, but should not cause any issue in clvk
    auto cfg1 =
        CLVK_CONFIG_SCOPED_OVERRIDE(printf_buffer_size, uint32_t, 36, true);

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

    ASSERT_STREQ(m_printf_output.c_str(), message);
}

TEST_F(WithCommandQueueAndPrintf, PrintfMissingLengthModifier) {
    const char message[] = "1,2,3,4";
    char source[512];
    sprintf(source,
            "kernel void test_printf() { printf(\"%%v4u\", (uint4)(%s));}",
            message);
    auto kernel = CreateKernel(source, "test_printf");

    size_t gws = 1;
    size_t lws = 1;
    EnqueueNDRangeKernel(kernel, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
    Finish();

    ASSERT_STREQ(m_printf_output.c_str(), message);
}

#endif
