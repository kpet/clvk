// Copyright 2026 The clvk authors.
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

// Tests that build options containing shell metacharacters are passed to
// the compiler verbatim: clBuildProgram must succeed with -D macros whose
// values include parentheses, dollar signs, quotes, and single quotes, and
// the macro must expand to its exact definition at runtime.

#include "testcl.hpp"

#include <gtest/gtest.h>

namespace {

const char* source = R"(
kernel void test_macro(__global uint* out, uint x) {
    out[0] = x;
}
)";

} // namespace

TEST_F(WithContext, BuildOptionsWithMetacharacters) {
    // Every option here contains characters that would break a naive
    // popen() command line. The first is the hashcat pattern that
    // motivated the fix: a single token with parentheses and a hash.
    const char* options[] = {
        "-DXM2S(x)=#x",      "-D MACRO(x)=#x",
        "-D VALUE=$HOME",    "-D NAME=\"hello world\"",
        "-D SEMI=a;b",       "-D PIPE=a|b",
        "-D PAREN=()",       "-D GLOB=*",
        "-D SINGLE='quote'",
    };
    for (const char* opts : options) {
        cl_int err;
        auto program =
            clCreateProgramWithSource(m_context, 1, &source, nullptr, &err);
        ASSERT_CL_SUCCESS(err);
        err = clBuildProgram(program, 1, &gDevice, opts, nullptr, nullptr);
        EXPECT_CL_SUCCESS(err);
        clReleaseProgram(program);
    }
}

TEST_F(WithCommandQueueAndPrintf, StringizeMacroPassedVerbatim) {
    // Same option as above, checking the macro value this time: the kernel
    // stringizes its argument, so a mangled command line shows up in the
    // printf output instead of the expected text.
    const char* src = "kernel void test() { printf(\"%s\", XM2S(hello)); }";
    auto kernel = CreateKernel(src, "-D XM2S(x)=#x", "test");
    size_t gws = 1;
    size_t lws = 1;
    EnqueueNDRangeKernel(kernel, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
    Finish();
    ASSERT_STREQ(m_printf_output.c_str(), "hello");
}
