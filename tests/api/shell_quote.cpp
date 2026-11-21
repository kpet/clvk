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

// Tests that build options containing shell metacharacters are relayed to
// the compiler verbatim: every case builds a kernel whose printf output is
// the exact value of the -D macro, so a mangled command line fails the
// string comparison instead of merely the build.

#include "testcl.hpp"

#include <gtest/gtest.h>

#include <cstdio>

TEST_F(WithCommandQueueAndPrintf, BuildOptionsWithMetacharacters) {
    // Every option here contains characters that would break a naive
    // popen() command line. The first two rows are the hashcat pattern
    // that motivated the fix: a single token with parentheses and a hash.
    // The S1/S2 pair forces argument expansion before stringizing, so the
    // relayed value of object-like macros is printed.
    struct Case {
        const char* options;
        const char* expr;
        const char* expected;
    };
    const Case cases[] = {
        {"-DXM2S(x)=#x", "XM2S(hello)", "hello"},
        {"-D MACRO(x)=#x", "MACRO(hello)", "hello"},
        {"-D VALUE=$HOME -D S1(x)=S2(x) -D S2(x)=#x", "S1(VALUE)", "$HOME"},
        {"-D NAME=\"hello world\" -D S1(x)=S2(x) -D S2(x)=#x", "S1(NAME)",
         "hello world"},
        {"-D SEMI=a;b -D S1(x)=S2(x) -D S2(x)=#x", "S1(SEMI)", "a;b"},
        {"-D PIPE=a|b -D S1(x)=S2(x) -D S2(x)=#x", "S1(PIPE)", "a|b"},
        {"-D PAREN=() -D S1(x)=S2(x) -D S2(x)=#x", "S1(PAREN)", "()"},
        {"-D GLOB=* -D S1(x)=S2(x) -D S2(x)=#x", "S1(GLOB)", "*"},
        {"-D SINGLE='quote' -D S1(x)=S2(x) -D S2(x)=#x", "S1(SINGLE)",
         "'quote'"},
    };
    for (const auto& c : cases) {
        char source[256];
        snprintf(source, sizeof(source),
                 "kernel void test() { printf(\"%%s\", %s); }", c.expr);
        auto kernel = CreateKernel(source, c.options, "test");
        size_t gws = 1;
        size_t lws = 1;
        EnqueueNDRangeKernel(kernel, 1, nullptr, &gws, &lws, 0, nullptr,
                             nullptr);
        Finish();
        ASSERT_STREQ(m_printf_output.c_str(), c.expected)
            << "options: " << c.options;
        m_printf_output.clear();
    }
}
