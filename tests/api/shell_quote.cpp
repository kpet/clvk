// Tests that build options containing shell metacharacters are passed to
// the compiler verbatim: clBuildProgram must succeed with -D macros whose
// values include parentheses, dollar signs, and quotes.

#include <gtest/gtest.h>

#ifdef CLVK_UNIT_TESTING_ENABLED
#include "unit.hpp"

#include <string>
#include <vector>

namespace {

const char* source = R"(
kernel void test_macro(kernel __global uint* out, uint x) {
    out[0] = x;
}
)";

} // namespace

TEST(ShellQuote, BuildOptionsWithMetacharacters) {
    // Every option here contains characters that would break a naive
    // popen() command line.
    const char* options[] = {
        "-D MACRO(x)=#x",
        "-D VALUE=$HOME",
        "-D NAME=\"hello world\"",
        "-D SEMI=a;b",
        "-D PIPE=a|b",
        "-D PAREN=()",
        "-D GLOB=*",
    };
    for (const char* opts : options) {
        cl_int err;
        auto program = clCreateProgramWithSource(context, 1, &source, nullptr, &err);
        ASSERT_CL_SUCCESS(err);
        err = clBuildProgram(program, 1, &device, opts, nullptr, nullptr);
        EXPECT_CL_SUCCESS(err);
        clReleaseProgram(program);
    }
}

#endif // CLVK_UNIT_TESTING_ENABLED
