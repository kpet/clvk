// Copyright 2019 The clvk authors.
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

#include "testcl.hpp"

static const char *program_source = R"(
kernel void donothing(int dummy)
{
}
)";

TEST_F(WithProfiledCommandQueue, QueueProfilingTimestampOrderingAndSanity)
{
    // Create kernel
    auto kernel = CreateKernel(program_source, "donothing");

    // Dispatch kernel
    size_t gws = 1;
    size_t lws = 1;

    cl_int dummy = 42;
    SetKernelArg(kernel, 0, &dummy);

    cl_event event;
    EnqueueNDRangeKernel(kernel, 1, nullptr, &gws, &lws, 0, nullptr, &event);

    // Complete execution
    Finish();

    cl_ulong ts_queued, ts_submit, ts_start, ts_end;

    GetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, &ts_queued);
    GetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT, &ts_submit);
    GetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, &ts_start);
    GetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, &ts_end);

    // Check that timestamps are ordered.
    ASSERT_GE(ts_submit, ts_queued);
    ASSERT_GE(ts_start, ts_submit);
    ASSERT_GE(ts_end, ts_start);

    // Check the delay between timestamps is less than a threshold.
    // This is meant to help with catching obvious time base differences.
    auto max_diff = 500*1000*1000;
    ASSERT_LT(ts_submit - ts_queued, max_diff);
    ASSERT_LT(ts_start - ts_submit, max_diff);
    ASSERT_LT(ts_end - ts_start, max_diff);
}
