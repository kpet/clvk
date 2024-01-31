// Copyright 2024 The clvk authors.
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

static const char* program_source = R"(
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
%s__kernel void test(__global int *output)
{
    output[0] = get_num_sub_groups();
}
)";

#ifdef CLVK_UNIT_TESTING_ENABLED
TEST_F(WithCommandQueue, SubgroupSizes) {
    std::vector<size_t> subgroup_sizes;
    size_t raw_size;
    GetDeviceInfo(CL_DEVICE_SUB_GROUP_SIZES_INTEL, 0, nullptr, &raw_size);
    subgroup_sizes.resize(raw_size / sizeof(size_t));
    GetDeviceInfo(CL_DEVICE_SUB_GROUP_SIZES_INTEL, raw_size,
                  subgroup_sizes.data(), nullptr);

    size_t max_work_group_size;
    GetDeviceInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
                  &max_work_group_size, nullptr);

    auto buffer = CreateBuffer(0, sizeof(cl_int));

    auto run = [this](const char* kernel_prefix, size_t subgroup_size,
                      size_t max_work_group_size, cl_mem buffer) {
        cl_int expected_num_sub_groups = max_work_group_size / subgroup_size;
        size_t work_size = expected_num_sub_groups * subgroup_size;
        char source[512];
        sprintf(source, program_source, kernel_prefix);
        auto kernel = CreateKernel(source, "test");
        SetKernelArg(kernel, 0, buffer);
        EnqueueNDRangeKernel(kernel, 1, nullptr, &work_size, &work_size);
        Finish();
        auto result =
            EnqueueMapBuffer<cl_int>(buffer, true, 0, 0, sizeof(cl_int));
#if 0
        printf("\tkernel_prefix '%s' subgroup_size %lu max_work_group_size %lu "
               "result %u expected %u\n",
               kernel_prefix, subgroup_size, max_work_group_size, *result,
               expected_num_sub_groups);
#endif
        ASSERT_EQ(expected_num_sub_groups, *result);
        EnqueueUnmapMemObject(buffer, result);
    };

    for (auto subgroup_size : subgroup_sizes) {
        char attribute[64];
        sprintf(attribute, "__attribute__((intel_reqd_sub_group_size(%lu))) ",
                subgroup_size);
        run(attribute, subgroup_size, max_work_group_size, buffer);
    }
    for (auto subgroup_size : subgroup_sizes) {
        auto cfg = CLVK_CONFIG_SCOPED_OVERRIDE(force_subgroup_size, uint32_t,
                                               subgroup_size, true);
        run("", subgroup_size, max_work_group_size, buffer);
    }
}
#endif
