// Copyright 2022 The clvk authors.
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

static bool check(cl_int* src, cl_int* dst, size_t size) {
    for (int i = 0; i < size; ++i) {
        if (src[i] != dst[i]) {
            return false;
        }
    }
    return true;
}

TEST_F(WithCommandQueue, SplitRegion) {
    static const char* source = R"(
      kernel void test(global int *src, global int* dst) {
        size_t gidx = get_global_id(0);
        size_t gidy = get_global_id(1);
        size_t gidz = get_global_id(2);
        size_t xsize = get_global_size(0);
        size_t ysize = get_global_size(1);

        size_t gid = gidx + xsize * (gidy + ysize * gidz);

        dst[gid] = src[gid];
      }
    )";

    clvk_override_device_max_compute_work_group_count(gDevice, 4, 4, 4);

    const size_t max_dim_size = 24;
    const size_t max_nb_elems = max_dim_size * max_dim_size * max_dim_size;
    const size_t buffer_size = max_nb_elems * sizeof(cl_int);

    cl_int src_buf[max_nb_elems], dst_buf[max_nb_elems];

    auto program = CreateAndBuildProgram(source, "-cl-std=CL3.0");
    auto src = CreateBuffer(CL_MEM_READ_ONLY, buffer_size);
    auto dst = CreateBuffer(CL_MEM_WRITE_ONLY, buffer_size);

    auto kernel = CreateKernel(program, "test");
    SetKernelArg(kernel, 0, src);
    SetKernelArg(kernel, 1, dst);

    const size_t scenarii[][3] = {
        {10, 12, 11}, {16, 12, 14}, {16, 16, 13}, {16, 16, 16}, {20, 16, 16},
        {19, 22, 16}, {21, 20, 18}, {23, 13, 10}, {23, 24, 12}, {22, 16, 11},
    };
    const int permutations[][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2},
                                   {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
    for (int scenario = 0; scenario < ARRAY_SIZE(scenarii); ++scenario) {
        const size_t* sc = scenarii[scenario];
        size_t nb_elems = sc[0] * sc[1] * sc[2];
        ASSERT_TRUE(sc[0] <= max_dim_size && sc[1] <= max_dim_size &&
                    sc[2] <= max_dim_size);

        for (int perm = 0; perm < ARRAY_SIZE(permutations); ++perm) {
            size_t gid[3] = {sc[permutations[perm][0]],
                             sc[permutations[perm][1]],
                             sc[permutations[perm][2]]};

            for (int elem = 0; elem < nb_elems; ++elem) {
                src_buf[elem] = elem + (scenario << 16) + (perm << 24);
            }

            EnqueueWriteBuffer(src, true, 0, buffer_size, src_buf);

            const size_t lid[] = {3, 3, 3};
            EnqueueNDRangeKernel(kernel, 3, nullptr, gid, lid);

            EnqueueReadBuffer(dst, true, 0, buffer_size, dst_buf);

            bool status = check(src_buf, dst_buf, nb_elems);
            if (!status) {
                clvk_restore_device_properties(gDevice);
            }
            ASSERT_TRUE(status);
        }
    }
    clvk_restore_device_properties(gDevice);
}

#endif
