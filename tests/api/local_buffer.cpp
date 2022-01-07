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

#include "testcl.hpp"

static bool check(cl_int* buf, size_t size, size_t limit) {
    for (size_t i = 0; i < size; i++) {
        if ((i >= limit && buf[i] != 0xffffffff) ||
            (i < limit && (buf[i] != (limit - 1 - i)))) {
            return false;
        }
    }
    return true;
}

TEST_F(WithCommandQueue, LocalBuffer) {
    static const char* source = R"(
      kernel void test(local int *loc, global int *dst, const int limit) {
        int lid = get_local_id(0);
        if (lid < limit) {
          loc[lid] = lid;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < limit) {
          dst[lid] = loc[limit - 1 - lid];
        }
      }
    )";

    const size_t nb_wis = 128;
    const size_t buffer_size = sizeof(cl_int) * nb_wis;

    auto program = CreateAndBuildProgram(source);
    auto dst = CreateBuffer(CL_MEM_WRITE_ONLY, buffer_size);

    auto kernel = CreateKernel(program, "test");
    SetKernelArg(kernel, 0, buffer_size, nullptr);
    SetKernelArg(kernel, 1, dst);

    for (size_t limit : {64, 96, 128}) {
        SetKernelArg(kernel, 2, sizeof(cl_int), &limit);
        cl_int dst_buf[nb_wis];
        memset(dst_buf, 0xff, buffer_size);

        EnqueueWriteBuffer(dst, true, 0, buffer_size, dst_buf);

        EnqueueNDRangeKernel(kernel, 1, nullptr, &nb_wis, &nb_wis);
        Finish();

        EnqueueReadBuffer(dst, true, 0, buffer_size, dst_buf);

        ASSERT_TRUE(check(dst_buf, nb_wis, limit));
    }
}
