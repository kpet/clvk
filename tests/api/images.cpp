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

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#include "testcl.hpp"

TEST_F(WithContext, CreateImageLegacy)
{
    cl_image_format format = { CL_RGBA, CL_UNORM_INT8 };

    size_t width = 97, height = 135, depth = 7;
    size_t row_pitch = 128, slice_pitch = row_pitch * height;
    char data[slice_pitch * depth * 4];

    cl_int err;

    // Create 2D image
    auto im2d = clCreateImage2D(m_context, CL_MEM_READ_WRITE, &format, width,
                                height, row_pitch, data, &err);
    ASSERT_CL_SUCCESS(err);

    size_t qval;
    cl_image_format qfmt;

    // Query its properties
    GetImageInfo(im2d, CL_IMAGE_WIDTH, sizeof(qval), &qval, nullptr);
    EXPECT_EQ(qval, width);
    GetImageInfo(im2d, CL_IMAGE_HEIGHT, sizeof(qval), &qval, nullptr);
    EXPECT_EQ(qval, height);
    GetImageInfo(im2d, CL_IMAGE_ROW_PITCH, sizeof(qval), &qval, nullptr);
    EXPECT_EQ(qval, row_pitch);
    GetImageInfo(im2d, CL_IMAGE_FORMAT, sizeof(qfmt), &qfmt, nullptr);
    EXPECT_EQ(qfmt.image_channel_order, format.image_channel_order);
    EXPECT_EQ(qfmt.image_channel_data_type, format.image_channel_data_type);

    cl_mem_object_type qtype;
    GetMemObjectInfo(im2d, CL_MEM_TYPE, sizeof(qtype), &qtype, nullptr);
    EXPECT_EQ(qtype, CL_MEM_OBJECT_IMAGE2D);
    void *qptr;
    GetMemObjectInfo(im2d, CL_MEM_HOST_PTR, sizeof(qptr), &qptr, nullptr);
    EXPECT_EQ(qptr, static_cast<void*>(data));

    // Do the same for 3D images
    auto im3d = clCreateImage3D(m_context, CL_MEM_READ_WRITE, &format, width,
                                height, depth, row_pitch, slice_pitch, data, &err);
    ASSERT_CL_SUCCESS(err);
    GetImageInfo(im3d, CL_IMAGE_WIDTH, sizeof(qval), &qval, nullptr);
    EXPECT_EQ(qval, width);
    GetImageInfo(im3d, CL_IMAGE_HEIGHT, sizeof(qval), &qval, nullptr);
    EXPECT_EQ(qval, height);
    GetImageInfo(im3d, CL_IMAGE_DEPTH, sizeof(qval), &qval, nullptr);
    EXPECT_EQ(qval, depth);
    GetImageInfo(im3d, CL_IMAGE_ROW_PITCH, sizeof(qval), &qval, nullptr);
    EXPECT_EQ(qval, row_pitch);
    GetImageInfo(im3d, CL_IMAGE_SLICE_PITCH, sizeof(qval), &qval, nullptr);
    EXPECT_EQ(qval, slice_pitch);
    GetImageInfo(im3d, CL_IMAGE_FORMAT, sizeof(qfmt), &qfmt, nullptr);
    EXPECT_EQ(qfmt.image_channel_order, format.image_channel_order);
    EXPECT_EQ(qfmt.image_channel_data_type, format.image_channel_data_type);

    GetMemObjectInfo(im3d, CL_MEM_TYPE, sizeof(qtype), &qtype, nullptr);
    EXPECT_EQ(qtype, CL_MEM_OBJECT_IMAGE3D);
    GetMemObjectInfo(im3d, CL_MEM_HOST_PTR, sizeof(qptr), &qptr, nullptr);
    EXPECT_EQ(qptr, static_cast<void*>(data));

    clReleaseMemObject(im2d);
    clReleaseMemObject(im3d);
}
