# Copyright 2018 The clvk authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_gtest_executable(api_tests
    compiler.cpp
    dependencies.cpp
    enqueue.cpp
    images.cpp
    local_buffer.cpp
    main.cpp
    platform.cpp
    printf.cpp
    profiling.cpp
    simple.cpp
    simple_image.cpp
    simple_ubo.cpp
    split_region.cpp
    subgroup_size.cpp
    workgroup.cpp
    buffer_device_address.cpp
)

if (${CLVK_COMPILER_AVAILABLE})
  add_definitions(-DCOMPILER_AVAILABLE)
endif()

if (${CLVK_VULKAN_IMPLEMENTATION} STREQUAL swiftshader)
  add_definitions(-DUSING_SWIFTSHADER)
endif()

