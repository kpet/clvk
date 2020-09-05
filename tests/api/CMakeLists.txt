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


set(BINARY_NAME api_tests)

add_executable(
    ${BINARY_NAME}
    compiler.cpp
    dependencies.cpp
    enqueue.cpp
    images.cpp
    main.cpp
    profiling.cpp
    simple.cpp
    simple_image.cpp
    simple_ubo.cpp
    workgroup.cpp
)

if (${CLVK_VULKAN_IMPLEMENTATION} STREQUAL talvos)
  add_definitions(-DUSING_TALVOS)
endif()

if (${CLVK_VULKAN_IMPLEMENTATION} STREQUAL swiftshader)
  add_definitions(-DUSING_SWIFTSHADER)
endif()

# Use the gtest copy that comes with clspv's version of LLVM
include_directories(${LLVM_BINARY_DIR}/include)
include_directories(${LLVM_SOURCE_DIR}/include)
include_directories(${LLVM_SOURCE_DIR}/utils/unittest/googletest/include/)

target_link_libraries(${BINARY_NAME} OpenCL gtest gtest_main)

set_target_properties(${BINARY_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR})

