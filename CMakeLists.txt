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

cmake_minimum_required(VERSION 3.9)

project(clvk VERSION 0.1 DESCRIPTION "OpenCL implementation on Vulkan" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_BUILD_TYPE RelWithDebInfo)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

include_directories(BEFORE
    ${CMAKE_CURRENT_SOURCE_DIR}/external/OpenCL-Headers)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/clspv EXCLUDE_FROM_ALL)
set_target_properties(clspv PROPERTIES RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR})

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)

option(BUILD_CONFORMANCE_TESTS "Build OpenCL conformance tests")

if (BUILD_CONFORMANCE_TESTS)
set(CLConform_LIBRARIES OpenCL)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/OpenCL-CTS/test_conformance)
endif (BUILD_CONFORMANCE_TESTS)
