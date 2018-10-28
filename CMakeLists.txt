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

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif ()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# OpenCL Headers
include_directories(BEFORE ${PROJECT_SOURCE_DIR}/external/OpenCL-Headers)

# SPIR-V Headers
set(SPIRV_HEADERS_SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/SPIRV-Headers)

# SPIR-V Tools
set(SPIRV_TOOLS_SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/SPIRV-Tools)

set(SPIRV-Headers_SOURCE_DIR ${SPIRV_HEADERS_SOURCE_DIR})

add_subdirectory(${SPIRV_TOOLS_SOURCE_DIR} EXCLUDE_FROM_ALL)

# clspv
add_subdirectory(${PROJECT_SOURCE_DIR}/external/clspv EXCLUDE_FROM_ALL)
set_target_properties(clspv PROPERTIES RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR})

# Vulkan
set(
  VULKAN_IMPLEMENTATION "system"
  CACHE STRING "Select the Vulkan implementation"
)

set(VULKAN_IMPLEMENTATION_OPTIONS
    system # Use the system's Vulkan
    talvos # Use talvos to provide Vulkan
)
set_property(CACHE VULKAN_IMPLEMENTATION PROPERTY STRINGS
    ${VULKAN_IMPLEMENTATION_OPTIONS})

message(STATUS "VULKAN_IMPLEMENTATION = ${VULKAN_IMPLEMENTATION}")

if (${VULKAN_IMPLEMENTATION} STREQUAL talvos)
  set(SPIRV_INCLUDE_DIR ${SPIRV_HEADERS_SOURCE_DIR}/include)
  set(SPIRV_TOOLS_INCLUDE_DIR ${SPIRV_TOOLS_SOURCE_DIR}/include)
  set(SPIRV_TOOLS_LIB "$<TARGET_FILE:SPIRV-Tools>")
  add_subdirectory(${PROJECT_SOURCE_DIR}/external/talvos)
  add_dependencies(talvos SPIRV-Tools)
  set(Vulkan_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/external/Vulkan-Headers/include/)
  set(Vulkan_LIBRARIES talvos talvos-vulkan)
elseif (${VULKAN_IMPLEMENTATION} STREQUAL system)
  find_package(Vulkan)
else ()
  message(FATAL_ERROR "Unknown Vulkan implementation "
                      "'${VULKAN_IMPLEMENTATION}', "
                      "valid values are: "
                      "'${VULKAN_IMPLEMENTATION_OPTIONS}'.")
endif ()

message(STATUS "Vulkan_INCLUDE_DIRS = '${Vulkan_INCLUDE_DIRS}'")
message(STATUS "Vulkan_LIBRARIES = '${Vulkan_LIBRARIES}'")

# libOpenCL
add_subdirectory(${PROJECT_SOURCE_DIR}/src)

# Tests
add_subdirectory(${PROJECT_SOURCE_DIR}/tests)

# OpenCL Conformance tests
option(BUILD_CONFORMANCE_TESTS "Build OpenCL conformance tests")

if (BUILD_CONFORMANCE_TESTS)
set(CLConform_LIBRARIES OpenCL)
add_subdirectory(${PROJECT_SOURCE_DIR}/external/OpenCL-CTS/test_conformance)
endif (BUILD_CONFORMANCE_TESTS)
