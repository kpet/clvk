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
project(clvk_test)

set(CLVK_PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/..)

set(CLVK_BUILD_STATIC_TESTS ON CACHE BOOL
    "Set to OFF to disable the build of tests with static OpenCL library")

macro(add_gtest_executable name)
  add_executable(${name} ${ARGN})
  # Use the gtest copy that comes with clspv's version of LLVM
  include_directories(${LLVM_BINARY_DIR}/include)
  include_directories(${LLVM_SOURCE_DIR}/include)
  include_directories(${LLVM_SOURCE_DIR}/utils/unittest/googletest/include/)
  include_directories(${CLVK_PROJECT_SOURCE_DIR}/src)
  include_directories(${Vulkan_INCLUDE_DIRS})

  target_link_libraries(${name} OpenCL llvm_gtest llvm_gtest_main)
  if (BUILD_SHARED_LIBS)
    target_link_libraries(${name} LLVMSupport)
  endif()
  if (CLVK_UNIT_TESTING)
    target_compile_definitions(${name} PUBLIC CLVK_UNIT_TESTING_ENABLED)
  endif()

  set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR})
endmacro()

macro(add_simple_executable name source libraries)
  add_executable(${name} ${source})
  target_link_libraries(${name} ${libraries})
  set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR})
endmacro()

macro(add_simple_static_and_dyn_executable name source)
  add_simple_executable(${name} ${source} OpenCL)
  if (CLVK_BUILD_STATIC_TESTS)
    add_simple_executable("${name}_static" ${source} OpenCL-static)
  endif()
  if(NOT MSVC)
    target_compile_options(${name} PUBLIC -Wall -W -Wextra)
  endif()
endmacro()

if (CLVK_COMPILER_AVAILABLE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/api)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/simple)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/simple-from-il-binary)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/simple-from-llvm-ir-binary)
endif()
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/sha1)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/simple-from-binary)

option(CLVK_BUILD_CONFORMANCE_TESTS "Build OpenCL conformance tests")
if (CLVK_BUILD_CONFORMANCE_TESTS)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/conformance)
endif (CLVK_BUILD_CONFORMANCE_TESTS)
