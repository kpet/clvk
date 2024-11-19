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

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif ()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  option(CLVK_ENABLE_ASSERTIONS "Enable assertions" ON)
else()
  option(CLVK_ENABLE_ASSERTIONS "Enable assertions" OFF)
endif()
message(STATUS "Assertions: ${CLVK_ENABLE_ASSERTIONS}")

if (CLVK_ENABLE_ASSERTIONS)
  if (NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    add_compile_options($<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-UNDEBUG>)
  endif()
endif()

if (CLVK_ENABLE_ASAN)
  add_compile_options(-fsanitize=address)
  add_link_options(-fsanitize=address)
endif()

if (CLVK_ENABLE_TSAN)
  add_compile_options(-fsanitize=thread)
  add_link_options(-fsanitize=thread)
endif()

if (CLVK_ENABLE_UBSAN)
  add_compile_options(-fsanitize=undefined)
  add_link_options(-fsanitize=undefined)
endif()

if(WIN32)
  add_compile_definitions(NOMINMAX)
endif()

# Options
option(CLVK_CLSPV_ONLINE_COMPILER "Use the Clspv C++ API for compilation of kernels")

option(CLVK_COMPILER_AVAILABLE "Enable compiler support" ON)
if (NOT CLVK_COMPILER_AVAILABLE AND CLVK_CLSPV_ONLINE_COMPILER)
  message(FATAL_ERROR "Enabling online compiler support when disabling
  compiler support does not make sense!")
endif()

# ccache
set(CLVK_CCACHE_BUILD OFF CACHE BOOL "Set to ON for a ccache enabled build")
if(CLVK_CCACHE_BUILD AND UNIX)
  message(STATUS "Enabling ccache")
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
      message(STATUS "Using ccache at ${CCACHE_PROGRAM}")
      set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PROGRAM})
  else()
      message(FATAL_ERROR "Unable to find the program ccache. Set CLVK_CCACHE_BUILD to OFF")
  endif()
endif()

include(CheckCXXCompilerFlag)

macro(AddCXXFlagIfSupported flag test)
   CHECK_CXX_COMPILER_FLAG(${flag} ${test})
   if( ${${test}} )
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}")
   endif()
endmacro()

if("Ninja" STREQUAL ${CMAKE_GENERATOR})
    AddCXXFlagIfSupported(-fdiagnostics-color COMPILER_SUPPORTS_COLOR_DIAGNOSTICS)
endif()

# OpenCL Headers
set(OPENCL_HEADERS_SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/OpenCL-Headers
    CACHE STRING "OpenCL-Headers source directory")
include_directories(BEFORE ${OPENCL_HEADERS_SOURCE_DIR})

# SPIR-V Headers
set(SPIRV_HEADERS_SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/SPIRV-Headers CACHE STRING
    "Path to SPIRV headers directory")

# SPIR-V Tools
set(SPIRV_SKIP_TESTS ON)
set(SPIRV_TOOLS_SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/SPIRV-Tools CACHE STRING
    "Path to SPIRV-Tools directory")

set(SPIRV-Headers_SOURCE_DIR ${SPIRV_HEADERS_SOURCE_DIR})

set(CLVK_BUILD_SPIRV_TOOLS ON CACHE BOOL "Set to OFF to disable SPIRV-Tools build")
if (CLVK_BUILD_SPIRV_TOOLS)
  add_subdirectory(${SPIRV_TOOLS_SOURCE_DIR}
                   ${PROJECT_BINARY_DIR}/external/SPIRV-Tools EXCLUDE_FROM_ALL)
endif()

# clspv
if(MSVC)
    add_compile_options(/wd4574 /wd4668)
endif()

if (CLVK_COMPILER_AVAILABLE)
  # clang used to test simple_test_from_il_binary in CI
  set(LLVM_ENABLE_PROJECTS clang CACHE STRING
      "Control which projects are enabled.")
endif()
set(CLSPV_SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/clspv
    CACHE STRING "Clspv source directory")
set(CLSPV_BUILD_SPIRV_DIS OFF)
add_subdirectory(${CLSPV_SOURCE_DIR} ${PROJECT_BINARY_DIR}/external/clspv
                 EXCLUDE_FROM_ALL)
set_target_properties(clspv PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                      ${CMAKE_BINARY_DIR})

option(CLVK_ENABLE_SPIRV_IL "Enable SPIR-V as an intermediate language" ON)
# SPIRV-LLVM-Translator
if (CLVK_COMPILER_AVAILABLE AND CLVK_ENABLE_SPIRV_IL)
  set(LLVM_DIR
      ${CMAKE_BINARY_DIR}/external/clspv/third_party/llvm/lib/cmake/llvm)
  set(LLVM_SPIRV_SOURCE ${PROJECT_SOURCE_DIR}/external/SPIRV-LLVM-Translator CACHE STRING
      "Path to SPIRV-LLVM-Translator directory")
  set(LLVM_EXTERNAL_SPIRV_HEADERS_SOURCE_DIR ${SPIRV_HEADERS_SOURCE_DIR})
  set(LLVM_SPIRV_BUILD_EXTERNAL YES)
  add_subdirectory(${LLVM_SPIRV_SOURCE}
                   ${CMAKE_CURRENT_BINARY_DIR}/external/SPIRV-LLVM-Translator
                   EXCLUDE_FROM_ALL)

  if (CLVK_CLSPV_ONLINE_COMPILER)
    # Include LLVM dependencies for SPIRV-LLVM
    find_package(LLVM ${BASE_LLVM_VERSION} REQUIRED)
    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
    message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
    include_directories(${LLVM_INCLUDE_DIRS})
  endif()
endif()

# Vulkan
set(
  CLVK_VULKAN_IMPLEMENTATION "system"
  CACHE STRING "Select the Vulkan implementation"
)

set(CLVK_VULKAN_IMPLEMENTATION_OPTIONS
    custom # User-provided libraries and includes
    system # Use the system's Vulkan
    loader # Build against a user-provided copy of the loader sources
    swiftshader # Build for a user-provided copy of the switftshader sources
)
set_property(CACHE CLVK_VULKAN_IMPLEMENTATION PROPERTY STRINGS
    ${CLVK_VULKAN_IMPLEMENTATION_OPTIONS})

message(STATUS "CLVK_VULKAN_IMPLEMENTATION = ${CLVK_VULKAN_IMPLEMENTATION}")

if (${CLVK_VULKAN_IMPLEMENTATION} STREQUAL system)
  find_package(Vulkan)
elseif (${CLVK_VULKAN_IMPLEMENTATION} STREQUAL swiftshader)
  find_package(Vulkan)

  set(SWIFTSHADER_WARNINGS_AS_ERRORS OFF)
  set(SWIFTSHADER_BUILD_VULKAN ON)
  set(SWIFTSHADER_BUILD_EGL OFF)
  set(SWIFTSHADER_BUILD_GLESv2 OFF)
  set(SWIFTSHADER_BUILD_GLES_CM OFF)
  set(SWIFTSHADER_BUILD_SAMPLES OFF)
  set(SWIFTSHADER_BUILD_TESTS OFF)
  set(SWIFTSHADER_BUILD_PVR OFF)
  if (CLVK_SWIFTSHADER_DIR)
    message(STATUS "Adding provided CLVK_SWIFTSHADER_DIR
            (${CLVK_SWIFTSHADER_DIR})")
    add_subdirectory(${CLVK_SWIFTSHADER_DIR} EXCLUDE_FROM_ALL)
  else ()
      message(WARNING "No CLVK_SWIFTSHADER_DIR provided")
  endif ()
elseif (${CLVK_VULKAN_IMPLEMENTATION} STREQUAL custom)
    # Nothing to do, trust the user
else ()
  message(FATAL_ERROR "Unknown Vulkan implementation "
                      "'${CLVK_VULKAN_IMPLEMENTATION}', "
                      "valid values are: "
                      "'${CLVK_VULKAN_IMPLEMENTATION_OPTIONS}'.")
endif ()

message(STATUS "Vulkan_INCLUDE_DIRS = '${Vulkan_INCLUDE_DIRS}'")
message(STATUS "Vulkan_LIBRARIES = '${Vulkan_LIBRARIES}'")

# libOpenCL
add_subdirectory(${PROJECT_SOURCE_DIR}/src)

# Tests
option(CLVK_BUILD_TESTS "Build tests" ON)
option(CLVK_UNIT_TESTING "Build unit tests that require vulkan properties to be overriden" OFF)
if (CLVK_BUILD_TESTS)
add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
endif()
