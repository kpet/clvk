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


if(NOT MSVC)
    add_compile_options(-Wall -W -Wextra)
    add_compile_options(-Wno-ignored-attributes)
else()
	set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

if (UNIX AND NOT APPLE)
    set(CMAKE_SHARED_LINKER_FLAGS
        "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
endif()

if (UNIX AND NOT APPLE)
    # Bind global symbol references to definitions within the shared library,
    # to allow the ICD dispatch table to be statically defined.
    set(CMAKE_SHARED_LINKER_FLAGS
        "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-Bsymbolic")
endif()

if (${CLVK_VULKAN_IMPLEMENTATION} STREQUAL swiftshader)
    add_compile_options(-DUSING_SWIFTSHADER=1)
endif()

# Config macros
# These are made available to all targets linking against OpenCL so they can
# correctly use the config in config.def
add_library(clvk-config-definitions INTERFACE)

if (CLVK_CLSPV_ONLINE_COMPILER)
  target_compile_definitions(clvk-config-definitions INTERFACE CLSPV_ONLINE_COMPILER=1)
endif()

if (CLVK_COMPILER_AVAILABLE)
    target_compile_definitions(clvk-config-definitions INTERFACE COMPILER_AVAILABLE=1)
    if (CLVK_ENABLE_SPIRV_IL)
        target_compile_definitions(clvk-config-definitions INTERFACE ENABLE_SPIRV_IL=1)
    endif()
endif()

# Core objects
add_library(OpenCL-objects OBJECT
  api.cpp
  config.cpp
  context.cpp
  device.cpp
  device_properties.cpp
  event.cpp
  image_format.cpp
  init.cpp
  kernel.cpp
  log.cpp
  memory.cpp
  printf.cpp
  program.cpp
  queue.cpp
  queue_controller.cpp
  semaphore.cpp
  sha1.cpp
  tracing.cpp
  unit.cpp
  utils.cpp
  exports.map
)
target_link_libraries(OpenCL-objects clvk-config-definitions)

if (CLVK_UNIT_TESTING)
   target_compile_definitions(OpenCL-objects PUBLIC CLVK_UNIT_TESTING_ENABLED)
   # unit.{cpp|hpp} is using "extern C" to avoid a mangling issue.
   enable_language(C)
endif()

# Disable deprecation warnings on the file that has the entrypoints.
if (NOT MSVC)
set_source_files_properties(api.cpp PROPERTIES COMPILE_FLAGS
    -Wno-deprecated-declarations)
endif()

set_property(TARGET OpenCL-objects PROPERTY POSITION_INDEPENDENT_CODE 1)
target_include_directories(OpenCL-objects SYSTEM BEFORE PRIVATE
    ${SPIRV_HEADERS_SOURCE_DIR}/include
    ${SPIRV_TOOLS_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)

find_package(Threads)

set(OpenCL-dependencies
  ${CMAKE_THREAD_LIBS_INIT}
  SPIRV-Tools
  ${Vulkan_LIBRARIES}
)

option(CLVK_PERFETTO_ENABLE "Enable perfetto tracing" OFF)
if (CLVK_PERFETTO_ENABLE)
  set(CLVK_PERFETTO_BACKEND "InProcess"
    CACHE STRING "Select perfetto backend")
  set(CLVK_PERFETTO_BACKEND_OPTIONS
    System
    InProcess
  )
  set_property(CACHE CLVK_PERFETTO_BACKEND PROPERTY STRINGS ${CLVK_PERFETTO_BACKEND_OPTIONS})
  if (${CLVK_PERFETTO_BACKEND} STREQUAL InProcess)
      target_compile_definitions(clvk-config-definitions INTERFACE CLVK_PERFETTO_BACKEND_INPROCESS)
  endif()

  target_compile_definitions(OpenCL-objects PUBLIC CLVK_PERFETTO_ENABLE)
  if (CLVK_PERFETTO_SDK_DIR)
    target_include_directories(OpenCL-objects SYSTEM BEFORE PRIVATE
      ${CLVK_PERFETTO_SDK_DIR})
    target_compile_definitions(OpenCL-objects PUBLIC CLVK_PERFETTO_AMALGAMATED)
  else()
    if (CLVK_PERFETTO_INCLUDE_DIR)
      target_include_directories(OpenCL-objects SYSTEM BEFORE PRIVATE
        ${CLVK_PERFETTO_INCLUDE_DIR})
    endif()
  endif()

  if (NOT CLVK_PERFETTO_LIBRARY)
    if (NOT CLVK_PERFETTO_SDK_DIR)
      message(FATAL_ERROR "CLVK_PERFETTO_SDK_DIR must be provided with
        CLVK_PERFETTO_ENABLE=ON")
    endif()

    add_library(perfetto STATIC ${CLVK_PERFETTO_SDK_DIR}/perfetto.cc)
    set_target_properties(perfetto PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set(CLVK_PERFETTO_LIBRARY perfetto)
  endif()
  set(OpenCL-dependencies ${OpenCL-dependencies} ${CLVK_PERFETTO_LIBRARY})

  message(STATUS "CLVK_PERFETTO_SDK_DIR = '${CLVK_PERFETTO_SDK_DIR}'")
  message(STATUS "CLVK_PERFETTO_LIBRARY = '${CLVK_PERFETTO_LIBRARY}'")
  message(STATUS "CLVK_PERFETTO_BACKEND = '${CLVK_PERFETTO_BACKEND}'")
endif()

if (CLVK_COMPILER_AVAILABLE OR ${CLVK_VULKAN_IMPLEMENTATION} STREQUAL swiftshader)
  set(OpenCL-dependencies ${OpenCL-dependencies} SPIRV-Tools-opt)
endif()
if (CLVK_COMPILER_AVAILABLE)
  set(OpenCL-dependencies ${OpenCL-dependencies} SPIRV-Tools-link)
endif()

if (ANDROID)
  set(OpenCL-dependencies ${OpenCL-dependencies} log)
endif()


# The reflection parser requires the headers.
target_include_directories(OpenCL-objects PRIVATE
  "${CLSPV_SOURCE_DIR}/include")
if (CLVK_CLSPV_ONLINE_COMPILER)
  set(OpenCL-dependencies ${OpenCL-dependencies} clspv_core)
  if (CLVK_ENABLE_SPIRV_IL)
    set(OpenCL-dependencies ${OpenCL-dependencies} LLVMSPIRVLib)
    target_include_directories(OpenCL-objects PRIVATE
      "${LLVM_SPIRV_SOURCE}/include")
  endif()
else()
  if (CLVK_COMPILER_AVAILABLE)
    add_dependencies(OpenCL-objects clspv)
    target_compile_definitions(clvk-config-definitions INTERFACE
      DEFAULT_CLSPV_BINARY_PATH="$<TARGET_FILE:clspv>")
  endif()
endif()

# llvm-spirv needed to test simple_test_from_il_binary in CI even when
# not used in online compilation mode
if (CLVK_COMPILER_AVAILABLE AND CLVK_ENABLE_SPIRV_IL)
  add_dependencies(OpenCL-objects llvm-spirv)
  set_target_properties(llvm-spirv PROPERTIES RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR})
  target_compile_definitions(clvk-config-definitions INTERFACE
    DEFAULT_LLVMSPIRV_BINARY_PATH="$<TARGET_FILE:llvm-spirv>")
endif()

option(CLVK_ENABLE_TIMING "Enable timing of clvk internals" OFF)
if (CLVK_ENABLE_TIMING)
	target_compile_definitions(OpenCL-objects PRIVATE CVK_ENABLE_TIMING)
endif()

function(CLLibrary target type)
  add_library(${target} ${type} $<TARGET_OBJECTS:OpenCL-objects>)
  target_link_libraries(${target} PRIVATE ${OpenCL-dependencies})
  target_link_libraries(${target} PUBLIC clvk-config-definitions)
  set_target_properties(${target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
      ${CMAKE_BINARY_DIR})
endfunction(CLLibrary)

CLLibrary(OpenCL SHARED)
CLLibrary(OpenCL-static STATIC)

if(NOT APPLE)
  set_target_properties(OpenCL PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1)
else()
  set_target_properties(OpenCL PROPERTIES
    FRAMEWORK TRUE
    FRAMEWORK_VERSION A
    MACOSX_FRAMEWORK_IDENTIFIER com.apple.opencl
    SOVERSION 1.0.0
  )
endif()
# Basic install rules
install(TARGETS OpenCL DESTINATION .)
if (NOT CLVK_CLSPV_ONLINE_COMPILER AND CLVK_COMPILER_AVAILABLE)
  install(TARGETS clspv DESTINATION .)
endif()
