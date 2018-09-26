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
endif()

add_library(OpenCL SHARED
  api.cpp
  device.cpp
  init.cpp
  kernel.cpp
  memory.cpp
  program.cpp
  queue.cpp
  utils.cpp
)

set_target_properties(OpenCL PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1)

set_target_properties(OpenCL PROPERTIES LIBRARY_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR})

find_package(Threads)
target_link_libraries(OpenCL ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(OpenCL SYSTEM BEFORE PRIVATE "${Vulkan_INCLUDE_DIRS}")
target_link_libraries(OpenCL ${Vulkan_LIBRARIES})

add_dependencies(OpenCL clspv)
target_link_libraries(OpenCL SPIRV-Tools-opt SPIRV-Tools-link)

target_compile_definitions(OpenCL PRIVATE
    DEFAULT_CLSPV_BINARY_PATH="$<TARGET_FILE:clspv>")
