# Copyright 2019 The clvk authors.
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

set(CLConf_OUT_DIR dummy)
set(CLConform_LIBRARIES -lOpenCL)
set(CL_LIBCLCXX_DIR dummy)
set(CL_INCLUDE_DIR ${CLVK_PROJECT_SOURCE_DIR}/external/OpenCL-Headers)
set(CL_LIB_DIR ${CLVK_PROJECT_SOURCE_DIR}/external/OpenCL-ICD-Loader/build)
set(SPIRV_INCLUDE_DIR "${SPIRV_HEADERS_SOURCE_DIR}")
set(VULKAN_IS_SUPPORTED ON)

option(CLVK_BUILD_CONFORMANCE_TESTS_GL_GLES_SUPPORTED
       "Build OpenCL conformance tests with `GL_IS_SUPPORTED` and `GLES_IS_SUPPORTED`")
if (CLVK_BUILD_CONFORMANCE_TESTS_GL_GLES_SUPPORTED)
set(GL_IS_SUPPORTED ON)
set(GLES_IS_SUPPORTED ON)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/conformance)
add_subdirectory(${CLVK_PROJECT_SOURCE_DIR}/external/OpenCL-CTS
                 ${CMAKE_BINARY_DIR}/conformance)
