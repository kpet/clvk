# Copyright 2020 The clvk authors.
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

set(BINARY_NAME sha1_tests)

add_executable(
    ${BINARY_NAME}
    main.cpp
    ${PROJECT_SOURCE_DIR}/src/sha1.cpp
)

include_directories(${PROJECT_SOURCE_DIR}/src)

# Use the gtest copy that comes with clspv's version of LLVM
include_directories(${LLVM_BINARY_DIR}/include)
include_directories(${LLVM_SOURCE_DIR}/include)
include_directories(${LLVM_SOURCE_DIR}/utils/unittest/googletest/include/)

target_link_libraries(${BINARY_NAME} gtest gtest_main)

set_target_properties(${BINARY_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR})

