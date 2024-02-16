# Copyright 2024 The clvk authors.
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

add_gtest_executable(config_test
    main.cpp
)

# Use the configure_file command to copy during configuration stage
configure_file(
    clvk.conf
    ${CMAKE_BINARY_DIR}/clvk.conf  # Destination, including subfolder
    COPYONLY  # Just copy, don't process as a template
)

configure_file(
    conf_test.conf
    ${CMAKE_BINARY_DIR}/conf_test.conf
    COPYONLY
)
