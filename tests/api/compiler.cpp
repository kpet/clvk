// Copyright 2020 The clvk authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "testcl.hpp"

TEST_F(WithContext, BuildLog) {
    static const char* source_warning =
        "#warning THIS IS A WARNING\nvoid kernel test(){}\n";
    static const char* source_error = "#error THIS IS AN ERROR";
    std::string build_log;

    // Test log for successful compilation
    auto program_warning = CreateAndBuildProgram(source_warning);

    build_log = GetProgramBuildLog(program_warning);
    ASSERT_TRUE(build_log.find("THIS IS A WARNING") != std::string::npos);

    // Test log for failed compilation
    auto program_error = CreateProgram(source_error);
    cl_int err =
        clBuildProgram(program_error, 1, &gDevice, nullptr, nullptr, nullptr);
    ASSERT_EQ(err, CL_BUILD_PROGRAM_FAILURE);

    build_log = GetProgramBuildLog(program_error);
    ASSERT_TRUE(build_log.find("THIS IS AN ERROR") != std::string::npos);
}
