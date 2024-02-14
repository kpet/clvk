// Copyright 2024 The clvk authors.
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

#ifdef _WIN32
#include <stdlib.h>
#define set_env(name, value) _putenv_s(name, value)
#else
#include <cstdlib>
#define set_env(name, value) setenv(name, value, 1)
#endif

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

#ifdef CLVK_UNIT_TESTING_ENABLED
#include "unit.hpp"

// Test for making sure the configs from env var are read
// and overwrite configs from other paths.
TEST(ConfigTest, FileFromEnvVar) {
    clGetPlatformIDs(1, nullptr, nullptr);
    EXPECT_EQ(clvk_get_config()->cache_dir.value, "testing");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#endif
