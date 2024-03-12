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

#ifdef CLVK_UNIT_TESTING_ENABLED
#include "unit.hpp"

#include <gtest/gtest.h>

// Test for making sure the configs from env var are read
// and overwrite configs from other paths.
TEST(ConfigTest, FileFromEnvVar) {
    clGetPlatformIDs(1, nullptr, nullptr);
    EXPECT_EQ(clvk_get_config()->cache_dir.value, "testing");
    EXPECT_EQ(clvk_get_config()->compiler_temp_dir.value, "not/overwritten/");
    EXPECT_EQ(clvk_get_config()->log_colour.value, true);
    EXPECT_EQ(clvk_get_config()->percentage_of_available_memory_reported.value,
              100);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#endif
