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

// Test for finding file through env.
// Note : make sure the other paths dont have config files.
TEST(ConfigTest, FileFromEnvVar) {
    std::filesystem::path conf_file =
        std::filesystem::temp_directory_path() / "temp_config_clvk.conf";
    std::ofstream temp_config_file(conf_file);
    temp_config_file << "cache_dir=testing\n";
    EXPECT_TRUE(temp_config_file.is_open());
    temp_config_file.close();

    std::string var_name = "CLVK_CONFIG_FILE";
    std::string org_val = "";
    const char* original_env = getenv(var_name.c_str());
    const char* path_as_cstr = conf_file.c_str();
    set_env(var_name.c_str(), path_as_cstr);
    clGetPlatformIDs(1, nullptr, nullptr);
    std::string org_config = var_name.append(org_val);
    if (original_env != nullptr) {
        set_env(var_name.c_str(), original_env);
    }
    EXPECT_EQ(clvk_get_config()->cache_dir.value, "testing");
}

int main(int argc, char** argv) {

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#endif