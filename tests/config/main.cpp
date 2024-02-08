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

#include <fstream>
#include <gtest/gtest.h>
#include <stdlib.h>

#ifdef CLVK_UNIT_TESTING_ENABLED
#include "config.hpp"

// Test for finding file through env.
// Note : make sure the other paths dont have config files.
TEST(ConfigTest, FileFromEnvVar) {

    std::string conf_file = "/tmp/tmp-test-config.conf";
    std::ofstream temp_config_file(conf_file);
    temp_config_file << "option1=value1\n";
    std::string expected_val = "test";
    EXPECT_TRUE(temp_config_file.is_open());
    temp_config_file.close();

    const std::string var_name = "CLVK_CONFIG_FILE";
    auto original_env = getenv(var_name.c_str());

    setenv("CLVK_CONFIG_FILE", conf_file.c_str(), 1);
    std::string used_file = configs::parse_config_file();
    if (original_env != nullptr) {
        setenv("CLVK_CONFIG_FILE", original_env, 1);
    }
    EXPECT_EQ(used_file, conf_file);
}

// Test for checking if file is being read properly
TEST(ConfigTest, ParseConfigFile) {
    std::unordered_map<std::string, std::string> configs;
    // Create file with desired content

    std::ofstream temp_config_file("/tmp/tmp-test-config1.conf");
    EXPECT_TRUE(temp_config_file.is_open());

    temp_config_file << "option1=value1\n";
    temp_config_file << "option2=false\n";
    temp_config_file << "option3=100\n";
    temp_config_file.close();
    std::ifstream config_stream("/tmp/tmp-test-config1.conf");
    EXPECT_TRUE(config_stream.is_open());

    configs::read_config_file(configs, config_stream);
    EXPECT_EQ(configs["option1"], "value1");
    EXPECT_EQ(configs["option2"], "false");
    EXPECT_EQ(configs["option3"], "100");
    config_stream.close();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#endif