#include <fstream>
#include <gtest/gtest.h>
#include <stdlib.h>

#include "config.cpp"
#include "log.cpp"

// Test for finding file through env.
// Note : make sure the other paths dont have config files.
TEST(ConfigTest, FileFromEnvVar) {
    std::ofstream temp_config_file("/tmp/tmp-test-config2.conf");
    std::string expected_val = "test";

    EXPECT_TRUE(temp_config_file.is_open());

    auto original_config_val = gConfigOptions[0].name;
    auto config_to_mod = gConfigOptions[0].name + "=" + expected_val + "\n";
    temp_config_file << config_to_mod;
    temp_config_file.close();

    const std::string var_name = "CLVK_CONFIG_FILE";
    auto original_env = getenv(var_name.c_str());

    setenv("CLVK_CONFIG_FILE", "/tmp/tmp-test-config2.conf", 1);
    int err = parse_config_file();
    if (original_env != nullptr) {
        setenv("CLVK_CONFIG_FILE", original_env, 1);
    }
    auto val_to_test = (*reinterpret_cast<const config_value<std::string>*>(
        gConfigOptions[0].value));
    EXPECT_EQ(err, 0);
    EXPECT_EQ(val_to_test.value, expected_val);
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

    read_config_file(configs, config_stream);
    EXPECT_EQ(configs["option1"], "value1");
    EXPECT_EQ(configs["option2"], "false");
    EXPECT_EQ(configs["option3"], "100");
    config_stream.close();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
