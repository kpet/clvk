#include <gtest/gtest.h>
#include <fstream>

#include "config.cpp"

// Test for finding file through env.
TEST(ConfigTest, FileFromEnvVar) {
    std::ofstream outfile("/tmp/tmp-test-config.def");

    if (!outfile.is_open()) {
        std::cerr << "Error creating file!" << std::endl;
        return; // Indicate an error
    }
    const std::string var_name = "CLVK_CONFIG_FILE";
    auto original_env = getenv(var_name.c_str());
    std::stringstream buffer;
    std::streambuf* old_err = std::cerr.rdbuf(buffer.rdbuf());

    setenv("CLVK_CONFIG_FILE", "/tmp/tmp-test-config.def", 1);
    parse_config_file();
    if (original_env != nullptr) {
        setenv("CLVK_CONFIG_FILE", original_env, 1);
    }
    std::cerr.rdbuf(old_err); // Restore std::err
    EXPECT_TRUE(buffer.str().empty());
}

// Test for checking if file is being read properly
TEST(ConfigTest, ParseConfigFile) {
    std::unordered_map<std::string, std::string> configs;
    // Create file with desired content
    std::ofstream temp_config_file("/tmp/tmp-test-config.def");
    EXPECT_TRUE(temp_config_file.is_open());

    temp_config_file << "option1=value1\n";
    temp_config_file << "option2=false\n";
    temp_config_file << "option3=100\n";
    temp_config_file.close();
    std::ifstream config_stream("/tmp/tmp-test-config.def");
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
