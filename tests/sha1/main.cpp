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

#include <cstring>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "sha1.hpp"

// Convert a SHA-1 hash to its hex string representation.
static std::string to_hex_string(const cvk_sha1_hash& hash) {
    const char chars[] = "0123456789abcdef";
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(hash.data());
    std::string str = "";
    for (uint32_t i = 0; i < SHA1_DIGEST_NUM_BYTES; i++) {
        str += *(chars + (bytes[i] >> 4));
        str += *(chars + (bytes[i] & 0xF));
    }
    return str;
}

// Convert a hex string to a vector of bytes.
static std::vector<uint8_t> hex_string_to_bytes(const std::string& str) {
    const char chars[] = "0123456789abcdef";
    std::vector<uint8_t> bytes;
    for (uint32_t i = 0; i < str.size(); i += 2) {
        uint8_t byte = (strchr(chars, str[i]) - chars) << 4;
        byte |= strchr(chars, str[i + 1]) - chars;
        bytes.push_back(byte);
    }
    return bytes;
}

class SHA1Test
    : public testing::TestWithParam<std::pair<std::string, std::string>> {};

TEST_P(SHA1Test, Test) {
    std::string msg = GetParam().first;
    std::string expected = GetParam().second;

    // Convert input to raw bytes.
    ASSERT_EQ(msg.size() % 2, 0);
    std::vector<uint8_t> data = hex_string_to_bytes(msg);

    // Compute hash and check result.
    cvk_sha1_hash result = cvk_sha1(data.data(), data.size());
    EXPECT_EQ(expected, to_hex_string(result));
}

std::pair<std::string, std::string> tests[] = {
#include "tests.inc"
};

INSTANTIATE_TEST_CASE_P(SHA1Tests, SHA1Test, testing::ValuesIn(tests));

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
