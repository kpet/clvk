// Copyright 2026 The clvk authors.
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

#include "utils.hpp"

#include <gtest/gtest.h>

#ifndef WIN32
TEST(ShellQuoteToken, PassesPlainTokensThroughSingleQuotes) {
    EXPECT_EQ(shell_quote_token("-cl-std=CL1.2"), "'-cl-std=CL1.2'");
}

TEST(ShellQuoteToken, EscapesSingleQuotes) {
    EXPECT_EQ(shell_quote_token("-DNAME='x'"),
              "'-DNAME=" "'\\''" "x" "'\\''" "'");
}

TEST(ShellQuoteToken, KeepsMetacharactersVerbatim) {
    EXPECT_EQ(shell_quote_token("-DXM2S(x)=#x"), "'-DXM2S(x)=#x'");
    EXPECT_EQ(shell_quote_token("-DVALUE=$HOME"), "'-DVALUE=$HOME'");
}

TEST(QuoteOptionsForShell, SplitsOnUnquotedSpacesOnly) {
    EXPECT_EQ(quote_options_for_shell("-w -cl-std=CL1.2"),
              "'-w' '-cl-std=CL1.2' ");
}

TEST(QuoteOptionsForShell, KeepsQuotedValuesInOneToken) {
    EXPECT_EQ(quote_options_for_shell("-DMSG=\"hello world\" -w"),
              "'-DMSG=hello world' '-w' ");
}

TEST(QuoteOptionsForShell, DropsEmptyTokens) {
    EXPECT_EQ(quote_options_for_shell("  -w   -O2  "), "'-w' '-O2' ");
}
#else
TEST(ShellQuoteToken, KeepsLegacyWindowsBehaviour) {
    EXPECT_EQ(shell_quote_token("-w"), "-w");
    EXPECT_EQ(shell_quote_token("hello world"), "\"hello world\"");
}
#endif
