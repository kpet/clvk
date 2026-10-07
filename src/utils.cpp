// Copyright 2018 The clvk authors.
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
#include <array>
#include <cstdio>
#include <cstdlib>
#include <sstream>

#ifdef __APPLE__
#include <unistd.h>
#endif

#ifdef WIN32
#include <Windows.h>
#include <io.h>
#endif

#if !defined(WIN32) && !defined(__APPLE__)
#include <pthread.h>
#endif

std::string shell_quote_token(const std::string& token) {
#ifdef WIN32
    // cmd.exe treats single quotes as regular characters, so keep the
    // historical double quoting behaviour there.
    if (token.find("-") == 0) {
        return token;
    }
    return "\"" + token + "\"";
#else
    // The command line is executed through popen(3), so shell-quote the
    // token to hand it to the child process verbatim.
    std::string quoted = "'";
    for (char c : token) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
#endif
}

std::string quote_options_for_shell(const std::string& options) {
    // Split the options on unquoted spaces and drop the double quotes: the
    // OpenCL options string uses shell-like quoting, while the child process
    // receives each option through its own argv entry.
    std::vector<std::string> tokens;
    std::string token;
    bool in_quotes = false;
    for (char c : options) {
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ' ' && !in_quotes) {
            if (!token.empty()) {
                tokens.push_back(token);
            }
            token.clear();
        } else {
            token += c;
        }
    }
    if (!token.empty()) {
        tokens.push_back(token);
    }

    std::string quoted;
    for (const auto& t : tokens) {
        quoted += shell_quote_token(t);
        quoted += " ";
    }
    return quoted;
}

char* cvk_mkdtemp(std::string& tmpl) {
#ifdef WIN32
    if (_mktemp_s(&tmpl.front(), tmpl.size() + 1) != 0) {
        return nullptr;
    }

    if (!CreateDirectory(tmpl.c_str(), nullptr)) {
        return nullptr;
    }

    return &tmpl.front();
#else
    return mkdtemp(&tmpl.front());
#endif
}

int cvk_exec(const std::string& cmd, std::string* output) {
#ifdef WIN32
#define popen _popen
#define pclose _pclose
#endif
    cvk_info("About to run \"%s\"", cmd.c_str());

    std::array<char, 512> buffer;
    std::string out;
    std::string cmd_with_err = cmd + " 2>&1";
    FILE* pipe = popen(cmd_with_err.c_str(), "r");

    if (pipe == nullptr) {
        return -1;
    }

    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        out += buffer.data();
    }

    if (output != nullptr) {
        *output = std::move(out);
    }

    int ret = pclose(pipe);

    cvk_info("Return code was: %d", ret);

    return ret;
#ifdef WIN32
#undef popen
#undef pclose
#endif
}

void cvk_set_current_thread_name_if_supported(const std::string& name) {
#if !defined(WIN32) && !defined(__APPLE__)
    pthread_setname_np(pthread_self(), name.c_str());
#endif
}
