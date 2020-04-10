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
#include <cstdio>
#include <cstdlib>

#ifdef __APPLE__
#include <unistd.h>
#endif

#ifdef WIN32
#include <Windows.h>
#include <io.h>
#endif

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

