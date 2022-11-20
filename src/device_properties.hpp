// Copyright 2022 The clvk authors.
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

#pragma once

#include <memory>
#include <string>

#include "cl_headers.hpp"

struct cvk_device_properties {
    virtual cl_ulong get_global_mem_cache_size() const { return 0; }
    virtual cl_ulong get_num_compute_units() const { return 1; }

    virtual cl_uint get_max_cmd_batch_size() const { return 10000; }
    virtual cl_uint get_max_first_cmd_batch_size() const { return 10000; }
    virtual cl_uint get_max_cmd_group_size() const { return UINT32_MAX; }
    virtual cl_uint get_max_first_cmd_group_size() const { return UINT32_MAX; }

    virtual std::string get_compile_options() const { return ""; }
};

cvk_device_properties create_cvk_device_properties(const char* name);
