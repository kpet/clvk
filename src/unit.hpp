// Copyright 2021 The clvk authors.
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

#ifdef CLVK_UNIT_TESTING_ENABLED

#include "config.hpp"

#include <CL/cl.h>

extern "C" {

void CL_API_CALL clvk_override_device_max_compute_work_group_count(
    cl_device_id device, uint32_t x, uint32_t y, uint32_t z);

void CL_API_CALL clvk_restore_device_properties(cl_device_id device);

const config_struct* CL_API_CALL clvk_get_config();
}

template <typename T> struct clvk_config_scoped_override {
    clvk_config_scoped_override(config_value<T>* cfgval, const T& newval,
                                bool user_set)
        : m_value_ptr(cfgval), m_old_value(*cfgval) {
        cfgval->value = newval;
        cfgval->set = user_set;
    }
    ~clvk_config_scoped_override() { *m_value_ptr = m_old_value; }

private:
    config_value<T>* m_value_ptr;
    config_value<T> m_old_value;
};

#define CLVK_CONFIG_SCOPED_OVERRIDE(opt, type, newval, user_set)               \
    clvk_config_scoped_override<type>(                                         \
        const_cast<config_value<type>*>(&clvk_get_config()->opt), newval,      \
        user_set)

#endif
