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

#include <cstring>

#include "device_properties.hpp"
#include "log.hpp"

#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif

struct cvk_device_properties_mali_exynos9820 : public cvk_device_properties {
    cl_ulong get_global_mem_cache_size() const override final { return 262144; }
    cl_ulong get_num_compute_units() const override final { return 12; }
};

struct cvk_device_properties_mali_exynos990 : public cvk_device_properties {
    cl_ulong get_global_mem_cache_size() const override final { return 262144; }
    cl_ulong get_num_compute_units() const override final { return 11; }
};

struct cvk_device_properties_adreno_615 : public cvk_device_properties {
    cl_ulong get_global_mem_cache_size() const override final { return 65536; }
    cl_ulong get_num_compute_units() const override final { return 1; }
};

struct cvk_device_properties_adreno_620 : public cvk_device_properties {
    cl_ulong get_global_mem_cache_size() const override final { return 65536; }
    cl_ulong get_num_compute_units() const override final { return 1; }
};

struct cvk_device_properties_adreno_630 : public cvk_device_properties {
    cl_ulong get_global_mem_cache_size() const override final { return 131072; }
    cl_ulong get_num_compute_units() const override final { return 2; }
};

struct cvk_device_properties_adreno_640 : public cvk_device_properties {
    cl_ulong get_global_mem_cache_size() const override final { return 131072; }
    cl_ulong get_num_compute_units() const override final { return 2; }
};

struct cvk_device_properties_intel : public cvk_device_properties {
    cl_uint get_max_first_cmd_batch_size() const override final { return 10; }
    cl_uint get_max_cmd_group_size() const override final { return 1; }
    std::string get_compile_options() const override final {
        return "-hack-mul-extended";
    }
};

cvk_device_properties create_cvk_device_properties(const char* name) {
    if (strncmp(name, "Mali-", 5) == 0) {
#ifdef __ANDROID__
        // Find out which SoC this is.
        char soc[PROP_VALUE_MAX + 1];
        int len = __system_property_get("ro.hardware", soc);
        if (len == 0) {
            cvk_warn("Unable to query 'ro.hardware' system property, some "
                     "device properties will be incorrect.");
        } else if (!strcmp(soc, "exynos9820")) {
            return cvk_device_properties_mali_exynos9820();
        } else if (!strcmp(soc, "exynos990")) {
            return cvk_device_properties_mali_exynos990();
        } else {
            cvk_warn("Unrecognized 'ro.hardware' value '%s', some device "
                     "properties will be incorrect.",
                     soc);
        }
#else
        cvk_warn("Unrecognized Mali device, some device properties will be "
                 "incorrect.");
#endif
    } else if (strcmp(name, "Adreno (TM) 615")) {
        return cvk_device_properties_adreno_615();
    } else if (strcmp(name, "Adreno (TM) 620")) {
        return cvk_device_properties_adreno_620();
    } else if (strcmp(name, "Adreno (TM) 630")) {
        return cvk_device_properties_adreno_630();
    } else if (strcmp(name, "Adreno (TM) 640")) {
        return cvk_device_properties_adreno_640();
    } else if (strstr(name, "Intel")) {
        return cvk_device_properties_intel();
    } else {
        cvk_warn("Unrecognized device '%s', some device properties will be "
                 "incorrect.",
                 name);
    }

    return cvk_device_properties();
}
