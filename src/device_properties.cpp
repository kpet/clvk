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
    const std::set<std::string> get_native_builtins() const override final {
        return std::set<std::string>({
            "ceil",           "copysign",    "exp2",      "fabs",
            "floor",          "fma",         "fmax",      "fmin",
            "half_exp",       "half_exp10",  "half_exp2", "half_log",
            "half_log10",     "half_log2",   "half_powr", "half_rsqrt",
            "half_sqrt",      "isequal",     "isfinite",  "isgreater",
            "isgreaterequal", "isinf",       "isless",    "islessequal",
            "islessgreater",  "isnan",       "isnormal",  "isnotequal",
            "isordered",      "isunordered", "mad",       "rint",
            "round",          "rsqrt",       "signbit",   "sqrt",
            "trunc",
        });
    }
    std::string get_compile_options() const override final {
        return "-hack-mul-extended";
    }
};

static bool isIntelDevice(const char* name, const uint32_t vendorID) {
    const uint32_t IntelVendorID = 0x8086;
    return vendorID == IntelVendorID || strncmp(name, "Intel", 5) == 0;
}

struct cvk_device_properties_amd : public cvk_device_properties {
    cl_uint get_max_first_cmd_batch_size() const override final { return 10; }
    cl_uint get_max_cmd_group_size() const override final { return 1; }
    const std::set<std::string> get_native_builtins() const override final {
        return std::set<std::string>({
            "ceil",        "copysign",       "exp2",        "fabs",
            "fdim",        "floor",          "fmax",        "fmin",
            "frexp",       "half_exp",       "half_exp10",  "half_exp2",
            "half_log",    "half_log10",     "half_log2",   "half_powr",
            "half_rsqrt",  "half_sqrt",      "isequal",     "isfinite",
            "isgreater",   "isgreaterequal", "isinf",       "isless",
            "islessequal", "islessgreater",  "isnan",       "isnormal",
            "isnotequal",  "isordered",      "isunordered", "ldexp",
            "log",         "log10",          "log2",        "mad",
            "rint",        "round",          "rsqrt",       "signbit",
            "sqrt",        "trunc",
        });
    }
};

static bool isAMDDevice(const char* name, const uint32_t vendorID) {
    const uint32_t AMDVendorID = 0x1002;
    return vendorID == AMDVendorID || strncmp(name, "AMD", 3) == 0;
}

struct cvk_device_properties_samsung_xclipse_920
    : public cvk_device_properties {
    const std::set<std::string> get_native_builtins() const override final {
        return std::set<std::string>({"fma"});
    }
};

struct cvk_device_properties_swiftshader : public cvk_device_properties {
    const std::set<std::string> get_native_builtins() const override final {
        return std::set<std::string>({
            "asin",          "asinpi",     "atan",
            "atanpi",        "ceil",       "copysign",
            "fabs",          "fdim",       "floor",
            "fma",           "fmax",       "fmin",
            "fmod",          "half_rsqrt", "half_sqrt",
            "isequal",       "isgreater",  "isgreaterequal",
            "isinf",         "isless",     "islessequal",
            "islessgreater", "isnan",      "isnormal",
            "isnotequal",    "isordered",  "isunordered",
            "mad",           "rint",       "round",
            "rsqrt",         "signbit",    "sqrt",
            "trunc",
        });
    }
};

static bool isSwiftShaderDevice(const char* name, const uint32_t vendorID,
                                const uint32_t deviceID) {
    const uint32_t SwiftshaderDeviceID = 0xc0de;
    const uint32_t SwiftshaderVendorID = 0x1ae0;
    return (vendorID == SwiftshaderVendorID &&
            deviceID == SwiftshaderDeviceID) ||
           strncmp(name, "SwiftShader Device", 18) == 0;
}

struct cvk_device_properties_nvidia : public cvk_device_properties {
    const std::set<std::string> get_native_builtins() const override final {
        return std::set<std::string>({
            "acos",        "acosh",          "acospi",      "asin",
            "asinh",       "asinpi",         "atan",        "atan2",
            "atan2pi",     "atanh",          "atanpi",      "ceil",
            "copysign",    "fabs",           "fdim",        "floor",
            "fma",         "fmax",           "fmin",        "frexp",
            "half_rsqrt",  "half_sqrt",      "isequal",     "isfinite",
            "isgreater",   "isgreaterequal", "isinf",       "isless",
            "islessequal", "islessgreater",  "isnan",       "isnormal",
            "isnotequal",  "isordered",      "isunordered", "ldexp",
            "mad",         "rint",           "round",       "rsqrt",
            "signbit",     "sqrt",           "tanh",        "trunc",
        });
    }
};

static bool isNVIDIADevice(const uint32_t vendorID) {
    const uint32_t NVIDIAVendorID = 0x10de;
    return vendorID == NVIDIAVendorID;
}

#define RETURN(x)                                                              \
    cvk_info_fn(#x);                                                           \
    return std::make_unique<x>();

std::unique_ptr<cvk_device_properties>
create_cvk_device_properties(const char* name, const uint32_t vendorID,
                             const uint32_t deviceID) {
    if (strncmp(name, "Mali-", 5) == 0) {
#ifdef __ANDROID__
        // Find out which SoC this is.
        char soc[PROP_VALUE_MAX + 1];
        int len = __system_property_get("ro.hardware", soc);
        if (len == 0) {
            cvk_warn("Unable to query 'ro.hardware' system property, some "
                     "device properties will be incorrect.");
        } else if (strcmp(soc, "exynos9820") == 0) {
            RETURN(cvk_device_properties_mali_exynos9820);
        } else if (strcmp(soc, "exynos990") == 0) {
            RETURN(cvk_device_properties_mali_exynos990);
        } else {
            cvk_warn("Unrecognized 'ro.hardware' value '%s', some device "
                     "properties will be incorrect.",
                     soc);
        }
#else
        cvk_warn("Unrecognized Mali device, some device properties will be "
                 "incorrect.");
#endif
    } else if (strcmp(name, "Adreno (TM) 615") == 0) {
        RETURN(cvk_device_properties_adreno_615);
    } else if (strcmp(name, "Adreno (TM) 620") == 0) {
        RETURN(cvk_device_properties_adreno_620);
    } else if (strcmp(name, "Adreno (TM) 630") == 0) {
        RETURN(cvk_device_properties_adreno_630);
    } else if (strcmp(name, "Adreno (TM) 640") == 0) {
        RETURN(cvk_device_properties_adreno_640);
    } else if (isIntelDevice(name, vendorID)) {
        RETURN(cvk_device_properties_intel);
    } else if (isAMDDevice(name, vendorID)) {
        RETURN(cvk_device_properties_amd);
    } else if (strcmp(name, "Samsung Xclipse 920") == 0) {
        RETURN(cvk_device_properties_samsung_xclipse_920);
    } else if (isSwiftShaderDevice(name, vendorID, deviceID)) {
        RETURN(cvk_device_properties_swiftshader);
    } else if (isNVIDIADevice(vendorID)) {
        RETURN(cvk_device_properties_nvidia);
    } else {
        cvk_warn("Unrecognized device '%s' (vendorID '0x%x' - deviceID "
                 "'0x%x'), some device properties will be "
                 "incorrect.",
                 name, vendorID, deviceID);
    }

    RETURN(cvk_device_properties);
}
