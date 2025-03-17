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

struct cvk_device_properties_mali : public cvk_device_properties {
    std::string vendor() const override final { return "ARM"; }
    cl_uint get_max_first_cmd_batch_size() const override final { return 10; }
    cl_uint get_max_cmd_group_size() const override final { return 1; }

    cvk_device_properties_mali(const uint32_t deviceID)
        : m_deviceID(deviceID) {}

    bool is_non_uniform_decoration_broken() const override final {
#define GPU_ID2_ARCH_MAJOR_SHIFT 28
#define GPU_ID2_ARCH_MAJOR (0xF << GPU_ID2_ARCH_MAJOR_SHIFT)
        // bifrost support of non uniform decoration is broken
        const uint32_t bifrost_arch_major = 8 << GPU_ID2_ARCH_MAJOR_SHIFT;
        return (m_deviceID & GPU_ID2_ARCH_MAJOR) <= bifrost_arch_major;
    }

private:
    const uint32_t m_deviceID;
};

struct cvk_device_properties_mali_exynos9820
    : public cvk_device_properties_mali {
    cl_ulong get_global_mem_cache_size() const override final { return 262144; }
    cl_ulong get_num_compute_units() const override final { return 12; }
    cvk_device_properties_mali_exynos9820(const uint32_t deviceID)
        : cvk_device_properties_mali(deviceID) {}
};

struct cvk_device_properties_mali_exynos990
    : public cvk_device_properties_mali {
    cl_ulong get_global_mem_cache_size() const override final { return 262144; }
    cl_ulong get_num_compute_units() const override final { return 11; }
    cvk_device_properties_mali_exynos990(const uint32_t deviceID)
        : cvk_device_properties_mali(deviceID) {}
};

static bool isMaliDevice(const char* name, const uint32_t vendorID) {
    const uint32_t ARMVendorID = 0x13B5;
    return vendorID == ARMVendorID || strncmp(name, "Mali-", 5) == 0;
}

struct cvk_device_properties_adreno : public cvk_device_properties {
    std::string vendor() const override final { return "Qualcomm"; }
};

struct cvk_device_properties_adreno_615 : public cvk_device_properties_adreno {
    cl_ulong get_global_mem_cache_size() const override final { return 65536; }
    cl_ulong get_num_compute_units() const override final { return 1; }
};

struct cvk_device_properties_adreno_620 : public cvk_device_properties_adreno {
    cl_ulong get_global_mem_cache_size() const override final { return 65536; }
    cl_ulong get_num_compute_units() const override final { return 1; }
};

struct cvk_device_properties_adreno_630 : public cvk_device_properties_adreno {
    cl_ulong get_global_mem_cache_size() const override final { return 131072; }
    cl_ulong get_num_compute_units() const override final { return 2; }
};

struct cvk_device_properties_adreno_640 : public cvk_device_properties_adreno {
    cl_ulong get_global_mem_cache_size() const override final { return 131072; }
    cl_ulong get_num_compute_units() const override final { return 2; }
};

struct cvk_device_properties_intel : public cvk_device_properties {
    std::string vendor() const override final { return "Intel Corporation"; }
    cl_uint get_max_first_cmd_batch_size() const override final { return 10; }
    cl_uint get_max_cmd_group_size() const override final { return 1; }
    const std::set<std::string> get_native_builtins() const override final {
        return std::set<std::string>({
            "ceil",        "copysign",  "exp2",        "floor",
            "fma",         "fmax",      "fmin",        "half_exp",
            "half_exp10",  "half_exp2", "half_log",    "half_log10",
            "half_log2",   "half_powr", "half_rsqrt",  "half_sqrt",
            "isequal",     "isfinite",  "isgreater",   "isgreaterequal",
            "isinf",       "isless",    "islessequal", "islessgreater",
            "isnan",       "isnormal",  "isnotequal",  "isordered",
            "isunordered", "mad",       "rint",        "round",
            "rsqrt",       "signbit",   "sqrt",        "trunc",
        });
    }
    std::string get_compile_options() const override final {
        return "-hack-mul-extended -hack-convert-to-float "
               "-hack-image1d-buffer-bgra";
    }
    uint32_t get_preferred_subgroup_size() const override final { return 16; }
    bool
    is_bgra_format_not_supported_for_image1d_buffer() const override final {
        return true;
    }

    const image_format_set& get_disabled_image_formats() const override final {
        static image_format_set disabled_formats(
            {{CL_RGB, CL_UNORM_SHORT_565}});
        return disabled_formats;
    }
};

static bool isIntelDevice(const char* name, const uint32_t vendorID) {
    const uint32_t IntelVendorID = 0x8086;
    return vendorID == IntelVendorID || strncmp(name, "Intel", 5) == 0;
}

struct cvk_device_properties_amd : public cvk_device_properties {
    std::string vendor() const override final {
        return "Advanced Micro Devices, Inc. [AMD/ATI]";
    }
    cl_uint get_max_first_cmd_batch_size() const override final { return 10; }
    cl_uint get_max_cmd_group_size() const override final { return 1; }
    const std::set<std::string> get_native_builtins() const override final {
        return std::set<std::string>({
            "ceil",           "copysign",      "exp2",
            "fdim",           "floor",         "fma",
            "fmax",           "fmin",          "frexp",
            "half_exp",       "half_exp10",    "half_exp2",
            "half_log",       "half_log10",    "half_log2",
            "half_powr",      "half_rsqrt",    "half_sqrt",
            "isequal",        "isfinite",      "isgreater",
            "isgreaterequal", "isinf",         "isless",
            "islessequal",    "islessgreater", "isnan",
            "isnormal",       "isnotequal",    "isordered",
            "isunordered",    "ldexp",         "log",
            "log10",          "log2",          "mad",
            "rint",           "round",         "rsqrt",
            "signbit",        "sqrt",          "trunc",
        });
    }
    std::string get_compile_options() const override final {
        return "-hack-convert-to-float";
    }
};

static bool isAMDDevice(const char* name, const uint32_t vendorID) {
    const uint32_t AMDVendorID = 0x1002;
    return vendorID == AMDVendorID || strncmp(name, "AMD", 3) == 0;
}

struct cvk_device_properties_samsung_xclipse_920
    : public cvk_device_properties {
    std::string vendor() const override final { return "Samsung"; }
    const std::set<std::string> get_native_builtins() const override final {
        return std::set<std::string>({
            "ceil",      "floor",       "fma",           "fmax",
            "fmin",      "half_exp2",   "half_log2",     "half_rsqrt",
            "half_sqrt", "isequal",     "isgreater",     "isgreaterequal",
            "isless",    "islessequal", "islessgreater", "isnotequal",
            "log2",      "mad",         "round",         "rsqrt",
            "sqrt",      "exp2",
        });
    }
};

struct cvk_device_properties_swiftshader : public cvk_device_properties {
    std::string vendor() const override final { return "Google, Inc."; }
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

struct cvk_device_properties_llvmpipe : public cvk_device_properties {
    std::string vendor() const override final { return "Mesa"; }
    const std::set<std::string> get_native_builtins() const override final {
        return std::set<std::string>({
            "ceil",     "copysign",    "fabs",           "fdim",
            "floor",    "fmax",        "fmin",           "isequal",
            "isfinite", "isgreater",   "isgreaterequal", "isinf",
            "isless",   "islessequal", "islessgreater",  "isnan",
            "isnormal", "isnotequal",  "isordered",      "isunordered",
            "mad",      "rint",        "round",          "rsqrt",
            "signbit",  "sqrt",        "trunc",          "half_cos",
            "half_exp", "half_exp2",   "half_exp10",     "half_rsqrt",
            "half_sin", "half_sqrt",   "half_tan",
        });
    }
    std::string get_compile_options() const override final {
        return "-hack-convert-to-float";
    }
};

static bool isllvmpipeDevice(const uint32_t vendorID) {
    return vendorID == 0x10005;
}

struct cvk_device_properties_nvidia : public cvk_device_properties {
    std::string vendor() const override final { return "NVIDIA Corporation"; }
    const std::set<std::string> get_native_builtins() const override final {
        return std::set<std::string>({
            "acos",           "acosh",       "acospi",   "asin",
            "asinh",          "asinpi",      "atan",     "atan2",
            "atan2pi",        "atanh",       "atanpi",   "ceil",
            "copysign",       "fdim",        "floor",    "fma",
            "fmax",           "fmin",        "frexp",    "half_rsqrt",
            "half_sqrt",      "isequal",     "isfinite", "isgreater",
            "isgreaterequal", "isinf",       "isless",   "islessequal",
            "islessgreater",  "isnan",       "isnormal", "isnotequal",
            "isordered",      "isunordered", "ldexp",    "mad",
            "rint",           "round",       "rsqrt",    "signbit",
            "sqrt",           "tanh",        "trunc",
        });
    }
};

static bool isNVIDIADevice(const uint32_t vendorID) {
    const uint32_t NVIDIAVendorID = 0x10de;
    return vendorID == NVIDIAVendorID;
}

#define RETURN(x, ...)                                                         \
    cvk_info_fn(#x);                                                           \
    return std::make_unique<x>(__VA_ARGS__);

std::unique_ptr<cvk_device_properties> create_cvk_device_properties(
    const char* name, const uint32_t vendorID, const uint32_t deviceID,
    const uint32_t driverVersion, const VkDriverId driverID) {
    if (isMaliDevice(name, vendorID)) {
#ifdef __ANDROID__
        // Find out which SoC this is.
        char soc[PROP_VALUE_MAX + 1];
        int len = __system_property_get("ro.hardware", soc);
        if (len == 0) {
            cvk_warn("Unable to query 'ro.hardware' system property, some "
                     "device properties will be incorrect.");
        } else if (strcmp(soc, "exynos9820") == 0) {
            RETURN(cvk_device_properties_mali_exynos9820, deviceID);
        } else if (strcmp(soc, "exynos990") == 0) {
            RETURN(cvk_device_properties_mali_exynos990, deviceID);
        } else {
            cvk_warn("Unrecognized 'ro.hardware' value '%s', some device "
                     "properties will be incorrect.",
                     soc);
        }
#else
        cvk_warn("Unrecognized Mali device, some device properties will be "
                 "incorrect.");
#endif
        RETURN(cvk_device_properties_mali, deviceID);
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
    } else if (isllvmpipeDevice(vendorID)) {
        RETURN(cvk_device_properties_llvmpipe);
    } else {
        cvk_warn("Unrecognized device '%s' (vendorID '0x%x' - deviceID "
                 "'0x%x'), some device properties will be "
                 "incorrect.",
                 name, vendorID, deviceID);
    }

    RETURN(cvk_device_properties);
}
