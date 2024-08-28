// Copyright 2018-2023 The clvk authors.
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

#include "device.hpp"
#include "objects.hpp"
#include "unit.hpp"

using cvk_printf_callback_t = void(CL_CALLBACK*)(const char* buffer, size_t len,
                                                 size_t complete,
                                                 void* user_data);

using cvk_context_callback_pointer_type = void(CL_CALLBACK*)(cl_context context,
                                                             void* user_data);
struct cvk_context_callback {
    cvk_context_callback_pointer_type pointer;
    void* data;
};

struct cvk_context : public _cl_context,
                     refcounted,
                     object_magic_header<object_magic::context> {

    cvk_context(cvk_device* device, const cl_context_properties* props,
                void* user_data)
        : m_device(device), m_user_data(user_data) {

        if (props) {
            while (*props) {
                // Save name
                m_properties.push_back(*props);
                // Save value
                m_properties.push_back(*(props + 1));
                props += 2;
            }
            m_properties.push_back(*props);
        }
        // Get printf buffer size from extension.
        auto buff_size_prop_index =
            get_property_value_index(CL_PRINTF_BUFFERSIZE_ARM);
        if (buff_size_prop_index != -1 && !config.printf_buffer_size.set) {
            m_printf_buffersize = m_properties[buff_size_prop_index];
        } else {
            m_printf_buffersize = 0;
        }

        // Get printf callback from extension
        auto printf_callback_prop_index =
            get_property_value_index(CL_PRINTF_CALLBACK_ARM);
        if (printf_callback_prop_index != -1) {
            m_printf_callback =
                (cvk_printf_callback_t)m_properties[printf_callback_prop_index];
        } else {
            m_printf_callback = nullptr;
        }
    }

    virtual ~cvk_context() {
        for (auto cbi = m_destuctor_callbacks.rbegin();
             cbi != m_destuctor_callbacks.rend(); ++cbi) {
            auto cb = *cbi;
            cb.pointer(this, cb.data);
        }
    }

    const std::vector<cl_context_properties>& properties() const {
        return m_properties;
    }

    cvk_device* device() const { return m_device; }
    unsigned num_devices() const { return 1u; }
    bool has_device(const cvk_device* device) const {
        return device == m_device;
    }

    void add_destructor_callback(cvk_context_callback_pointer_type ptr,
                                 void* user_data) {
        cvk_context_callback cb = {ptr, user_data};
        std::lock_guard<std::mutex> lock(m_callbacks_lock);
        m_destuctor_callbacks.push_back(cb);
    }

    bool is_mem_alloc_size_valid(size_t size) {
        // TODO support multiple devices
        return size <= m_device->max_mem_alloc_size();
    }

    int get_property_value_index(const int prop) {
        for (unsigned i = 0; i < m_properties.size(); i += 2) {
            if (m_properties[i] == prop) {
                return i + 1;
            }
        }
        return -1;
    }

    size_t get_printf_buffersize() {
        if (m_printf_buffersize) {
            return m_printf_buffersize;
        } else {
            return config.printf_buffer_size;
        }
    }
    cvk_printf_callback_t get_printf_callback() { return m_printf_callback; }
    void* get_printf_userdata() { return m_user_data; }

private:
    cvk_device* m_device;
    std::mutex m_callbacks_lock;
    std::vector<cvk_context_callback> m_destuctor_callbacks;
    std::vector<cl_context_properties> m_properties;
    size_t m_printf_buffersize;
    cvk_printf_callback_t m_printf_callback;
    void* m_user_data;
};

static inline cvk_context* icd_downcast(cl_context context) {
    return static_cast<cvk_context*>(context);
}

using cvk_context_holder = refcounted_holder<cvk_context>;

template <object_magic magic>
struct api_object : public refcounted, object_magic_header<magic> {

    api_object(cvk_context* context) : m_context(context) {}
    cvk_context* context() const { return m_context; }

protected:
    cvk_context_holder m_context;
};
