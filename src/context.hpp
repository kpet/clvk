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

using cvk_context_callback_pointer_type = void(CL_CALLBACK*)(cl_context context,
                                                             void* user_data);
struct cvk_context_callback {
    cvk_context_callback_pointer_type pointer;
    void* data;
};

struct cvk_context : public _cl_context,
                     refcounted,
                     object_magic_header<object_magic::context> {

    cvk_context(cvk_device* device, const cl_context_properties* props)
        : m_device(device) {

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

private:
    cvk_device* m_device;
    std::mutex m_callbacks_lock;
    std::vector<cvk_context_callback> m_destuctor_callbacks;
    std::vector<cl_context_properties> m_properties;
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
