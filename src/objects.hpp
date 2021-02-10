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

#pragma once

#include <cstring>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <list>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

#include "cl_headers.hpp"
#include "icd.hpp"
#include "log.hpp"

enum class object_magic : uint32_t
{
    platform = 0x11223344U,
    device = 0x22334455U,
    context = 0x33445566U,
    command_queue = 0x44556677U,
    event = 0x55667788U,
    program = 0x66778899U,
    kernel = 0x778899AAU,
    memory_object = 0x8899AABBU,
    sampler = 0x99AABBCCU,
    semaphore = 0xAABBCCDDU,
};

template <object_magic magic> struct object_magic_header {
    object_magic_header() : m_magic(magic) {}
    bool is_valid() const { return m_magic == magic; }

private:
    object_magic m_magic;
};

struct refcounted {

    refcounted() : m_refcount(1) {}

    virtual ~refcounted() = default;

    void retain() {
        unsigned int refcount = m_refcount.fetch_add(1);
        cvk_debug_fn("obj = %p, refcount now %u", this, refcount + 1);
    }

    void release() {
        unsigned int refcount = m_refcount.fetch_sub(1);
        cvk_debug_fn("obj = %p, refcount now %u", this, refcount - 1);

        if (refcount == 1) {
            delete this;
        }
    }

    unsigned int refcount() const { return m_refcount.load(); }

private:
    std::atomic<unsigned int> m_refcount;
};

template <typename T> struct refcounted_holder {

    refcounted_holder() : m_refcounted(nullptr) {}

    refcounted_holder(T* refcounted) : m_refcounted(refcounted) {
        if (m_refcounted != nullptr) {
            m_refcounted->retain();
        }
    }

    refcounted_holder(const refcounted_holder& other)
        : m_refcounted(other.m_refcounted) {
        if (m_refcounted != nullptr) {
            m_refcounted->retain();
        }
    }

    refcounted_holder(refcounted_holder&& other) noexcept
        : m_refcounted(other.m_refcounted) {
        other.m_refcounted = nullptr;
    }

    ~refcounted_holder() {
        if (m_refcounted != nullptr) {
            m_refcounted->release();
        }
    }

    T* operator->() const { return m_refcounted; }

    operator T*() const { return m_refcounted; }

    refcounted_holder& operator=(const refcounted_holder&) = delete;
    refcounted_holder& operator=(const refcounted_holder&&) = delete;

    void reset(T* refc) {
        if (m_refcounted != nullptr) {
            m_refcounted->release();
        }
        m_refcounted = refc;
        if (m_refcounted != nullptr) {
            m_refcounted->retain();
        }
    }

private:
    T* m_refcounted;
};

using cvk_context_callback_pointer_type = void(CL_CALLBACK*)(cl_context context,
                                                             void* user_data);
struct cvk_context_callback {
    cvk_context_callback_pointer_type pointer;
    void* data;
};

struct cvk_device;

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
