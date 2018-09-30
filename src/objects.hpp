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
#include "device.hpp"
#include "utils.hpp"

struct refcounted {

    refcounted() : m_refcount(1) { }

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

    unsigned int refcount() const {
        return m_refcount.load();
    }

private:
    std::atomic<unsigned int> m_refcount;
};

template<typename T>
struct refcounted_holder {

    refcounted_holder(refcounted *refc) : m_refcounted(refc) {
        m_refcounted->retain();
    }

    ~refcounted_holder() {
        m_refcounted->release();
    }

    T* operator->() {
        return static_cast<T*>(m_refcounted);
    }

    operator T*() {
        return static_cast<T*>(m_refcounted);
    }

private:
    refcounted *m_refcounted;
};

typedef struct _cl_context : public refcounted {

    _cl_context(cvk_device *device, const cl_context_properties* props)
        : m_device(device) {

        if (props) {
            while (*props) {
                // Save name
                m_properties.push_back(*props);
                // Save value
                m_properties.push_back(*(props+1));
                props += 2;
            }
            m_properties.push_back(*props);
        }
    }

    virtual ~_cl_context() {}

    const std::vector<cl_context_properties>& properties() const {
        return m_properties;
    }

    cvk_device* device() const { return m_device; }
    unsigned num_devices() const { return 1u; }

private:
    cvk_device *m_device;
    std::vector<cl_context_properties> m_properties;

} cvk_context;

struct api_object : public refcounted {

    api_object(cvk_context *context) : m_context(context) {
        m_context->retain();
    }
    ~api_object() {
        m_context->release();
    }

    cvk_context* context() const { return m_context; }

protected:
    cvk_context *m_context;
};

