// Copyright 2024 The clvk authors.
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

#include "context.hpp"
#include "queue.hpp"

cvk_command_queue* cvk_context::get_or_create_image_init_command_queue() {
    std::unique_lock<std::mutex> lock(m_queue_image_init_lock);
    if (m_queue_image_init != nullptr) {
        return m_queue_image_init;
    }
    std::vector<cl_queue_properties> properties_array;
    m_queue_image_init =
        new cvk_command_queue(this, m_device, 0, std::move(properties_array));
    cl_int ret = m_queue_image_init->init();
    if (ret != CL_SUCCESS) {
        return nullptr;
    }
    m_queue_image_init->detach_from_context();
    return m_queue_image_init;
}

void cvk_context::free_image_init_command_queue() { delete m_queue_image_init; }
