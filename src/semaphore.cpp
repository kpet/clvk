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

#include "semaphore.hpp"

cl_int cvk_semaphore::init() {

    auto vkdev = m_context->device()->vulkan_device();

    VkSemaphoreCreateInfo info = {
        VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, nullptr,
        0 // flags
    };

    auto res = vkCreateSemaphore(vkdev, &info, nullptr, &m_semaphore);
    if (res != VK_SUCCESS) {
        return CL_OUT_OF_RESOURCES;
    }

    return CL_SUCCESS;
}
