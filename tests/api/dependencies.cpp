// Copyright 2019 The clvk authors.
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

#include "testcl.hpp"

static const size_t BUFFER_SIZE = 1024;

TEST_F(WithCommandQueue, FailedAndCompleteDependencies) {
    // Create buffer
    auto buffer = CreateBuffer(CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                               BUFFER_SIZE, nullptr);

    // Create two user events
    auto event1 = CreateUserEvent();
    auto event2 = CreateUserEvent();

    // Complete one and fail the other
    SetUserEventStatus(event1, CL_INVALID_OPERATION);
    SetUserEventStatus(event2, CL_COMPLETE);

    cl_int err;

    // Test fail then complete
    std::vector<cl_event> failedThenComplete = {event1, event2};
    clEnqueueMapBuffer(m_queue, buffer, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION,
                       0, BUFFER_SIZE, failedThenComplete.size(),
                       failedThenComplete.data(), nullptr, &err);

    ASSERT_EQ(err, CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);

    // Test complete then fail
    std::vector<cl_event> completeThenFailed = {event2, event1};
    clEnqueueMapBuffer(m_queue, buffer, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION,
                       0, BUFFER_SIZE, completeThenFailed.size(),
                       completeThenFailed.data(), nullptr, &err);

    ASSERT_EQ(err, CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
}

TEST_F(WithCommandQueue, InOrderQueueStopsExecutionAfterFailedCommand) {
    auto buffer = CreateBuffer(CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                               BUFFER_SIZE, nullptr);

    // Create a user event in a terminated status
    auto uevent = CreateUserEvent();
    SetUserEventStatus(uevent, CL_INVALID_OPERATION);

    // Enqueue a command that depends on it
    cl_int err;
    std::vector<cl_event> dependencies = {uevent};
    clEnqueueMapBuffer(m_queue, buffer, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION,
                       0, BUFFER_SIZE, dependencies.size(), dependencies.data(),
                       nullptr, &err);

    // Check the enqueue fails
    ASSERT_EQ(err, CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);

    // Enqueue another command with no dependencies to the same queue. The queue
    // is in-order
    cl_event mapev;
    clEnqueueMapBuffer(m_queue, buffer, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION,
                       0, BUFFER_SIZE, 0, nullptr, &mapev, &err);
    cl_int status;
    GetEventInfo(mapev, CL_EVENT_COMMAND_EXECUTION_STATUS, &status);
    ASSERT_NE(status, CL_COMPLETE);
}
