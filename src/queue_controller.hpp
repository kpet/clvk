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

#pragma once

#include "queue.hpp"

struct cvk_queue_controller {
    cvk_queue_controller(cvk_command_queue* queue) : m_queue(queue) {}

    virtual ~cvk_queue_controller() {}

    virtual void update_after_end_current_command_batch(bool from_flush) {
        (void)from_flush;
    }

    virtual void update_after_empty_flush() {}

protected:
    cvk_command_queue* m_queue;
};

struct cvk_queue_controller_batch_parameters : public cvk_queue_controller {
    cvk_queue_controller_batch_parameters(cvk_command_queue* queue);

    void update_after_end_current_command_batch(bool from_flush) override final;

    void update_after_empty_flush() override final;

private:
    void update_trace_counter();
    void reset_after_flush();

    cl_uint m_max_cmd_batch_size_limit;
    cl_uint m_max_first_cmd_batch_size_limit;
    cl_uint m_max_first_cmd_batch_size_limit_hit;
    cl_uint m_last_batch_size;
    bool m_no_batch_in_flight_since_last_flush;
    bool m_executor_idle_since_last_flush;

    TRACE_CNT_VAR(max_cmd_batch_size_counter);
    TRACE_CNT_VAR(max_first_cmd_batch_size_counter);
    TRACE_CNT_VAR(max_first_cmd_batch_size_limit_counter);
    TRACE_CNT_VAR(max_first_cmd_batch_size_limit_hit_counter);
    TRACE_CNT_VAR(last_batch_size_counter);
};
