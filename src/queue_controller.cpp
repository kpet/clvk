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

#include "queue_controller.hpp"
#include "queue.hpp"

cvk_queue_controller_batch_parameters::cvk_queue_controller_batch_parameters(
    cvk_command_queue* queue)
    : cvk_queue_controller(queue),
      m_max_cmd_batch_size_limit(queue->device()->get_max_cmd_batch_size()),
      m_max_first_cmd_batch_size_limit(
          queue->device()->get_max_first_cmd_batch_size()),
      m_max_first_cmd_batch_size_limit_hit(0), m_last_batch_size(0),
      m_no_batch_in_flight_since_last_flush(false),
      m_executor_idle_since_last_flush(false) {
    TRACE_CNT_VAR_INIT(max_cmd_batch_size_counter,
                       "clvk-queue_" + std::to_string((uintptr_t)this) +
                           "-max_batch_size");
    TRACE_CNT_VAR_INIT(max_first_cmd_batch_size_counter,
                       "clvk-queue_" + std::to_string((uintptr_t)this) +
                           "-max_first_batch_size");
    TRACE_CNT_VAR_INIT(max_first_cmd_batch_size_limit_counter,
                       "clvk-queue_" + std::to_string((uintptr_t)this) +
                           "-max_first_batch_size_limit");
    TRACE_CNT_VAR_INIT(max_first_cmd_batch_size_limit_hit_counter,
                       "clvk-queue_" + std::to_string((uintptr_t)this) +
                           "-max_first_batch_size_limit_hit");
    TRACE_CNT_VAR_INIT(last_batch_size_counter,
                       "clvk-queue_" + std::to_string((uintptr_t)this) +
                           "-last_batch_size");

    TRACE_CNT(max_cmd_batch_size_counter, queue->m_max_cmd_batch_size);
    TRACE_CNT(max_first_cmd_batch_size_counter,
              queue->m_max_first_cmd_batch_size);
    TRACE_CNT(max_first_cmd_batch_size_limit_counter,
              m_max_first_cmd_batch_size_limit);
    TRACE_CNT(max_first_cmd_batch_size_limit_hit_counter, 0);
    TRACE_CNT(last_batch_size_counter, m_last_batch_size);
}

void cvk_queue_controller_batch_parameters::update_trace_counter() {
    TRACE_CNT(max_cmd_batch_size_counter, m_queue->m_max_cmd_batch_size);
    TRACE_CNT(max_first_cmd_batch_size_counter,
              m_queue->m_max_first_cmd_batch_size);
    TRACE_CNT(max_first_cmd_batch_size_limit_counter,
              m_max_first_cmd_batch_size_limit);
    TRACE_CNT(max_first_cmd_batch_size_limit_hit_counter,
              m_max_first_cmd_batch_size_limit_hit);
    TRACE_CNT(last_batch_size_counter, m_last_batch_size);
}

void cvk_queue_controller_batch_parameters::reset_after_flush() {
    if (m_queue->m_nb_batch_in_flight > 2 &&
        !m_no_batch_in_flight_since_last_flush) {
        // Increase max_cmd_batch_size if there was always batches in flight
        // since last flush.
        m_queue->m_max_cmd_batch_size += m_queue->m_nb_batch_in_flight;
    }
    // Reset after flush
    m_last_batch_size = 0;
    m_no_batch_in_flight_since_last_flush = false;
    m_executor_idle_since_last_flush = false;
}

void cvk_queue_controller_batch_parameters::update_after_empty_flush() {
    TRACE_FUNCTION();
    reset_after_flush();
    update_trace_counter();
}

void cvk_queue_controller_batch_parameters::
    update_after_end_current_command_batch(bool from_flush) {
    TRACE_FUNCTION();
    if (!m_queue->m_command_batch) {
        if (from_flush) {
            reset_after_flush();
        }
        update_trace_counter();
        return;
    }

    // update m_executor_idle_since_last_flush
    if (m_queue->m_executor != nullptr) {
        m_executor_idle_since_last_flush |= m_queue->m_executor->is_idle();
    }

    auto batch_size = m_queue->m_command_batch->batch_size();
    if (m_last_batch_size == 0) {
        m_last_batch_size = batch_size;
        update_trace_counter();
        return;
    }

    // update m_no_batch_in_flight_since_last_flush
    // It needs to be after m_last_batch_size is first initialized as we do not
    // care of what is happening before the first batch.
    m_no_batch_in_flight_since_last_flush |= m_queue->m_nb_batch_in_flight == 0;

    if (m_queue->m_nb_batch_in_flight == 0 &&
        m_last_batch_size == m_queue->m_max_first_cmd_batch_size) {
        // Nothing in flight and we flush because of first_cmd_batch_size
        // reached. It should only happen the first time. Otherwise it means
        // that our batches were too small to have the time to enqueue. Let's
        // increase the first_cmd_batch_size_limit, reset the
        // first_cmd_batch_size_limit_hit, increase first_cmd_batch_size to
        // create bigger batch, and reset max_cmd_batch_size to make many small
        // (size of first) batches to begin with.
        m_max_first_cmd_batch_size_limit =
            m_queue->m_max_first_cmd_batch_size + 1;
        m_max_first_cmd_batch_size_limit_hit = 0;
        m_queue->m_max_first_cmd_batch_size += 5;
        m_queue->m_max_cmd_batch_size = m_queue->m_max_first_cmd_batch_size;
    } else if (m_queue->m_nb_batch_in_flight == 0 &&
               m_queue->m_max_cmd_batch_size >=
                   m_queue->m_max_first_cmd_batch_size + 2) {
        // Nothing in flight and we flush because of either a flush or
        // max_cmd_batch_size has been reached. Decrease max_cmd_batch_size if
        // it does not go under max_first_cmd_batch_size to try to create batch
        // before the end of the first batch.
        m_queue->m_max_cmd_batch_size -= 2;
    } else if (m_queue->m_nb_batch_in_flight > 0 &&
               m_last_batch_size <= m_queue->m_max_first_cmd_batch_size &&
               m_executor_idle_since_last_flush) {
        // Commands in flight and the last batch was smaller that
        // max_first_cmd_batch_size and the executor has been idle. Make smaller
        // first batch to try to reduce the latency.
        m_queue->m_max_first_cmd_batch_size -= 1;
    }

    if (from_flush) {
        if (!m_executor_idle_since_last_flush &&
            m_max_first_cmd_batch_size_limit_hit > 0) {
            // Executor has always been busy.
            // Decrease limit_hot to avoid first batch to decrease when we can
            // keep it as big as possible.
            m_max_first_cmd_batch_size_limit_hit--;
        }
        reset_after_flush();
    } else {
        m_last_batch_size = batch_size;
    }

    // Do not increate m_max_cmd_batch_size over the initial value
    if (m_queue->m_max_cmd_batch_size > m_max_cmd_batch_size_limit) {
        m_queue->m_max_cmd_batch_size = m_max_cmd_batch_size_limit;
    }
    // Do not decrease m_max_first_cmd_batch_size under the limit.
    if (m_queue->m_max_first_cmd_batch_size <
        m_max_first_cmd_batch_size_limit) {
        m_max_first_cmd_batch_size_limit_hit += 2;
        if (m_max_first_cmd_batch_size_limit_hit > 8 &&
            m_max_first_cmd_batch_size_limit > 1) {
            // Too many hit, reset limit hit and decrease limit
            m_max_first_cmd_batch_size_limit_hit = 0;
            m_max_first_cmd_batch_size_limit--;
        } else if (m_max_first_cmd_batch_size_limit_hit > 16 &&
                   m_max_first_cmd_batch_size_limit == 1) {
            // Too many hit with limit at minimum, we might have end up in a
            // corner case, let's reset limit to initial value
            m_max_first_cmd_batch_size_limit_hit = 0;
            m_max_first_cmd_batch_size_limit =
                m_queue->device()->get_max_first_cmd_batch_size();
        } else {
            m_queue->m_max_first_cmd_batch_size =
                m_max_first_cmd_batch_size_limit;
        }
    }
    // max_first_cmd_batch_size should not get bigger than max_cmd_batch_size.
    if (m_queue->m_max_cmd_batch_size < m_queue->m_max_first_cmd_batch_size) {
        m_queue->m_max_first_cmd_batch_size = m_queue->m_max_cmd_batch_size;
    }
    update_trace_counter();
}
