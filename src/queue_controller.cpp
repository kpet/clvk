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
      m_no_batch_in_flight_since_last_flush(false) {
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

void cvk_queue_controller_batch_parameters::
    update_after_end_current_command_batch(bool from_flush) {
    TRACE_FUNCTION();
    auto reset_after_flush = [this]() {
        if (m_queue->m_nb_batch_in_flight > 1 &&
            !m_no_batch_in_flight_since_last_flush) {
            // Increase max_cmd_batch_size if there was always batches in flight
            // since last flush.
            m_queue->m_max_cmd_batch_size += m_queue->m_nb_batch_in_flight;
        }
        // Reset after flush
        m_last_batch_size = 0;
        m_no_batch_in_flight_since_last_flush = false;
    };
    auto trace = [this]() {
        TRACE_CNT(max_cmd_batch_size_counter, m_queue->m_max_cmd_batch_size);
        TRACE_CNT(max_first_cmd_batch_size_counter,
                  m_queue->m_max_first_cmd_batch_size);
        TRACE_CNT(max_first_cmd_batch_size_limit_counter,
                  m_max_first_cmd_batch_size_limit);
        TRACE_CNT(max_first_cmd_batch_size_limit_hit_counter,
                  m_max_first_cmd_batch_size_limit_hit);
        TRACE_CNT(last_batch_size_counter, m_last_batch_size);
    };
    if (!m_queue->m_command_batch) {
        if (from_flush) {
            reset_after_flush();
        }
        trace();
        return;
    }
    auto batch_size = m_queue->m_command_batch->batch_size();
    if (m_last_batch_size == 0) {
        m_last_batch_size = batch_size;
        trace();
        return;
    }

    // update m_no_batch_in_flight_since_last_flush
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
               m_last_batch_size <= m_queue->m_max_first_cmd_batch_size) {
        // Commands in flight and the last batch was smaller that
        // max_first_cmd_batch_size. Make smaller first batch to try to reduce
        // the latency.
        m_queue->m_max_first_cmd_batch_size -= 1;
    }

    if (from_flush) {
        reset_after_flush();
    } else {
        m_last_batch_size = batch_size;
    }

    // Do not increate m_max_cmd_batch_size over the initial value
    if (m_queue->m_max_cmd_batch_size > m_max_cmd_batch_size_limit) {
        m_queue->m_max_cmd_batch_size = m_max_cmd_batch_size_limit;
    }
    // Do not decrease m_max_first_cmd_batch_size under the limit.
    // After 4 tries, reset the limit to 1, allowing any possitive value one
    // time.
    if (m_queue->m_max_first_cmd_batch_size <
        m_max_first_cmd_batch_size_limit) {
        m_max_first_cmd_batch_size_limit_hit++;
        if (m_max_first_cmd_batch_size_limit_hit == 4) {
            m_max_first_cmd_batch_size_limit_hit = 0;
            m_max_first_cmd_batch_size_limit = 1;
        } else {
            m_queue->m_max_first_cmd_batch_size =
                m_max_first_cmd_batch_size_limit;
        }
    }
    // max_first_cmd_batch_size should not get bigger than max_cmd_batch_size.
    if (m_queue->m_max_cmd_batch_size < m_queue->m_max_first_cmd_batch_size) {
        m_queue->m_max_first_cmd_batch_size = m_queue->m_max_cmd_batch_size;
    }
    trace();
}
