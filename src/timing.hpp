// Copyright 2020 The clvk authors.
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

#include <atomic>
#include <chrono>

#include "log.hpp"

class cvk_timer {
public:
    cvk_timer(std::string name) : m_name(name) {}

#ifdef CVK_ENABLE_TIMING

    ~cvk_timer() {
        double average = m_count ? (m_total_time / 1.0e6) / m_count : 0.0;
        cvk_warn("%.2f ms -> %s (%d blocks, avg %.3f ms)", m_total_time / 1.0e6,
                 m_name.c_str(), m_count.load(), average);
    }

    void add_time(long nanoseconds) {
        m_total_time += nanoseconds;
        m_count++;
    }

#endif

private:
    std::string m_name;
#ifdef CVK_ENABLE_TIMING
    std::atomic<long> m_total_time{0};
    std::atomic<int> m_count{0};
#endif
};

class cvk_unscoped_timer {
public:
    cvk_unscoped_timer(cvk_timer& timer) : m_timer(timer) {}

    void start() {
#ifdef CVK_ENABLE_TIMING
        m_start_time = std::chrono::steady_clock::now();
#endif
    }

    void stop() {
#ifdef CVK_ENABLE_TIMING
        auto elapsed = std::chrono::steady_clock::now() - m_start_time;
        auto nanoseconds =
            std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed);
        m_timer.add_time(nanoseconds.count());
#endif
    }

private:
    [[maybe_unused]] cvk_timer& m_timer;
#ifdef CVK_ENABLE_TIMING
    std::chrono::steady_clock::time_point m_start_time;
#endif
};

class cvk_scoped_timer {
public:
    cvk_scoped_timer(cvk_timer& timer) : m_timer(timer) {
#ifdef CVK_ENABLE_TIMING
        m_timer.start();
#endif
    }
    ~cvk_scoped_timer() {
#ifdef CVK_ENABLE_TIMING
        m_timer.stop();
#endif
    }

private:
    cvk_unscoped_timer m_timer;
};

#define CVK_TIMED_BLOCK(name, description)                                     \
    static cvk_timer name##timer(description);                                 \
    cvk_scoped_timer name##timed_scope(name##timer)

#define CVK_TIMED_FUNCTION CVK_TIMED_BLOCK(__func__, __func__)

#define CVK_UNSCOPED_TIMER(name, description)                                  \
    static cvk_timer name##timer(description);                                 \
    cvk_unscoped_timer name(name##timer);                                      \
    name
