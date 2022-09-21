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

#pragma once

#ifdef CLVK_PERFETTO_ENABLE

#include "cl_headers.hpp"
#include "log.hpp"
#include "perfetto.h"

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("clvk").SetDescription("CLVK Events"));

void init_tracing();
void term_tracing();

#define TRACE_FUNCTION(...)                                                    \
    perfetto::StaticString __perfetto_fct_name = __func__;                     \
    TRACE_EVENT("clvk", __perfetto_fct_name, ##__VA_ARGS__)
#define TRACE_BEGIN_CMD(cmd_type, ...)                                         \
    TRACE_EVENT_BEGIN(                                                         \
        "clvk", perfetto::StaticString(cl_command_type_to_string(cmd_type)),   \
        ##__VA_ARGS__)
#define TRACE_BEGIN_EVENT(cmd_type, ...)                                       \
    TRACE_EVENT_BEGIN(                                                         \
        "clvk", "event_wait", "cl_command_type",                               \
        perfetto::StaticString(cl_command_type_to_string(cmd_type)),           \
        ##__VA_ARGS__)
#define TRACE_BEGIN(name, ...) TRACE_EVENT_BEGIN("clvk", name, ##__VA_ARGS__)
#define TRACE_END() TRACE_EVENT_END("clvk")

#define TRACE_CNT(counter, value) TRACE_COUNTER("clvk", *counter, value)
#define TRACE_CNT_VAR(name)                                                    \
    std::string string_##name;                                                 \
    std::unique_ptr<perfetto::CounterTrack> name
#define TRACE_CNT_VAR_INIT(name, value)                                        \
    string_##name = value;                                                     \
    name = std::make_unique<perfetto::CounterTrack>(string_##name.c_str())

#else // CLVK_PERFETTO_ENABLE

#define init_tracing()
#define term_tracing()

#define TRACE_FUNCTION(...)
#define TRACE_BEGIN_CMD(cmd_type, ...)
#define TRACE_BEGIN_EVENT(cmd_type, ...)
#define TRACE_BEGIN(name, ...)
#define TRACE_END()

#define TRACE_CNT(str, value)
#define TRACE_CNT_VAR(name)
#define TRACE_CNT_VAR_INIT(name, value)

#endif // CLVK_PERFETTO_ENABLE
