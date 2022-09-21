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

#include "traces.hpp"
#include "config.hpp"
#include "queue.hpp"

#ifdef CLVK_PERFETTO_ENABLE

PERFETTO_TRACK_EVENT_STATIC_STORAGE();

#ifdef CLVK_PERFETTO_BACKEND_INPROCESS
std::unique_ptr<perfetto::TracingSession> tracing_session;
#endif

void init_tracing() {
    perfetto::TracingInitArgs args;
#ifdef CLVK_PERFETTO_BACKEND_INPROCESS
    args.backends |= perfetto::kInProcessBackend;
#else
    args.backends |= perfetto::kSystemBackend;
#endif
    perfetto::Tracing::Initialize(args);
    perfetto::TrackEvent::Register();

#ifdef CLVK_PERFETTO_BACKEND_INPROCESS
    perfetto::protos::gen::TrackEventConfig track_event_cfg;
    perfetto::TraceConfig cfg;
    cfg.add_buffers()->set_size_kb(config.perfetto_max_trace_size);
    auto* ds_cfg = cfg.add_data_sources()->mutable_config();
    ds_cfg->set_name("track_event");
    ds_cfg->set_track_event_config_raw(track_event_cfg.SerializeAsString());

    tracing_session = perfetto::Tracing::NewTrace();
    tracing_session->Setup(cfg);
    tracing_session->StartBlocking();
#endif
}

void term_tracing() {
#ifdef CLVK_PERFETTO_BACKEND_INPROCESS
    tracing_session->StopBlocking();
    std::vector<char> trace_data(tracing_session->ReadTraceBlocking());

    std::ofstream output;
    output.open(config.perfetto_dest_trace, std::ios::out | std::ios::binary);
    output.write(&trace_data[0], trace_data.size());
    output.close();
#endif
}

#endif // CLVK_PERFETTO_ENABLE
