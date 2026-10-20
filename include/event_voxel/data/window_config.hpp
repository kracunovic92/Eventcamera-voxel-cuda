#pragma once

#include <cstddef>
#include <cstdint>

namespace event_voxel::data {

// Origin is the timestamp of the first CD event. The last partial interval is emitted.
struct WindowConfig {
    std::int64_t window_us = 100'000;
    std::int64_t chunk_us = 10'000'000;
    std::size_t max_chunk_events = 1'000'000;
    std::size_t max_window_events = 1'000'000;
};

} // namespace event_voxel::data
