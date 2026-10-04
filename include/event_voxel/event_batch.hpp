
#pragma once

#include <cstdint>
#include <vector>

#include "event_voxel/event.hpp"

namespace event_voxel {

struct EventBatch {
    std::vector<Event> events;

    std::uint32_t width;
    std::uint32_t height;

    std::int64_t start_us;
    std::int64_t end_us;
};

} // namespace event_voxel
