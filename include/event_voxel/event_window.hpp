#pragma once

#include <cstdint>
#include <span>

#include "event_voxel/event.hpp"
#include "event_voxel/event_batch.hpp"

namespace event_voxel {

// Borrowed events. A source's view remains valid only until its next next() call.
struct EventWindowView {
    std::span<const Event> events;
    std::uint32_t width;
    std::uint32_t height;
    std::int64_t start_us;
    std::int64_t end_us;
    std::uint64_t sequence = 0;
    bool partial_final = false;
};

[[nodiscard]] inline EventWindowView view_of(const EventBatch& batch) noexcept {
    return {batch.events, batch.width, batch.height, batch.start_us, batch.end_us};
}

} // namespace event_voxel
