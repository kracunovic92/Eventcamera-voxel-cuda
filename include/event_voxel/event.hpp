#pragma once

#include <cstdint>

namespace event_voxel {

struct Event {

    std::uint16_t x;
    std::uint16_t y;
    std::int64_t timestamp_us;
    std::uint8_t polarity;
};
} // namespace event_voxel
