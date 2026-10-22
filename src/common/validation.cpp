#include "event_voxel/validation.hpp"

#include <stdexcept>

namespace event_voxel {

void validate_window(const EventWindowView& window, const VoxelizationSpec& spec) {
    if (window.width == 0 || window.height == 0 || window.start_us < 0 ||
        window.end_us <= window.start_us || spec.bins < 2) {
        throw std::invalid_argument("Invalid voxelization window or bin count");
    }
    std::int64_t previous = window.start_us;
    for (const Event& event : window.events) {
        if (event.x >= window.width || event.y >= window.height || event.polarity > 1 ||
            event.timestamp_us < previous || event.timestamp_us >= window.end_us) {
            throw std::invalid_argument("Invalid or unordered event in window");
        }
        previous = event.timestamp_us;
    }
}

} // namespace event_voxel
