#include "event_voxel/validation.hpp"

#include <stdexcept>

namespace event_voxel {

void validate_configuration(const EventBatch& batch, const VoxelizationSpec& spec) {
    if (batch.width == 0 || batch.height == 0) {
        throw std::invalid_argument("Sensor dimensions must be positive");
    }

    if (batch.start_us < 0 || batch.end_us <= batch.start_us) {
        throw std::invalid_argument("Invalid temporal window");
    }

    if (spec.bins < 2) {
        throw std::invalid_argument("At least two temporal bins are required");
    }
}

void validate_event(const Event& event, const EventBatch& batch) {
    if (event.x >= batch.width || event.y >= batch.height) {
        throw std::invalid_argument("Event coordinates outside sensor dimensions");
    }

    if (event.timestamp_us < batch.start_us || event.timestamp_us >= batch.end_us) {
        throw std::invalid_argument("Event timestamp outside temporal window");
    }

    if (event.polarity > 1) {
        throw std::invalid_argument("Event polarity must be 0 or 1");
    }
}

void validate_batch(const EventBatch& batch, const VoxelizationSpec& spec) {
    validate_window(view_of(batch), spec);
}

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
