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
    validate_configuration(batch, spec);

    for (const Event& event : batch.events) {
        validate_event(event, batch);
    }
}

} // namespace event_voxel
