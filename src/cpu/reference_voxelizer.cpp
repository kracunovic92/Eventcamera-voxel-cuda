
#include "event_voxel/cpu/reference_voxelizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace event_voxel {
namespace {

// Represents the temporal contribution of one event.
// Spatial coordinates and polarity remain in the Event.
struct TemporalContribution {
    std::size_t left_bin;
    std::size_t right_bin;

    float left_weight;
    float right_weight;
};

// Step 1: Validate the batch metadata and configuration.
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

// Step 2: Validate an individual event.
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

// Step 3: Calculate where an event contributes in time.
//
// The caller supplies a validated event, configuration
// and precomputed conversion factor.
[[nodiscard]]
TemporalContribution interpolate_timestamp(const Event& event, const EventBatch& batch,
                                           const VoxelizationSpec& spec, double time_scale) {
    // Safe because validated timestamps satisfy:
    // 0 <= start_us <= timestamp_us < end_us.
    const auto relative_time = event.timestamp_us - batch.start_us;

    // Map recording time to continuous bin coordinates.
    const double position = std::clamp(static_cast<double>(relative_time) * time_scale, 0.0,
                                       static_cast<double>(spec.bins - 1));

    const auto left_bin = static_cast<std::size_t>(std::floor(position));

    // Avoid overflowing left_bin + 1 at the last bin.
    const auto right_bin = left_bin == spec.bins - 1 ? left_bin : left_bin + 1;

    const float right_weight = static_cast<float>(position - left_bin);

    const float left_weight = 1.0f - right_weight;

    return TemporalContribution{.left_bin = left_bin,
                                .right_bin = right_bin,
                                .left_weight = left_weight,
                                .right_weight = right_weight};
}

// Step 4: Accumulate the weighted event contributions.
void accumulate_event(const Event& event, const TemporalContribution& contribution,
                      VoxelGrid& grid) {
    const float polarity = event.polarity == 1 ? 1.0f : -1.0f;

    grid.accumulate(contribution.left_bin, event.y, event.x, polarity * contribution.left_weight);

    // An event at the final temporal bin has no
    // distinct right-hand bin.
    if (contribution.right_bin != contribution.left_bin && contribution.right_weight > 0.0f) {
        grid.accumulate(contribution.right_bin, event.y, event.x,
                        polarity * contribution.right_weight);
    }
}

} // namespace

// Public entry point: coordinates the complete algorithm.
VoxelGrid CpuReferenceVoxelizer::voxelize(const EventBatch& batch, const VoxelizationSpec& spec) {
    // 1. Validate input before allocating the output.
    validate_configuration(batch, spec);

    for (const Event& event : batch.events) {
        validate_event(event, batch);
    }

    // 2. Allocate a zero-initialized output grid.
    const VoxelGridSpec grid_spec{.width = batch.width, .height = batch.height, .bins = spec.bins};

    VoxelGrid grid(grid_spec);

    // 3. Calculate the time conversion factor once.
    const auto duration_us = batch.end_us - batch.start_us;

    const double time_scale = static_cast<double>(spec.bins - 1) / static_cast<double>(duration_us);

    // 4. Calculate and accumulate each event's
    //    temporal contribution.
    for (const Event& event : batch.events) {
        const auto contribution = interpolate_timestamp(event, batch, spec, time_scale);

        accumulate_event(event, contribution, grid);
    }

    return grid;
}

} // namespace event_voxel
