
#include "event_voxel/cpu/reference_voxelizer.hpp"
#include "event_voxel/validation.hpp"
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

// Step 3: Calculate where an event contributes in time.
//
// The caller supplies a validated event, configuration
// and precomputed conversion factor.
[[nodiscard]]
TemporalContribution interpolate_timestamp(const Event& event, const EventWindowView& batch,
                                           std::size_t last_bin, double time_scale) {
    const auto relative_time = event.timestamp_us - batch.start_us;

    const double position = std::clamp(static_cast<double>(relative_time) * time_scale, 0.0,
                                       static_cast<double>(last_bin));

    const auto left_bin = static_cast<std::size_t>(position);

    const auto right_bin = left_bin == last_bin ? left_bin : left_bin + 1;

    const float right_weight = static_cast<float>(position - static_cast<double>(left_bin));

    const float left_weight = 1.0f - right_weight;

    return {
        .left_bin = left_bin,
        .right_bin = right_bin,
        .left_weight = left_weight,
        .right_weight = right_weight,
    };
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
    return voxelize(view_of(batch), spec);
}

VoxelGrid CpuReferenceVoxelizer::voxelize(const EventWindowView& batch,
                                          const VoxelizationSpec& spec) {

    event_voxel::validate_window(batch, spec);
    return voxelize_prevalidated(batch, spec);
}

VoxelGrid CpuReferenceVoxelizer::voxelize_prevalidated(const EventWindowView& batch,
                                                       const VoxelizationSpec& spec) {

    const VoxelGridSpec grid_spec{
        .width = batch.width,
        .height = batch.height,
        .bins = spec.bins,
    };

    VoxelGrid grid(grid_spec);

    const auto duration_us = batch.end_us - batch.start_us;

    const auto last_bin = spec.bins - 1;

    const double time_scale = static_cast<double>(last_bin) / static_cast<double>(duration_us);

    for (const Event& event : batch.events) {
        const auto contribution = interpolate_timestamp(event, batch, last_bin, time_scale);

        accumulate_event(event, contribution, grid);
    }

    return grid;
}

} // namespace event_voxel
