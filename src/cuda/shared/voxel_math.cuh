#pragma once

#include "event_voxel/event.hpp"

#include <cstddef>
#include <cstdint>

namespace event_voxel::cuda::shared {

struct TemporalContribution {
    std::size_t left_bin;
    std::size_t right_bin;

    float left_weight;
    float right_weight;
};

__device__ inline TemporalContribution interpolate_timestamp(const Event& event,
                                                             std::int64_t start_us,
                                                             std::size_t last_bin,
                                                             double time_scale) {
    const std::int64_t relative_time = event.timestamp_us - start_us;

    double position = static_cast<double>(relative_time) * time_scale;

    // Defensive protection against floating-point
    // rounding around the boundaries.
    if (position < 0.0) {
        position = 0.0;
    }

    const double max_position = static_cast<double>(last_bin);

    if (position > max_position) {
        position = max_position;
    }

    const std::size_t left_bin = static_cast<std::size_t>(position);

    const std::size_t right_bin = left_bin == last_bin ? left_bin : left_bin + 1;

    const float right_weight = static_cast<float>(position - static_cast<double>(left_bin));

    const float left_weight = 1.0F - right_weight;

    return {
        .left_bin = left_bin,
        .right_bin = right_bin,
        .left_weight = left_weight,
        .right_weight = right_weight,
    };
}

__device__ inline float polarity_value(const Event& event) {
    return event.polarity == 1 ? 1.0F : -1.0F;
}

} // namespace event_voxel::cuda::shared
