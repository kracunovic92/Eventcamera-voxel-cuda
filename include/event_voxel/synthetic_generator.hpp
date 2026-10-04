
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "event_voxel/event.hpp"

namespace event_voxel {

enum class SpatialDistribution { Uniform, Hotspot };

struct GeneratorConfig {
    std::uint16_t width = 640;
    std::uint16_t height = 480;

    std::size_t event_count = 10'000;
    std::int64_t duration_us = 100'000;

    std::uint32_t seed = 42;

    SpatialDistribution distribution = SpatialDistribution::Uniform;

    double hotspot_probability = 0.9;
    double hotspot_size = 0.1;
};

[[nodiscard]]
std::vector<Event> generate_events(const GeneratorConfig& config);

} // namespace event_voxel
