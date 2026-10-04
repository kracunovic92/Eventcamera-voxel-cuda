
#include "event_voxel/synthetic_generator.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace event_voxel {

std::vector<Event> generate_events(const GeneratorConfig& config) {
    if (config.width == 0 || config.height == 0 || config.duration_us <= 0 ||
        !std::isfinite(config.hotspot_probability) || !std::isfinite(config.hotspot_size) ||
        config.hotspot_probability < 0.0 || config.hotspot_probability > 1.0 ||
        config.hotspot_size <= 0.0 || config.hotspot_size > 1.0) {
        throw std::invalid_argument("Invalid synthetic generator configuration");
    }

    std::mt19937 rng(config.seed);

    std::uniform_int_distribution<int> x_dist(0, config.width - 1);
    std::uniform_int_distribution<int> y_dist(0, config.height - 1);

    std::uniform_int_distribution<std::int64_t> time_dist(0, config.duration_us - 1);

    std::bernoulli_distribution polarity_dist(0.5);
    std::bernoulli_distribution hotspot_dist(config.hotspot_probability);

    const int hot_width = std::max(1, static_cast<int>(config.width * config.hotspot_size));

    const int hot_height = std::max(1, static_cast<int>(config.height * config.hotspot_size));

    const int x_min = (config.width - hot_width) / 2;
    const int y_min = (config.height - hot_height) / 2;

    std::uniform_int_distribution<int> hot_x_dist(x_min, x_min + hot_width - 1);

    std::uniform_int_distribution<int> hot_y_dist(y_min, y_min + hot_height - 1);

    std::vector<Event> events;
    events.reserve(config.event_count);

    for (std::size_t i = 0; i < config.event_count; ++i) {

        int x = x_dist(rng);
        int y = y_dist(rng);

        if (config.distribution == SpatialDistribution::Hotspot && hotspot_dist(rng)) {
            x = hot_x_dist(rng);
            y = hot_y_dist(rng);
        }

        events.push_back({static_cast<std::uint16_t>(x), static_cast<std::uint16_t>(y),
                          time_dist(rng), static_cast<std::uint8_t>(polarity_dist(rng) ? 1 : 0)});
    }

    std::stable_sort(events.begin(), events.end(), [](const Event& a, const Event& b) {
        return a.timestamp_us < b.timestamp_us;
    });

    return events;
}

} // namespace event_voxel
