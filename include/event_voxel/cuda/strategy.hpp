#pragma once

#include <string_view>

namespace event_voxel::cuda {

enum class Strategy {
    EventScatter,
    PixelGather,
    SortReduce,
};

[[nodiscard]]
constexpr std::string_view to_string(Strategy strategy) noexcept {
    switch (strategy) {
    case Strategy::EventScatter:
        return "event-scatter";

    case Strategy::PixelGather:
        return "pixel-gather";

    case Strategy::SortReduce:
        return "sort-reduce";
    }

    return "unknown";
}

} // namespace event_voxel::cuda
