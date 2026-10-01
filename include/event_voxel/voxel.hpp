#pragma once

#include "event_voxel/event.hpp"

#include <cstddef>

namespace event_voxel {

[[nodiscard]]
constexpr std::size_t voxel_index(int bin, int y, int x, int height, int width) {
    return (static_cast<std::size_t>(bin) * static_cast<std::size_t>(height) +
            static_cast<std::size_t>(y)) *
               static_cast<std::size_t>(width) +
           static_cast<std::size_t>(x);
}

} // namespace event_voxel
