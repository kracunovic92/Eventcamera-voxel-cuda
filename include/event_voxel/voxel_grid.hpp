#pragma once

#include <cstddef>

namespace event_voxel {
struct VoxelGridSpec {

    std::size_t width;
    std::size_t height;
    std::size_t bins;

    [[nodiscard]]
    constexpr std::size_t element_count() const noexcept {
        return width * height * bins;
    }
};

[[nodiscard]]
constexpr std::size_t voxel_index(const VoxelGridSpec& grid, std::size_t bin, std::size_t y,
                                  std::size_t x) noexcept {
    return (bin * grid.height + y) * grid.width + x;
}

} // namespace event_voxel
