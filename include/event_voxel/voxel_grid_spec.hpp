#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace event_voxel {

struct VoxelGridSpec {
    std::size_t width;
    std::size_t height;
    std::size_t bins;

    [[nodiscard]]
    constexpr std::size_t element_count() const {
        if (height != 0 && width > std::numeric_limits<std::size_t>::max() / height) {
            throw std::overflow_error("Grid pixel count overflow");
        }
        const auto pixels = width * height;
        if (bins != 0 && pixels > std::numeric_limits<std::size_t>::max() / bins) {
            throw std::overflow_error("Grid element count overflow");
        }
        return width * height * bins;
    }

    [[nodiscard]]
    constexpr std::size_t memory_bytes() const {
        const auto elements = element_count();
        if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            throw std::overflow_error("Grid byte count overflow");
        }
        return elements * sizeof(float);
    }
};

[[nodiscard]]
constexpr std::size_t voxel_index(const VoxelGridSpec& spec, std::size_t bin, std::size_t y,
                                  std::size_t x) noexcept {
    return (bin * spec.height + y) * spec.width + x;
}
} // namespace event_voxel
