#pragma once

#include <limits>
#include <stdexcept>
#include <vector>

#include "event_voxel/voxel_grid_spec.hpp"

namespace event_voxel {

class VoxelGrid {
  public:
    explicit VoxelGrid(VoxelGridSpec spec)
        : spec_(spec), data_(validated_element_count(spec), 0.0f) {}

    [[nodiscard]]
    const VoxelGridSpec& spec() const noexcept {
        return spec_;
    }

    [[nodiscard]]
    const std::vector<float>& data() const noexcept {
        return data_;
    }

    [[nodiscard]]
    std::vector<float>& data() noexcept {
        return data_;
    }

    [[nodiscard]]
    float at(std::size_t bin, std::size_t y, std::size_t x) const {
        validate_coordinates(bin, y, x);
        return data_[voxel_index(spec_, bin, y, x)];
    }

    void accumulate(std::size_t bin, std::size_t y, std::size_t x, float value) {
        validate_coordinates(bin, y, x);
        data_[voxel_index(spec_, bin, y, x)] += value;
    }

  private:
    static std::size_t validated_element_count(const VoxelGridSpec& spec) {
        if (spec.width == 0 || spec.height == 0 || spec.bins == 0) {
            throw std::invalid_argument("Voxel grid dimensions must be positive");
        }

        constexpr auto max = std::numeric_limits<std::size_t>::max();

        if (spec.width > max / spec.height) {
            throw std::overflow_error("Grid size overflow");
        }

        const auto pixels = spec.width * spec.height;

        if (spec.bins > max / pixels || pixels * spec.bins > max / sizeof(float)) {
            throw std::overflow_error("Grid size overflow");
        }

        return pixels * spec.bins;
    }

    void validate_coordinates(std::size_t bin, std::size_t y, std::size_t x) const {
        if (bin >= spec_.bins || y >= spec_.height || x >= spec_.width) {
            throw std::out_of_range("Voxel coordinates out of range");
        }
    }

    VoxelGridSpec spec_;
    std::vector<float> data_;
};

} // namespace event_voxel
