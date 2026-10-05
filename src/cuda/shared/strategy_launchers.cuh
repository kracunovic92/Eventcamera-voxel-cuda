#pragma once

#include "event_voxel/event.hpp"

#include <cstddef>
#include <cstdint>

namespace event_voxel::cuda::shared {

struct VoxelizationParams {
    std::size_t event_count;
    std::size_t width;
    std::size_t height;
    std::size_t bins;
    std::size_t voxel_count;
    std::int64_t start_us;
    double time_scale;
};

void run_event_scatter(const Event* device_events, float* device_voxels,
                       const VoxelizationParams& params);

void run_pixel_gather(const Event* device_events, float* device_voxels,
                      const VoxelizationParams& params);

void run_sort_reduce(const Event* device_events, float* device_voxels,
                     const VoxelizationParams& params);

} // namespace event_voxel::cuda::shared
