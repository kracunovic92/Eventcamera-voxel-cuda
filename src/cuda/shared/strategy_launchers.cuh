#pragma once

#include "event_voxel/event.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>

#include "device_buffer.cuh"

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

struct PixelGatherWorkspace {
    DeviceBuffer<std::uint32_t> counts, offsets, cursors;
    DeviceBuffer<Event> grouped;
    void reserve(std::size_t events, std::size_t pixels) {
        counts.reserve(pixels, "cudaMalloc(pixel counts)");
        offsets.reserve(pixels + 1, "cudaMalloc(pixel offsets)");
        cursors.reserve(pixels, "cudaMalloc(pixel cursors)");
        grouped.reserve(events, "cudaMalloc(grouped events)");
    }
};

struct SortReduceWorkspace {
    DeviceBuffer<std::uint64_t> keys, reduced_keys;
    DeviceBuffer<float> values, reduced_values;
    std::size_t reduced_count = 0;
    void reserve(std::size_t events) {
        if (events > std::numeric_limits<std::size_t>::max() / 2) {
            throw std::overflow_error("Contribution count overflow");
        }
        const auto contributions = events * 2;
        keys.reserve(contributions, "cudaMalloc(contribution keys)");
        values.reserve(contributions, "cudaMalloc(contribution values)");
        reduced_keys.reserve(contributions, "cudaMalloc(reduced keys)");
        reduced_values.reserve(contributions, "cudaMalloc(reduced values)");
    }
};

void run_event_scatter(const Event* device_events, float* device_voxels,
                       const VoxelizationParams& params, cudaStream_t stream,
                       unsigned int block_size);

void prepare_pixel_gather(const Event* device_events, const VoxelizationParams& params,
                          PixelGatherWorkspace& workspace, cudaStream_t stream,
                          unsigned int block_size);
void finish_pixel_gather(float* device_voxels, const VoxelizationParams& params,
                         PixelGatherWorkspace& workspace, cudaStream_t stream,
                         unsigned int block_size);

void prepare_sort_reduce(const Event* device_events, const VoxelizationParams& params,
                         SortReduceWorkspace& workspace, cudaStream_t stream,
                         unsigned int block_size);
void finish_sort_reduce(float* device_voxels, const VoxelizationParams& params,
                        SortReduceWorkspace& workspace, cudaStream_t stream,
                        unsigned int block_size);

} // namespace event_voxel::cuda::shared
