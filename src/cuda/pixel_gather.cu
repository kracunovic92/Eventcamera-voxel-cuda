#include "shared/strategy_launchers.cuh"

#include "shared/cuda_check.cuh"
#include "shared/device_buffer.cuh"
#include "shared/voxel_math.cuh"

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace event_voxel::cuda::shared {

namespace {

__global__ void count_events_per_pixel_kernel(const Event* events, std::size_t event_count,
                                              std::uint32_t* pixel_counts, std::size_t width) {
    const std::size_t event_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (event_index >= event_count) {
        return;
    }

    const Event event = events[event_index];

    const std::size_t pixel =
        static_cast<std::size_t>(event.y) * width + static_cast<std::size_t>(event.x);

    atomicAdd(&pixel_counts[pixel], 1U);
}

__global__ void group_events_by_pixel_kernel(const Event* events, std::size_t event_count,
                                             Event* grouped_events, std::uint32_t* write_cursors,
                                             std::size_t width) {
    const std::size_t event_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (event_index >= event_count) {
        return;
    }

    const Event event = events[event_index];

    const std::size_t pixel =
        static_cast<std::size_t>(event.y) * width + static_cast<std::size_t>(event.x);

    const std::uint32_t destination = atomicAdd(&write_cursors[pixel], 1U);

    grouped_events[destination] = event;
}

__global__ void set_last_offset_kernel(std::uint32_t* offsets, std::size_t pixels,
                                       std::uint32_t event_count) {
    offsets[pixels] = event_count;
}

__global__ void pixel_gather_kernel(const Event* grouped_events, const std::uint32_t* pixel_offsets,
                                    float* voxels, shared::VoxelizationParams params) {
    const std::size_t pixel = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    const std::size_t pixel_count = params.width * params.height;

    if (pixel >= pixel_count) {
        return;
    }

    const std::uint32_t begin = pixel_offsets[pixel];

    const std::uint32_t end = pixel_offsets[pixel + 1];

    const std::size_t plane_size = pixel_count;

    for (std::uint32_t i = begin; i < end; ++i) {

        const Event event = grouped_events[i];

        const auto contribution = shared::interpolate_timestamp(event, params.start_us,
                                                                params.bins - 1, params.time_scale);

        const float polarity = shared::polarity_value(event);

        const std::size_t left_index = contribution.left_bin * plane_size + pixel;

        // No atomic required:
        // this thread uniquely owns this pixel.
        voxels[left_index] += polarity * contribution.left_weight;

        if (contribution.right_bin != contribution.left_bin && contribution.right_weight > 0.0F) {

            const std::size_t right_index = contribution.right_bin * plane_size + pixel;

            voxels[right_index] += polarity * contribution.right_weight;
        }
    }
}

} // namespace

void prepare_pixel_gather(const Event* device_events, const shared::VoxelizationParams& params,
                          PixelGatherWorkspace& workspace, cudaStream_t stream,
                          unsigned int block_size) {
    if (params.event_count > std::numeric_limits<std::uint32_t>::max()) {

        throw std::overflow_error("PixelGather currently supports at most "
                                  "UINT32_MAX events per window");
    }

    const std::size_t pixel_count = params.width * params.height;

    if (params.event_count == 0)
        return;
    shared::check_cuda(
        cudaMemsetAsync(workspace.counts.data(), 0, pixel_count * sizeof(std::uint32_t), stream),
        "cudaMemset(pixel_counts)");

    const std::size_t event_blocks =
        params.event_count / block_size + (params.event_count % block_size != 0);
    if (event_blocks > std::numeric_limits<unsigned int>::max())
        throw std::overflow_error("PixelGather event CUDA grid dimension overflow");

    count_events_per_pixel_kernel<<<static_cast<unsigned int>(event_blocks), block_size, 0,
                                    stream>>>(device_events, params.event_count,
                                              workspace.counts.data(), params.width);

    shared::check_cuda(cudaGetLastError(), "count_events_per_pixel_kernel launch");

    auto counts_begin = thrust::device_pointer_cast(workspace.counts.data());

    auto offsets_begin = thrust::device_pointer_cast(workspace.offsets.data());

    thrust::exclusive_scan(thrust::cuda::par.on(stream), counts_begin, counts_begin + pixel_count,
                           offsets_begin);

    set_last_offset_kernel<<<1, 1, 0, stream>>>(workspace.offsets.data(), pixel_count,
                                                static_cast<std::uint32_t>(params.event_count));
    shared::check_cuda(cudaGetLastError(), "set_last_offset_kernel launch");

    // Each pixel needs a mutable cursor starting
    // at its offset.
    shared::check_cuda(cudaMemcpyAsync(workspace.cursors.data(), workspace.offsets.data(),
                                       pixel_count * sizeof(std::uint32_t),
                                       cudaMemcpyDeviceToDevice, stream),
                       "cudaMemcpy(pixel offsets -> cursors)");

    group_events_by_pixel_kernel<<<static_cast<unsigned int>(event_blocks), block_size, 0,
                                   stream>>>(device_events, params.event_count,
                                             workspace.grouped.data(), workspace.cursors.data(),
                                             params.width);

    shared::check_cuda(cudaGetLastError(), "group_events_by_pixel_kernel launch");
}

void finish_pixel_gather(float* device_voxels, const shared::VoxelizationParams& params,
                         PixelGatherWorkspace& workspace, cudaStream_t stream,
                         unsigned int block_size) {
    if (params.event_count == 0)
        return;
    const std::size_t pixel_count = params.width * params.height;

    const std::size_t pixel_blocks = pixel_count / block_size + (pixel_count % block_size != 0);
    if (pixel_blocks > std::numeric_limits<unsigned int>::max())
        throw std::overflow_error("PixelGather pixel CUDA grid dimension overflow");

    pixel_gather_kernel<<<static_cast<unsigned int>(pixel_blocks), block_size, 0, stream>>>(
        workspace.grouped.data(), workspace.offsets.data(), device_voxels, params);

    shared::check_cuda(cudaGetLastError(), "pixel_gather_kernel launch");
}

} // namespace event_voxel::cuda::shared
