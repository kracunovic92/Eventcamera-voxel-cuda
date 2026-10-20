#include "shared/cuda_check.cuh"
#include "shared/strategy_launchers.cuh"
#include "shared/voxel_math.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace event_voxel::cuda::shared {

namespace {

__global__ void event_scatter_kernel(const Event* __restrict__ events, float* __restrict__ voxels,
                                     shared::VoxelizationParams params) {
    const std::size_t event_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (event_index >= params.event_count) {
        return;
    }

    const Event event = events[event_index];

    const auto contribution =
        shared::interpolate_timestamp(event, params.start_us, params.bins - 1, params.time_scale);

    const float polarity = shared::polarity_value(event);

    const std::size_t plane_size = params.width * params.height;

    const std::size_t pixel_index =
        static_cast<std::size_t>(event.y) * params.width + static_cast<std::size_t>(event.x);

    const std::size_t left_index = contribution.left_bin * plane_size + pixel_index;

    atomicAdd(&voxels[left_index], polarity * contribution.left_weight);

    if (contribution.right_bin != contribution.left_bin && contribution.right_weight > 0.0F) {

        const std::size_t right_index = contribution.right_bin * plane_size + pixel_index;

        atomicAdd(&voxels[right_index], polarity * contribution.right_weight);
    }
}

} // namespace

void run_event_scatter(const Event* device_events, float* device_voxels,
                       const shared::VoxelizationParams& params, cudaStream_t stream,
                       unsigned int block_size) {
    if (params.event_count == 0)
        return;
    const std::size_t block_count =
        params.event_count / block_size + (params.event_count % block_size != 0);
    if (block_count > std::numeric_limits<unsigned int>::max())
        throw std::overflow_error("EventScatter CUDA grid dimension overflow");

    event_scatter_kernel<<<static_cast<unsigned int>(block_count), block_size, 0, stream>>>(
        device_events, device_voxels, params);

    shared::check_cuda(cudaGetLastError(), "event_scatter_kernel launch");
}

} // namespace event_voxel::cuda::shared
