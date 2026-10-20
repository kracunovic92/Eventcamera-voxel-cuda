#include "shared/strategy_launchers.cuh"

#include "shared/cuda_check.cuh"
#include "shared/device_buffer.cuh"
#include "shared/voxel_math.cuh"

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace event_voxel::cuda::shared {

namespace {

using VoxelKey = std::uint64_t;

__global__ void generate_contributions_kernel(const Event* events, VoxelKey* keys, float* values,
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

    const std::size_t output_offset = event_index * 2;

    const std::size_t left_index = contribution.left_bin * plane_size + pixel_index;

    keys[output_offset] = static_cast<VoxelKey>(left_index);

    values[output_offset] = polarity * contribution.left_weight;

    if (contribution.right_bin != contribution.left_bin && contribution.right_weight > 0.0F) {

        const std::size_t right_index = contribution.right_bin * plane_size + pixel_index;

        keys[output_offset + 1] = static_cast<VoxelKey>(right_index);

        values[output_offset + 1] = polarity * contribution.right_weight;
    } else {
        // One-past-the-end sentinel.
        // All invalid second contributions sort
        // together at the end.
        keys[output_offset + 1] = static_cast<VoxelKey>(params.voxel_count);

        values[output_offset + 1] = 0.0F;
    }
}

__global__ void scatter_reduced_kernel(const VoxelKey* keys, const float* values,
                                       std::size_t reduced_count, float* voxels,
                                       std::size_t voxel_count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= reduced_count) {
        return;
    }

    const VoxelKey key = keys[index];

    // Ignore the sentinel key.
    if (key >= voxel_count) {
        return;
    }

    voxels[key] = values[index];
}

} // namespace

void prepare_sort_reduce(const Event* device_events, const shared::VoxelizationParams& params,
                         SortReduceWorkspace& workspace, cudaStream_t stream,
                         unsigned int block_size) {
    if (params.event_count > std::numeric_limits<std::size_t>::max() / 2) {

        throw std::overflow_error("Contribution count overflow");
    }

    const std::size_t contribution_count = params.event_count * 2;
    if (contribution_count == 0) {
        workspace.reduced_count = 0;
        return;
    }

    const std::size_t event_blocks =
        params.event_count / block_size + (params.event_count % block_size != 0);
    if (event_blocks > std::numeric_limits<unsigned int>::max())
        throw std::overflow_error("SortReduce event CUDA grid dimension overflow");

    generate_contributions_kernel<<<static_cast<unsigned int>(event_blocks), block_size, 0,
                                    stream>>>(device_events, workspace.keys.data(),
                                              workspace.values.data(), params);

    shared::check_cuda(cudaGetLastError(), "generate_contributions_kernel launch");

    auto keys_begin = thrust::device_pointer_cast(workspace.keys.data());

    auto values_begin = thrust::device_pointer_cast(workspace.values.data());

    thrust::sort_by_key(thrust::cuda::par.on(stream), keys_begin, keys_begin + contribution_count,
                        values_begin);

    auto reduced_keys_begin = thrust::device_pointer_cast(workspace.reduced_keys.data());

    auto reduced_values_begin = thrust::device_pointer_cast(workspace.reduced_values.data());

    const auto reduced_end = thrust::reduce_by_key(
        thrust::cuda::par.on(stream), keys_begin, keys_begin + contribution_count, values_begin,
        reduced_keys_begin, reduced_values_begin, thrust::equal_to<VoxelKey>{},
        thrust::plus<float>{});

    workspace.reduced_count = static_cast<std::size_t>(reduced_end.first - reduced_keys_begin);
}

void finish_sort_reduce(float* device_voxels, const shared::VoxelizationParams& params,
                        SortReduceWorkspace& workspace, cudaStream_t stream,
                        unsigned int block_size) {
    const auto reduced_count = workspace.reduced_count;

    const std::size_t reduced_blocks =
        reduced_count / block_size + (reduced_count % block_size != 0);
    if (reduced_blocks > std::numeric_limits<unsigned int>::max())
        throw std::overflow_error("SortReduce reduced CUDA grid dimension overflow");

    if (reduced_count != 0) {
        scatter_reduced_kernel<<<static_cast<unsigned int>(reduced_blocks), block_size, 0,
                                 stream>>>(workspace.reduced_keys.data(),
                                           workspace.reduced_values.data(), reduced_count,
                                           device_voxels, params.voxel_count);

        shared::check_cuda(cudaGetLastError(), "scatter_reduced_kernel launch");
    }
}

} // namespace event_voxel::cuda::shared
