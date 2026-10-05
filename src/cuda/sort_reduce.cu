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

constexpr unsigned int THREADS_PER_BLOCK = 256;

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

void run_sort_reduce(const Event* device_events, float* device_voxels,
                     const shared::VoxelizationParams& params) {
    if (params.event_count > std::numeric_limits<std::size_t>::max() / 2) {

        throw std::overflow_error("Contribution count overflow");
    }

    const std::size_t contribution_count = params.event_count * 2;

    shared::DeviceBuffer<VoxelKey> keys{contribution_count, "cudaMalloc(contribution keys)"};

    shared::DeviceBuffer<float> values{contribution_count, "cudaMalloc(contribution values)"};

    const std::size_t event_blocks =
        (params.event_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    generate_contributions_kernel<<<static_cast<unsigned int>(event_blocks), THREADS_PER_BLOCK>>>(
        device_events, keys.data(), values.data(), params);

    shared::check_cuda(cudaGetLastError(), "generate_contributions_kernel launch");

    auto keys_begin = thrust::device_pointer_cast(keys.data());

    auto values_begin = thrust::device_pointer_cast(values.data());

    thrust::sort_by_key(thrust::device, keys_begin, keys_begin + contribution_count, values_begin);

    shared::DeviceBuffer<VoxelKey> reduced_keys{contribution_count, "cudaMalloc(reduced keys)"};

    shared::DeviceBuffer<float> reduced_values{contribution_count, "cudaMalloc(reduced values)"};

    auto reduced_keys_begin = thrust::device_pointer_cast(reduced_keys.data());

    auto reduced_values_begin = thrust::device_pointer_cast(reduced_values.data());

    const auto reduced_end =
        thrust::reduce_by_key(thrust::device, keys_begin, keys_begin + contribution_count,
                              values_begin, reduced_keys_begin, reduced_values_begin,
                              thrust::equal_to<VoxelKey>{}, thrust::plus<float>{});

    const std::size_t reduced_count =
        static_cast<std::size_t>(reduced_end.first - reduced_keys_begin);

    const std::size_t reduced_blocks = (reduced_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    if (reduced_count != 0) {
        scatter_reduced_kernel<<<static_cast<unsigned int>(reduced_blocks), THREADS_PER_BLOCK>>>(
            reduced_keys.data(), reduced_values.data(), reduced_count, device_voxels,
            params.voxel_count);

        shared::check_cuda(cudaGetLastError(), "scatter_reduced_kernel launch");
    }
}

} // namespace event_voxel::cuda::shared
