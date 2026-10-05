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

constexpr unsigned int THREADS_PER_BLOCK = 256;

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

void run_pixel_gather(const Event* device_events, float* device_voxels,
                      const shared::VoxelizationParams& params) {
    if (params.event_count > std::numeric_limits<std::uint32_t>::max()) {

        throw std::overflow_error("PixelGather currently supports at most "
                                  "UINT32_MAX events per window");
    }

    const std::size_t pixel_count = params.width * params.height;

    shared::DeviceBuffer<std::uint32_t> pixel_counts{pixel_count, "cudaMalloc(pixel_counts)"};

    shared::DeviceBuffer<std::uint32_t> pixel_offsets{pixel_count + 1, "cudaMalloc(pixel_offsets)"};

    shared::DeviceBuffer<std::uint32_t> write_cursors{pixel_count, "cudaMalloc(write_cursors)"};

    shared::DeviceBuffer<Event> grouped_events{params.event_count, "cudaMalloc(grouped_events)"};

    shared::check_cuda(cudaMemset(pixel_counts.data(), 0, pixel_counts.bytes()),
                       "cudaMemset(pixel_counts)");

    const std::size_t event_blocks =
        (params.event_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    count_events_per_pixel_kernel<<<static_cast<unsigned int>(event_blocks), THREADS_PER_BLOCK>>>(
        device_events, params.event_count, pixel_counts.data(), params.width);

    shared::check_cuda(cudaGetLastError(), "count_events_per_pixel_kernel launch");

    auto counts_begin = thrust::device_pointer_cast(pixel_counts.data());

    auto offsets_begin = thrust::device_pointer_cast(pixel_offsets.data());

    thrust::exclusive_scan(thrust::device, counts_begin, counts_begin + pixel_count, offsets_begin);

    const std::uint32_t final_offset = static_cast<std::uint32_t>(params.event_count);

    shared::check_cuda(cudaMemcpy(pixel_offsets.data() + pixel_count, &final_offset,
                                  sizeof(final_offset), cudaMemcpyHostToDevice),
                       "cudaMemcpy(final pixel offset)");

    // Each pixel needs a mutable cursor starting
    // at its offset.
    shared::check_cuda(cudaMemcpy(write_cursors.data(), pixel_offsets.data(),
                                  pixel_count * sizeof(std::uint32_t), cudaMemcpyDeviceToDevice),
                       "cudaMemcpy(pixel offsets -> cursors)");

    group_events_by_pixel_kernel<<<static_cast<unsigned int>(event_blocks), THREADS_PER_BLOCK>>>(
        device_events, params.event_count, grouped_events.data(), write_cursors.data(),
        params.width);

    shared::check_cuda(cudaGetLastError(), "group_events_by_pixel_kernel launch");

    const std::size_t pixel_blocks = (pixel_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    pixel_gather_kernel<<<static_cast<unsigned int>(pixel_blocks), THREADS_PER_BLOCK>>>(
        grouped_events.data(), pixel_offsets.data(), device_voxels, params);

    shared::check_cuda(cudaGetLastError(), "pixel_gather_kernel launch");
}

} // namespace event_voxel::cuda::shared
