#include "event_voxel/cuda/voxelizer.hpp"

#include "event_voxel/event.hpp"
#include "event_voxel/validation.hpp"
#include "event_voxel/voxel_grid_spec.hpp"

#include "shared/cuda_check.cuh"
#include "shared/device_buffer.cuh"
#include "shared/strategy_launchers.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <type_traits>

namespace event_voxel::cuda {

bool is_available() noexcept {
    int device_count = 0;

    const cudaError_t status = cudaGetDeviceCount(&device_count);

    return status == cudaSuccess && device_count > 0;
}

VoxelGrid CudaVoxelizer::voxelize(const EventBatch& batch, const VoxelizationSpec& spec) const {
    event_voxel::validate_batch(batch, spec);

    const VoxelGridSpec grid_spec{
        .width = batch.width,
        .height = batch.height,
        .bins = spec.bins,
    };

    VoxelGrid grid{grid_spec};

    if (batch.events.empty()) {
        return grid;
    }

    static_assert(std::is_trivially_copyable_v<Event>, "Event must be trivially copyable "
                                                       "for direct CUDA transfer");

    shared::DeviceBuffer<Event> device_events{batch.events.size(), "cudaMalloc(events)"};

    shared::DeviceBuffer<float> device_voxels{grid_spec.element_count(), "cudaMalloc(voxels)"};

    shared::check_cuda(cudaMemcpy(device_events.data(), batch.events.data(), device_events.bytes(),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(events HostToDevice)");

    shared::check_cuda(cudaMemset(device_voxels.data(), 0, device_voxels.bytes()),
                       "cudaMemset(voxels)");

    const std::int64_t duration_us = batch.end_us - batch.start_us;

    const double time_scale = static_cast<double>(spec.bins - 1) / static_cast<double>(duration_us);

    const shared::VoxelizationParams params{
        .event_count = batch.events.size(),

        .width = batch.width,

        .height = batch.height,

        .bins = spec.bins,

        .voxel_count = grid_spec.element_count(),

        .start_us = batch.start_us,

        .time_scale = time_scale,
    };

    switch (strategy_) {
    case Strategy::EventScatter:
        run_event_scatter(device_events.data(), device_voxels.data(), params);
        break;

    case Strategy::PixelGather:
        run_pixel_gather(device_events.data(), device_voxels.data(), params);
        break;

    case Strategy::SortReduce:
        run_sort_reduce(device_events.data(), device_voxels.data(), params);
        break;
    }

    shared::check_cuda(cudaDeviceSynchronize(), "CUDA voxelization execution");

    shared::check_cuda(cudaMemcpy(grid.data().data(), device_voxels.data(), device_voxels.bytes(),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy(voxels DeviceToHost)");

    return grid;
}

} // namespace event_voxel::cuda
