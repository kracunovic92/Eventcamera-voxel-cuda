#include "event_voxel/cuda/execution_session.hpp"
#include "event_voxel/cuda/voxelizer.hpp"
#include "event_voxel/event_window.hpp"
#include "event_voxel/validation.hpp"

#include <cuda_runtime.h>

namespace event_voxel::cuda {

bool is_available() noexcept {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

VoxelGrid CudaVoxelizer::voxelize(const EventBatch& batch, const VoxelizationSpec& spec) const {
    const auto window = view_of(batch);
    validate_window(window, spec);
    VoxelGrid output({window.width, window.height, spec.bins});
    if (window.events.empty())
        return output;
    CudaExecutionSession session(strategy_);
    session.reserve(window.events.size(), window.width, window.height, spec.bins);
    session.set_window(window, spec);
    session.upload_events();
    session.run_strategy();
    session.download_grid(output.data());
    return output;
}

} // namespace event_voxel::cuda
