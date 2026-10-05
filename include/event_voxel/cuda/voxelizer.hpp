#pragma once

#include "event_voxel/cuda/strategy.hpp"
#include "event_voxel/event_batch.hpp"
#include "event_voxel/voxel_grid.hpp"
#include "event_voxel/voxelization_spec.hpp"

namespace event_voxel::cuda {

[[nodiscard]]
bool is_available() noexcept;

class CudaVoxelizer {
  public:
    explicit CudaVoxelizer(Strategy strategy = Strategy::EventScatter) noexcept
        : strategy_(strategy) {}

    [[nodiscard]]
    VoxelGrid voxelize(const EventBatch& batch, const VoxelizationSpec& spec) const;

    [[nodiscard]]
    Strategy strategy() const noexcept {
        return strategy_;
    }

  private:
    Strategy strategy_;
};

} // namespace event_voxel::cuda
