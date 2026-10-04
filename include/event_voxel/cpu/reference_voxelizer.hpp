#pragma once

#include "event_voxel/event_batch.hpp"
#include "event_voxel/voxel_grid.hpp"
#include "event_voxel/voxelization_spec.hpp"

namespace event_voxel {

class CpuReferenceVoxelizer final {
  public:
    [[nodiscard]]
    VoxelGrid voxelize(const EventBatch& batch, const VoxelizationSpec& spec);
};

} // namespace event_voxel
