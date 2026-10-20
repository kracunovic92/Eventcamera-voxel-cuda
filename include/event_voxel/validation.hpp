#pragma once

#include "event_voxel/event_batch.hpp"
#include "event_voxel/event_window.hpp"
#include "event_voxel/voxelization_spec.hpp"

namespace event_voxel {

void validate_configuration(const EventBatch& batch, const VoxelizationSpec& spec);

void validate_event(const Event& event, const EventBatch& batch);

void validate_batch(const EventBatch& batch, const VoxelizationSpec& spec);
void validate_window(const EventWindowView& window, const VoxelizationSpec& spec);

} // namespace event_voxel
