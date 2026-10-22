#pragma once

#include "event_voxel/event_window.hpp"
#include "event_voxel/voxelization_spec.hpp"

namespace event_voxel {

void validate_window(const EventWindowView& window, const VoxelizationSpec& spec);

} // namespace event_voxel
