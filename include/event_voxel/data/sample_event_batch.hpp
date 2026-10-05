#pragma once

#include "event_voxel/event_batch.hpp"

#include <cstddef>

namespace event_voxel {

// Select ordered events across the entire recorded window.
// The returned batch retains the original temporal window.
[[nodiscard]] EventBatch sample_evenly(EventBatch source, std::size_t event_count);

} // namespace event_voxel
