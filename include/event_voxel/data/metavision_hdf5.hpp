#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>

#include "event_voxel/data/event_window_source.hpp"

namespace event_voxel::data {

// Available only when EVENT_VOXEL_ENABLE_METAVISION is enabled at configure time.
std::unique_ptr<EventReader> open_metavision_hdf5(const std::filesystem::path& path,
                                                  std::size_t max_decoded_block_events);

} // namespace event_voxel::data
