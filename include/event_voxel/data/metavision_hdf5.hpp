#pragma once

#include "event_voxel/data/time_slice_source.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

namespace event_voxel::data {

// Metavision SDK Stream is required by every build.
std::unique_ptr<TimeSliceSource> open_metavision_hdf5(const std::filesystem::path& path,
                                                      std::int64_t window_us,
                                                      std::size_t max_window_events);

} // namespace event_voxel::data
