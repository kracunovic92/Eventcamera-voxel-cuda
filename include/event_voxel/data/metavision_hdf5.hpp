#pragma once

#include "event_voxel/data/time_slice_source.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>

namespace event_voxel::data {

// Metavision SDK Stream is required by every build.
std::unique_ptr<TimeSliceSource> open_metavision_hdf5(const std::filesystem::path& path,
                                                      std::int64_t window_us);

} // namespace event_voxel::data
