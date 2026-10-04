#pragma once

#include <filesystem>

#include "event_voxel/voxel_grid.hpp"

namespace event_voxel::io {

/// Write a voxel grid to a NumPy .npy file.
///
/// Output specification:
/// - Format: NumPy .npy, version 1.0
/// - Shape: [bins, height, width]
/// - Storage: C-contiguous (row-major)
///
/// The function does not modify or normalize voxel values.
///
/// Throws std::runtime_error if the output cannot be written
/// or the NumPy format's limitations are exceeded.

void write_npy(const VoxelGrid& grid, const std::filesystem::path& output_path);

} // namespace event_voxel::io
