
#include "event_voxel/io/npy_writer.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace event_voxel::io {

namespace {

constexpr std::size_t kNpyPreambleSize = 10;
constexpr std::size_t kNpyAlignment = 16;
constexpr std::size_t kFloatBytes = 4;

// Our file format explicitly requires IEEE 754 float32.
static_assert(sizeof(float) == kFloatBytes);
static_assert(std::numeric_limits<float>::is_iec559);

// Build the NumPy v1.0 header.
//
// A .npy file contains:
//   6 bytes: magic identifier
//   2 bytes: version (1, 0)
//   2 bytes: little-endian header length
//   N bytes: padded header dictionary
//   remaining bytes: raw tensor data
//
// The complete header, including the 10-byte preamble,
// must be aligned to a 16-byte boundary.
[[nodiscard]]
std::string make_npy_header(const VoxelGridSpec& spec) {
    std::string header = "{'descr': '<f4', "
                         "'fortran_order': False, "
                         "'shape': (" +
                         std::to_string(spec.bins) + ", " + std::to_string(spec.height) + ", " +
                         std::to_string(spec.width) + "), }";

    // Reserve one byte for the mandatory final newline.
    const std::size_t unpadded_size = kNpyPreambleSize + header.size() + 1;

    const std::size_t padding = (kNpyAlignment - (unpadded_size % kNpyAlignment)) % kNpyAlignment;

    header.append(padding, ' ');
    header.push_back('\n');

    // NumPy v1.0 stores the header length as uint16.
    if (header.size() > std::numeric_limits<std::uint16_t>::max()) {
        throw std::runtime_error("Voxel dimensions exceed the NumPy v1.0 "
                                 "header size limit");
    }

    return header;
}

// Write the NumPy magic bytes, format version,
// header length and padded dictionary.
void write_header(std::ofstream& output, const std::string& header) {
    constexpr char magic[] = {static_cast<char>(0x93), 'N', 'U', 'M', 'P', 'Y'};

    output.write(magic, sizeof(magic));

    // NumPy format version 1.0.
    output.put(static_cast<char>(1));
    output.put(static_cast<char>(0));

    // Header length: unsigned 16-bit little-endian.
    const auto length = static_cast<std::uint16_t>(header.size());

    output.put(static_cast<char>(length & 0xFF));
    output.put(static_cast<char>((length >> 8) & 0xFF));

    output.write(header.data(), static_cast<std::streamsize>(header.size()));

    if (!output) {
        throw std::runtime_error("Failed to write NumPy file header");
    }
}

// Encode one float explicitly in little-endian order.
//
// std::bit_cast preserves its exact IEEE 754 bit pattern.
// Consequently, we don't normalize, round or otherwise
// transform the numerical voxel values.
void encode_float32(float value, std::uint8_t* destination) noexcept {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);

    destination[0] = static_cast<std::uint8_t>(bits & 0xFF);

    destination[1] = static_cast<std::uint8_t>((bits >> 8) & 0xFF);

    destination[2] = static_cast<std::uint8_t>((bits >> 16) & 0xFF);

    destination[3] = static_cast<std::uint8_t>((bits >> 24) & 0xFF);
}

} // namespace

void write_npy(const VoxelGrid& grid, const std::filesystem::path& output_path) {
    if (output_path.empty()) {
        throw std::runtime_error("NumPy output path cannot be empty");
    }

    const VoxelGridSpec& spec = grid.spec();

    if (spec.width == 0 || spec.height == 0 || spec.bins == 0) {
        throw std::runtime_error("Cannot export an empty voxel grid");
    }

    // Prepare the NumPy metadata before creating the file.
    const std::string header = make_npy_header(spec);

    // Create the output directory if necessary.
    const auto parent = output_path.parent_path();

    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);

    if (!output) {
        throw std::runtime_error("Cannot open NumPy output file: " + output_path.string());
    }

    write_header(output, header);

    // Export one spatial row at a time.
    //
    // This avoids making a second full-size copy
    // of the voxel grid in memory.
    if (spec.width > std::numeric_limits<std::size_t>::max() / kFloatBytes) {
        throw std::runtime_error("Voxel row exceeds addressable memory");
    }

    const std::size_t row_bytes = spec.width * kFloatBytes;

    if (row_bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("Voxel row is too large to write");
    }

    std::vector<std::uint8_t> row(row_bytes);

    // C-contiguous tensor ordering: [bin][y][x].
    for (std::size_t bin = 0; bin < spec.bins; ++bin) {
        for (std::size_t y = 0; y < spec.height; ++y) {
            for (std::size_t x = 0; x < spec.width; ++x) {
                encode_float32(grid.at(bin, y, x), row.data() + x * kFloatBytes);
            }

            output.write(reinterpret_cast<const char*>(row.data()),
                         static_cast<std::streamsize>(row.size()));

            if (!output) {
                throw std::runtime_error("Failed while writing NumPy voxel data");
            }
        }
    }

    output.close();

    if (!output) {
        throw std::runtime_error("Failed to finalize NumPy output file");
    }
}

} // namespace event_voxel::io
