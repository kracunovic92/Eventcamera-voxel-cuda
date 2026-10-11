
#include "event_voxel/data/binary_event_loader.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace event_voxel {
namespace {

// Binary format constants.
constexpr std::size_t kHeaderSize = 40;
constexpr std::size_t kEventSize = 16;

constexpr std::array<std::uint8_t, 8> kMagic{'E', 'V', 'X', 'E', 'V', 'T', '0', '1'};

// Read an exact number of bytes.
// This function operates on raw byte buffers, never C++ Event objects.
template <std::size_t N>
void read_exact(std::ifstream& input, std::array<std::uint8_t, N>& buffer) {
    input.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));

    if (!input) {
        throw std::runtime_error("Unexpected end of binary event file");
    }
}

// Explicit little-endian decoding.
[[nodiscard]]
std::uint16_t read_u16(const std::uint8_t* bytes) noexcept {
    return static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[0]) |
                                      (static_cast<std::uint16_t>(bytes[1]) << 8));
}

[[nodiscard]]
std::uint32_t read_u32(const std::uint8_t* bytes) noexcept {
    std::uint32_t result = 0;

    for (std::size_t i = 0; i < 4; ++i) {
        result |= static_cast<std::uint32_t>(bytes[i]) << (8 * i);
    }

    return result;
}

[[nodiscard]]
std::uint64_t read_u64(const std::uint8_t* bytes) noexcept {
    std::uint64_t result = 0;

    for (std::size_t i = 0; i < 8; ++i) {
        result |= static_cast<std::uint64_t>(bytes[i]) << (8 * i);
    }

    return result;
}

[[nodiscard]]
std::int64_t read_i64(const std::uint8_t* bytes) noexcept {
    // Preserve all 64 bits, including the sign bit.
    return std::bit_cast<std::int64_t>(read_u64(bytes));
}

// The binary header is an implementation detail.
// It does not need to appear in the public loader interface.
struct BinaryHeader {
    std::uint32_t width;
    std::uint32_t height;
    std::uint64_t event_count;
    std::int64_t start_us;
    std::int64_t end_us;
};

[[nodiscard]]
BinaryHeader read_header(std::ifstream& input) {
    std::array<std::uint8_t, kHeaderSize> bytes{};
    read_exact(input, bytes);

    for (std::size_t i = 0; i < kMagic.size(); ++i) {
        if (bytes[i] != kMagic[i]) {
            throw std::runtime_error("Invalid binary event format or version");
        }
    }

    const BinaryHeader header{.width = read_u32(bytes.data() + 8),
                              .height = read_u32(bytes.data() + 12),
                              .event_count = read_u64(bytes.data() + 16),
                              .start_us = read_i64(bytes.data() + 24),
                              .end_us = read_i64(bytes.data() + 32)};

    if (header.width == 0 || header.height == 0) {
        throw std::runtime_error("Invalid sensor dimensions in binary header");
    }

    if (header.start_us < 0 || header.end_us <= header.start_us) {
        throw std::runtime_error("Invalid temporal window in binary header");
    }

    if (header.event_count == 0) {
        throw std::runtime_error("Binary recording contains no events");
    }

    return header;
}

// Decode and validate exactly one event record.
[[nodiscard]]
Event read_event(const std::array<std::uint8_t, kEventSize>& bytes, const BinaryHeader& header,
                 std::uint64_t index) {
    const auto x = read_u16(bytes.data());
    const auto y = read_u16(bytes.data() + 2);
    const auto polarity = read_u16(bytes.data() + 4);
    const auto reserved = read_u16(bytes.data() + 6);
    const auto timestamp = read_i64(bytes.data() + 8);

    const auto fail = [index](const char* message) {
        throw std::runtime_error("Invalid event at index " + std::to_string(index) + ": " +
                                 message);
    };

    if (reserved != 0) {
        fail("reserved bytes must be zero");
    }

    if (x >= header.width || y >= header.height) {
        fail("coordinates outside sensor");
    }

    if (polarity > 1) {
        fail("polarity must be 0 or 1");
    }

    if (timestamp < header.start_us || timestamp >= header.end_us) {
        fail("timestamp outside recording window");
    }

    // Event's in-memory field order differs from
    // the binary record's field order.
    return Event{
        .x = x, .y = y, .timestamp_us = timestamp, .polarity = static_cast<std::uint8_t>(polarity)};
}

} // namespace

BinaryEventLoader::BinaryEventLoader(std::filesystem::path path) : path_(std::move(path)) {
    if (path_.empty()) {
        throw std::invalid_argument("Binary recording path cannot be empty");
    }
}

EventBatch BinaryEventLoader::load() {
    std::ifstream input(path_, std::ios::binary);

    if (!input) {
        throw std::runtime_error("Cannot open binary recording: " + path_.string());
    }

    // 1. Read and validate the file header.
    const BinaryHeader header = read_header(input);

    // 2. Verify the exact expected file size.
    // Division avoids overflow from event_count * 16.
    const auto file_size = std::filesystem::file_size(path_);

    if (file_size < kHeaderSize) {
        throw std::runtime_error("Binary recording is smaller than its header");
    }

    const auto event_bytes = file_size - kHeaderSize;

    if (event_bytes % kEventSize != 0 || event_bytes / kEventSize != header.event_count) {
        throw std::runtime_error("Binary recording size does not match event count");
    }

    // 3. Initialize our existing application data structure.
    EventBatch batch{.events = {},
                     .width = header.width,
                     .height = header.height,
                     .start_us = header.start_us,
                     .end_us = header.end_us};

    if (header.event_count > batch.events.max_size()) {
        throw std::runtime_error("Recording contains too many events");
    }

    batch.events.reserve(static_cast<std::size_t>(header.event_count));

    // 4. Deserialize events and check chronological order.
    std::array<std::uint8_t, kEventSize> buffer{};

    std::int64_t previous_timestamp = header.start_us;

    for (std::uint64_t i = 0; i < header.event_count; ++i) {
        read_exact(input, buffer);

        Event event = read_event(buffer, header, i);

        if (event.timestamp_us < previous_timestamp) {
            throw std::runtime_error("Events are not ordered at index " + std::to_string(i));
        }

        previous_timestamp = event.timestamp_us;

        batch.events.push_back(event);
    }

    return batch;
}

} // namespace event_voxel
