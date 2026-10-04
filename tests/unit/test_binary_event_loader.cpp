
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "event_voxel/data/binary_event_loader.hpp"

namespace {

using namespace event_voxel;

namespace fs = std::filesystem;

constexpr std::size_t kHeaderSize = 40;
constexpr std::size_t kEventSize = 16;

// Our test data is deliberately independent of the
// production binary deserializer.
using Bytes = std::vector<std::uint8_t>;

struct TestEvent {
    std::uint16_t x;
    std::uint16_t y;
    std::uint16_t polarity;
    std::int64_t timestamp;
    std::uint16_t reserved = 0;
};

// Append an unsigned integer using little-endian encoding.
void append_le(Bytes& bytes, std::uint64_t value, std::size_t byte_count) {
    for (std::size_t i = 0; i < byte_count; ++i) {
        bytes.push_back(static_cast<std::uint8_t>((value >> (8 * i)) & 0xFF));
    }
}

void append_event(Bytes& bytes, const TestEvent& event) {
    append_le(bytes, event.x, 2);
    append_le(bytes, event.y, 2);
    append_le(bytes, event.polarity, 2);
    append_le(bytes, event.reserved, 2);

    // All our test timestamps are nonnegative.
    append_le(bytes, static_cast<std::uint64_t>(event.timestamp), 8);
}

Bytes make_recording(const std::vector<TestEvent>& events, std::uint32_t width = 640,
                     std::uint32_t height = 480, std::int64_t start_us = 0,
                     std::int64_t end_us = 1000) {
    Bytes bytes;

    // Versioned file identifier.
    for (const char character : std::string("EVXEVT01")) {
        bytes.push_back(static_cast<std::uint8_t>(character));
    }

    append_le(bytes, width, 4);
    append_le(bytes, height, 4);
    append_le(bytes, events.size(), 8);

    append_le(bytes, static_cast<std::uint64_t>(start_us), 8);

    append_le(bytes, static_cast<std::uint64_t>(end_us), 8);

    for (const TestEvent& event : events) {
        append_event(bytes, event);
    }

    return bytes;
}

// A temporary directory is created for each fixture.
//
// It prevents tests from modifying repository files
// and supports running tests in parallel.
class TemporaryRecording {
  public:
    explicit TemporaryRecording(const Bytes& bytes) {
        static std::atomic<std::uint64_t> counter{0};

        bool created = false;

        for (int attempt = 0; attempt < 20; ++attempt) {
            const auto id = counter.fetch_add(1);

            const auto now = std::chrono::steady_clock::now().time_since_epoch().count();

            const auto random = std::random_device{}();

            directory_ =
                fs::temp_directory_path() / ("event_voxel_test_" + std::to_string(now) + "_" +
                                             std::to_string(random) + "_" + std::to_string(id));

            created = fs::create_directory(directory_);

            if (created) {
                break;
            }
        }

        if (!created) {
            throw std::runtime_error("Cannot create temporary test directory");
        }

        path_ = directory_ / "recording.bin";

        std::ofstream output(path_, std::ios::binary | std::ios::trunc);

        if (!output) {
            throw std::runtime_error("Cannot create temporary recording");
        }

        output.write(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));

        if (!output) {
            throw std::runtime_error("Cannot write temporary recording");
        }
    }

    TemporaryRecording(const TemporaryRecording&) = delete;
    TemporaryRecording& operator=(const TemporaryRecording&) = delete;

    ~TemporaryRecording() {
        std::error_code error;
        fs::remove_all(directory_, error);
    }

    [[nodiscard]]
    const fs::path& path() const noexcept {
        return path_;
    }

  private:
    fs::path directory_;
    fs::path path_;
};

} // namespace

TEST_CASE("Binary loader reads valid events", "[binary][unit]") {
    const auto bytes = make_recording(
        {TestEvent{35, 443, 1, 0}, TestEvent{36, 443, 1, 0}, TestEvent{74, 443, 0, 500}});

    TemporaryRecording recording(bytes);

    BinaryEventLoader loader(recording.path());
    const EventBatch batch = loader.load();

    REQUIRE(batch.width == 640);
    REQUIRE(batch.height == 480);
    REQUIRE(batch.start_us == 0);
    REQUIRE(batch.end_us == 1000);
    REQUIRE(batch.events.size() == 3);

    CHECK(batch.events[0].x == 35);
    CHECK(batch.events[0].y == 443);
    CHECK(batch.events[0].polarity == 1);
    CHECK(batch.events[0].timestamp_us == 0);

    CHECK(batch.events[1].x == 36);
    CHECK(batch.events[1].timestamp_us == 0);

    CHECK(batch.events[2].x == 74);
    CHECK(batch.events[2].polarity == 0);
    CHECK(batch.events[2].timestamp_us == 500);
}

TEST_CASE("Binary loader implements the shared loader interface", "[binary][unit]") {
    TemporaryRecording recording(make_recording({TestEvent{1, 2, 1, 100}}));

    BinaryEventLoader implementation(recording.path());

    IEventLoader& loader = implementation;

    const EventBatch batch = loader.load();

    REQUIRE(batch.events.size() == 1);
    CHECK(batch.events.front().x == 1);
    CHECK(batch.events.front().y == 2);
}

TEST_CASE("Binary loader rejects an invalid format", "[binary][unit]") {
    auto bytes = make_recording({TestEvent{1, 2, 1, 100}});

    // Corrupt the first magic byte.
    bytes[0] = 'X';

    TemporaryRecording recording(bytes);
    BinaryEventLoader loader(recording.path());

    REQUIRE_THROWS_AS(loader.load(), std::runtime_error);
}

TEST_CASE("Binary loader rejects a truncated header", "[binary][unit]") {
    auto bytes = make_recording({TestEvent{1, 2, 1, 100}});

    bytes.resize(kHeaderSize - 1);

    TemporaryRecording recording(bytes);
    BinaryEventLoader loader(recording.path());

    REQUIRE_THROWS_AS(loader.load(), std::runtime_error);
}

TEST_CASE("Binary loader rejects incorrect file length", "[binary][unit]") {
    SECTION("Truncated event") {
        auto bytes = make_recording({TestEvent{1, 2, 1, 100}});

        bytes.pop_back();

        TemporaryRecording recording(bytes);
        BinaryEventLoader loader(recording.path());

        REQUIRE_THROWS_AS(loader.load(), std::runtime_error);
    }

    SECTION("Unexpected trailing byte") {
        auto bytes = make_recording({TestEvent{1, 2, 1, 100}});

        bytes.push_back(0xFF);

        TemporaryRecording recording(bytes);
        BinaryEventLoader loader(recording.path());

        REQUIRE_THROWS_AS(loader.load(), std::runtime_error);
    }
}

TEST_CASE("Binary loader rejects invalid recording metadata", "[binary][unit]") {
    SECTION("Zero sensor width") {
        TemporaryRecording recording(make_recording({TestEvent{1, 2, 1, 100}}, 0, 480));

        BinaryEventLoader loader(recording.path());

        REQUIRE_THROWS_AS(loader.load(), std::runtime_error);
    }

    SECTION("Invalid temporal window") {
        TemporaryRecording recording(
            make_recording({TestEvent{1, 2, 1, 100}}, 640, 480, 1000, 1000));

        BinaryEventLoader loader(recording.path());

        REQUIRE_THROWS_AS(loader.load(), std::runtime_error);
    }
}

TEST_CASE("Binary loader rejects corrupted event records", "[binary][unit]") {
    SECTION("Invalid polarity") {
        TemporaryRecording recording(make_recording({TestEvent{1, 2, 2, 100}}));

        BinaryEventLoader loader(recording.path());

        REQUIRE_THROWS_AS(loader.load(), std::runtime_error);
    }

    SECTION("Coordinates outside sensor") {
        TemporaryRecording recording(make_recording({TestEvent{640, 2, 1, 100}}));

        BinaryEventLoader loader(recording.path());

        REQUIRE_THROWS_AS(loader.load(), std::runtime_error);
    }

    SECTION("Timestamp at exclusive end") {
        TemporaryRecording recording(make_recording({TestEvent{1, 2, 1, 1000}}));

        BinaryEventLoader loader(recording.path());

        REQUIRE_THROWS_AS(loader.load(), std::runtime_error);
    }

    SECTION("Reserved bytes are nonzero") {
        TemporaryRecording recording(make_recording({TestEvent{1, 2, 1, 100, 1}}));

        BinaryEventLoader loader(recording.path());

        REQUIRE_THROWS_AS(loader.load(), std::runtime_error);
    }
}

TEST_CASE("Binary loader rejects decreasing timestamps", "[binary][unit]") {
    TemporaryRecording recording(
        make_recording({TestEvent{1, 2, 1, 500}, TestEvent{2, 3, 0, 100}}));

    BinaryEventLoader loader(recording.path());

    REQUIRE_THROWS_AS(loader.load(), std::runtime_error);
}

TEST_CASE("Binary loader rejects empty input path", "[binary][unit]") {
    REQUIRE_THROWS_AS(BinaryEventLoader{fs::path{}}, std::invalid_argument);
}

TEST_CASE("Binary loader rejects a nonexistent recording", "[binary][unit]") {
    const auto missing = fs::temp_directory_path() / "event_voxel_nonexistent_recording.bin";

    // Ensure the path cannot accidentally reference
    // a preexisting file.
    if (fs::exists(missing)) {
        SKIP("Unexpected test file already exists");
    }

    BinaryEventLoader loader(missing);

    REQUIRE_THROWS_AS(loader.load(), std::runtime_error);
}
