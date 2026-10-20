#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "event_voxel/cpu/reference_voxelizer.hpp"
#include "event_voxel/data/event_window_source.hpp"

namespace {
class VectorReader final : public event_voxel::data::EventReader {
  public:
    explicit VectorReader(std::vector<event_voxel::Event> events) : events_(std::move(events)) {}
    bool next(event_voxel::Event& event) override {
        if (position_ == events_.size())
            return false;
        event = events_[position_++];
        return true;
    }
    std::uint32_t width() const override {
        return 2;
    }
    std::uint32_t height() const override {
        return 2;
    }

  private:
    std::vector<event_voxel::Event> events_;
    std::size_t position_ = 0;
};
} // namespace

TEST_CASE("window source preserves boundaries, gaps and final interval") {
    using namespace event_voxel;
    data::WindowConfig config{
        .window_us = 100, .chunk_us = 65, .max_chunk_events = 2, .max_window_events = 4};
    data::EventWindowSource source(
        std::make_unique<VectorReader>(
            std::vector<Event>{{0, 0, 10, 1}, {0, 0, 99, 0}, {1, 1, 110, 1}, {1, 1, 310, 1}}),
        config);
    EventWindowView window;
    REQUIRE(source.next(window));
    REQUIRE(window.start_us == 10);
    REQUIRE(window.end_us == 110);
    REQUIRE(window.events.size() == 2);
    REQUIRE(window.events[1].timestamp_us == 99);
    REQUIRE(source.next(window));
    REQUIRE(window.start_us == 110);
    REQUIRE(window.events.size() == 1);
    REQUIRE(window.events[0].timestamp_us == 110);
    REQUIRE(source.next(window));
    REQUIRE(window.events.empty());
    REQUIRE(source.next(window));
    REQUIRE(window.start_us == 310);
    REQUIRE(window.end_us == 311);
    REQUIRE(window.partial_final);
    REQUIRE(window.events.size() == 1);
    REQUIRE_FALSE(source.next(window));
    REQUIRE(source.peak_buffered_events() <= 6);
}

TEST_CASE("dense window fails without dropping events") {
    using namespace event_voxel;
    data::EventWindowSource source(
        std::make_unique<VectorReader>(
            std::vector<Event>{{0, 0, 1, 1}, {0, 0, 2, 1}, {0, 0, 3, 1}}),
        {.window_us = 100, .chunk_us = 1000, .max_chunk_events = 2, .max_window_events = 2});
    EventWindowView window;
    REQUIRE_THROWS(source.next(window));
}

TEST_CASE("CPU view and owning batch agree") {
    using namespace event_voxel;
    EventBatch batch{{{0, 0, 10, 1}, {0, 0, 30, 0}}, 2, 2, 10, 110};
    CpuReferenceVoxelizer voxelizer;
    REQUIRE(voxelizer.voxelize(batch, {5}).data() ==
            voxelizer.voxelize(view_of(batch), {5}).data());
}

TEST_CASE("unordered events across chunks are rejected") {
    using namespace event_voxel;
    data::EventWindowSource source(
        std::make_unique<VectorReader>(std::vector<Event>{{0, 0, 10, 1}, {0, 0, 9, 1}}),
        {.window_us = 100, .chunk_us = 1000, .max_chunk_events = 1, .max_window_events = 10});
    EventWindowView window;
    REQUIRE_THROWS(source.next(window));
}

TEST_CASE("grid byte count is checked before multiplication") {
    using namespace event_voxel;
    REQUIRE_THROWS_AS(
        (VoxelGridSpec{std::numeric_limits<std::size_t>::max(), 2, 5}.element_count()),
        std::overflow_error);
    REQUIRE_THROWS_AS(
        (VoxelGridSpec{std::numeric_limits<std::size_t>::max() / 2, 1, 1}.memory_bytes()),
        std::overflow_error);
}
