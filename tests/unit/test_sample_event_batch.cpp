#include "event_voxel/data/sample_event_batch.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

TEST_CASE("Even sampling keeps the first and last real events and window") {
    event_voxel::EventBatch batch{};
    batch.width = 8;
    batch.height = 6;
    batch.start_us = 100;
    batch.end_us = 200;
    for (std::int64_t timestamp = 100; timestamp < 105; ++timestamp) {
        batch.events.push_back({.x = 1, .y = 2, .timestamp_us = timestamp, .polarity = 1});
    }
    const auto selected = event_voxel::sample_evenly(batch, 3);
    REQUIRE(selected.events.size() == 3);
    REQUIRE(selected.events[0].timestamp_us == 100);
    REQUIRE(selected.events[1].timestamp_us == 102);
    REQUIRE(selected.events[2].timestamp_us == 104);
    REQUIRE(selected.start_us == 100);
    REQUIRE(selected.end_us == 200);
    REQUIRE_THROWS_AS(event_voxel::sample_evenly(batch, 6), std::invalid_argument);
}
