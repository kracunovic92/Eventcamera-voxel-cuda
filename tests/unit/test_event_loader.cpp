#include <catch2/catch_test_macros.hpp>

#include "event_voxel/data/event_loader.hpp"
#include "event_voxel/data/synthetic_event_loader.hpp"

using namespace event_voxel;

TEST_CASE("Synthetic loader implements event loader interface") {

    GeneratorConfig config;
    config.width = 128;
    config.height = 128;
    config.event_count = 1000;
    config.seed = 42;

    SyntheticEventLoader synthetic(config);
    IEventLoader& loader = synthetic;

    const EventBatch batch = loader.load();

    REQUIRE(batch.events.size() == 1000);
    REQUIRE(batch.width == 128);
    REQUIRE(batch.height == 128);

    for (const auto& event : batch.events) {

        CHECK(event.x < batch.width);
        CHECK(event.y < batch.height);
        CHECK(event.polarity <= 1);
        CHECK(event.timestamp_us >= batch.start_us);
        CHECK(event.timestamp_us < batch.end_us);
    }
}
