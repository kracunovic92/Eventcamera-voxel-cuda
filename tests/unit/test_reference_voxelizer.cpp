
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "event_voxel/cpu/reference_voxelizer.hpp"

namespace {

using Catch::Approx;
using event_voxel::CpuReferenceVoxelizer;
using event_voxel::Event;
using event_voxel::EventBatch;
using event_voxel::VoxelizationSpec;

// All standard test batches use the same recording window.
EventBatch make_batch(std::vector<Event> events) {
    return EventBatch{
        .events = std::move(events), .width = 4, .height = 3, .start_us = 0, .end_us = 100};
}

} // namespace

TEST_CASE("Empty batch produces a zero-initialized voxel grid", "[cpu]") {
    CpuReferenceVoxelizer voxelizer;

    const auto grid = voxelizer.voxelize(make_batch({}), VoxelizationSpec{.bins = 3});

    REQUIRE(grid.spec().width == 4);
    REQUIRE(grid.spec().height == 3);
    REQUIRE(grid.spec().bins == 3);
    REQUIRE(grid.data().size() == 36);

    for (const float value : grid.data()) {
        REQUIRE(value == 0.0f);
    }
}

TEST_CASE("An event at the start contributes to the first bin", "[cpu]") {
    CpuReferenceVoxelizer voxelizer;

    const auto grid =
        voxelizer.voxelize(make_batch({Event{1, 2, 0, 1}}), VoxelizationSpec{.bins = 3});

    REQUIRE(grid.at(0, 2, 1) == Approx(1.0f));
    REQUIRE(grid.at(1, 2, 1) == Approx(0.0f));
    REQUIRE(grid.at(2, 2, 1) == Approx(0.0f));
}

TEST_CASE("An event is interpolated between temporal bins", "[cpu]") {
    CpuReferenceVoxelizer voxelizer;

    const auto grid =
        voxelizer.voxelize(make_batch({Event{1, 2, 25, 1}}), VoxelizationSpec{.bins = 3});

    // Position = (25 / 100) * 2 = 0.5.
    REQUIRE(grid.at(0, 2, 1) == Approx(0.5f));
    REQUIRE(grid.at(1, 2, 1) == Approx(0.5f));
    REQUIRE(grid.at(2, 2, 1) == Approx(0.0f));
}

TEST_CASE("An OFF event produces negative contributions", "[cpu]") {
    CpuReferenceVoxelizer voxelizer;

    const auto grid =
        voxelizer.voxelize(make_batch({Event{1, 2, 75, 0}}), VoxelizationSpec{.bins = 3});

    // Position = 1.5.
    REQUIRE(grid.at(0, 2, 1) == Approx(0.0f));
    REQUIRE(grid.at(1, 2, 1) == Approx(-0.5f));
    REQUIRE(grid.at(2, 2, 1) == Approx(-0.5f));
}

TEST_CASE("Multiple events accumulate at the same voxel", "[cpu]") {
    CpuReferenceVoxelizer voxelizer;

    const auto grid =
        voxelizer.voxelize(make_batch({Event{1, 2, 25, 1}, Event{1, 2, 25, 1}, Event{1, 2, 25, 0}}),
                           VoxelizationSpec{.bins = 3});

    REQUIRE(grid.at(0, 2, 1) == Approx(0.5f));
    REQUIRE(grid.at(1, 2, 1) == Approx(0.5f));
    REQUIRE(grid.at(2, 2, 1) == Approx(0.0f));
}

TEST_CASE("Events at different pixels remain independent", "[cpu]") {
    CpuReferenceVoxelizer voxelizer;

    const auto grid = voxelizer.voxelize(make_batch({Event{0, 0, 0, 1}, Event{3, 2, 0, 0}}),
                                         VoxelizationSpec{.bins = 3});

    REQUIRE(grid.at(0, 0, 0) == Approx(1.0f));
    REQUIRE(grid.at(0, 2, 3) == Approx(-1.0f));

    // Two unrelated positions must remain zero.
    REQUIRE(grid.at(0, 0, 3) == Approx(0.0f));
    REQUIRE(grid.at(1, 2, 3) == Approx(0.0f));
}

TEST_CASE("An event at an exact bin position is not split", "[cpu]") {
    CpuReferenceVoxelizer voxelizer;

    const auto grid =
        voxelizer.voxelize(make_batch({Event{1, 2, 50, 1}}), VoxelizationSpec{.bins = 3});

    REQUIRE(grid.at(0, 2, 1) == Approx(0.0f));
    REQUIRE(grid.at(1, 2, 1) == Approx(1.0f));
    REQUIRE(grid.at(2, 2, 1) == Approx(0.0f));
}

TEST_CASE("An event near the end contributes to the last bins", "[cpu]") {
    CpuReferenceVoxelizer voxelizer;

    const auto grid =
        voxelizer.voxelize(make_batch({Event{1, 2, 99, 1}}), VoxelizationSpec{.bins = 3});

    // Position = 1.98.
    REQUIRE(grid.at(0, 2, 1) == Approx(0.0f));
    REQUIRE(grid.at(1, 2, 1) == Approx(0.02f));
    REQUIRE(grid.at(2, 2, 1) == Approx(0.98f));
}

TEST_CASE("Total signed contribution is conserved", "[cpu]") {
    CpuReferenceVoxelizer voxelizer;

    const auto grid = voxelizer.voxelize(
        make_batch({Event{0, 0, 0, 1}, Event{1, 0, 25, 1}, Event{2, 1, 75, 0}, Event{3, 2, 99, 1}}),
        VoxelizationSpec{.bins = 5});

    double total = 0.0;

    for (const float value : grid.data()) {
        total += value;
    }

    // Three ON events and one OFF event.
    REQUIRE(total == Approx(2.0));
}

TEST_CASE("Invalid voxelization configuration is rejected", "[cpu]") {
    CpuReferenceVoxelizer voxelizer;

    SECTION("Zero temporal bins") {
        REQUIRE_THROWS_AS(voxelizer.voxelize(make_batch({}), VoxelizationSpec{.bins = 0}),
                          std::invalid_argument);
    }

    SECTION("One temporal bin") {
        REQUIRE_THROWS_AS(voxelizer.voxelize(make_batch({}), VoxelizationSpec{.bins = 1}),
                          std::invalid_argument);
    }

    SECTION("Invalid recording interval") {
        auto batch = make_batch({});
        batch.end_us = batch.start_us;

        REQUIRE_THROWS_AS(voxelizer.voxelize(batch, VoxelizationSpec{.bins = 3}),
                          std::invalid_argument);
    }

    SECTION("Negative start timestamp") {
        auto batch = make_batch({});
        batch.start_us = -1;

        REQUIRE_THROWS_AS(voxelizer.voxelize(batch, VoxelizationSpec{.bins = 3}),
                          std::invalid_argument);
    }

    SECTION("Invalid sensor dimensions") {
        auto batch = make_batch({});
        batch.width = 0;

        REQUIRE_THROWS_AS(voxelizer.voxelize(batch, VoxelizationSpec{.bins = 3}),
                          std::invalid_argument);
    }
}

TEST_CASE("Invalid events are rejected", "[cpu]") {
    CpuReferenceVoxelizer voxelizer;

    SECTION("X coordinate outside sensor") {
        REQUIRE_THROWS_AS(
            voxelizer.voxelize(make_batch({Event{4, 0, 25, 1}}), VoxelizationSpec{.bins = 3}),
            std::invalid_argument);
    }

    SECTION("Y coordinate outside sensor") {
        REQUIRE_THROWS_AS(
            voxelizer.voxelize(make_batch({Event{0, 3, 25, 1}}), VoxelizationSpec{.bins = 3}),
            std::invalid_argument);
    }

    SECTION("Timestamp outside recording interval") {
        REQUIRE_THROWS_AS(
            voxelizer.voxelize(make_batch({Event{0, 0, 100, 1}}), VoxelizationSpec{.bins = 3}),
            std::invalid_argument);
    }

    SECTION("Invalid polarity") {
        REQUIRE_THROWS_AS(
            voxelizer.voxelize(make_batch({Event{0, 0, 25, 2}}), VoxelizationSpec{.bins = 3}),
            std::invalid_argument);
    }
}

TEST_CASE("Nonzero window origin preserves temporal placement", "[cpu]") {
    CpuReferenceVoxelizer voxelizer;
    EventBatch batch{
        .events =
            {
                Event{0, 0, 1000, 1},
                Event{1, 0, 1025, 1},
                Event{1, 0, 1075, 0},
                Event{2, 1, 1099, 1},
            },
        .width = 3,
        .height = 2,
        .start_us = 1000,
        .end_us = 1100,
    };
    const auto grid = voxelizer.voxelize(batch, VoxelizationSpec{.bins = 5});

    REQUIRE(grid.at(0, 0, 0) == Approx(1.0f));
    REQUIRE(grid.at(1, 0, 1) == Approx(1.0f));
    REQUIRE(grid.at(3, 0, 1) == Approx(-1.0f));
    REQUIRE(grid.at(3, 1, 2) == Approx(0.04f).margin(1e-6f));
    REQUIRE(grid.at(4, 1, 2) == Approx(0.96f).margin(1e-6f));
    for (std::size_t bin = 0; bin < 5; ++bin) {
        REQUIRE(grid.at(bin, 0, 2) == 0.0f);
    }
}

TEST_CASE("Repeated colliding events retain signed counts", "[cpu]") {
    CpuReferenceVoxelizer voxelizer;
    auto batch = make_batch({});
    batch.events.reserve(2001);
    for (std::size_t index = 0; index < 2001; ++index) {
        batch.events.push_back(Event{1, 1, 25, static_cast<std::uint8_t>(index < 1200)});
    }
    const auto grid = voxelizer.voxelize(batch, VoxelizationSpec{.bins = 3});
    // 1200 positive and 801 negative events, each split equally.
    REQUIRE(grid.at(0, 1, 1) == Approx(199.5f));
    REQUIRE(grid.at(1, 1, 1) == Approx(199.5f));
    REQUIRE(grid.at(2, 1, 1) == 0.0f);
}
