
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>

#include "event_voxel/cpu/reference_voxelizer.hpp"

namespace {

double sum_voxels(const event_voxel::VoxelGrid& grid) {
    double total = 0.0;

    for (const float value : grid.data()) {
        REQUIRE(std::isfinite(value));
        total += value;
    }

    return total;
}

double sum_event_polarities(const event_voxel::EventBatch& batch) {
    double total = 0.0;

    for (const auto& event : batch.events) {
        total += event.polarity == 1 ? 1.0 : -1.0;
    }

    return total;
}

} // namespace

TEST_CASE("CPU voxelization conserves signed contributions", "[cpu][integration]") {
    const event_voxel::EventBatch batch{
        .events = {{0, 0, 0, 1},
                   {1, 1, 20, 0},
                   {2, 1, 40, 1},
                   {0, 0, 60, 1},
                   {1, 1, 80, 0},
                   {2, 1, 99, 1}},
        .width = 3,
        .height = 2,
        .start_us = 0,
        .end_us = 100,
    };

    event_voxel::CpuReferenceVoxelizer voxelizer;

    const double expected = sum_event_polarities(batch);

    for (const std::size_t bins : {2U, 5U, 10U}) {
        const auto grid = voxelizer.voxelize(batch, event_voxel::VoxelizationSpec{.bins = bins});

        REQUIRE(grid.spec().bins == bins);
        REQUIRE(grid.spec().width == batch.width);
        REQUIRE(grid.spec().height == batch.height);

        REQUIRE(sum_voxels(grid) == Catch::Approx(expected).margin(0.01));
    }
}
