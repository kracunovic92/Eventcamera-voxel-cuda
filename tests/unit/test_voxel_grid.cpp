
#include <catch2/catch_test_macros.hpp>

#include <stdexcept>

#include "event_voxel/voxel_grid.hpp"

using namespace event_voxel;

TEST_CASE("Voxel grid dimensions and size", "[grid]") {
    const VoxelGridSpec spec{.width = 4, .height = 3, .bins = 5};

    REQUIRE(spec.element_count() == 60);
    REQUIRE(spec.memory_bytes() == 60 * sizeof(float));
}

TEST_CASE("Voxel grid uses bin-major row-major indexing", "[grid]") {
    const VoxelGridSpec spec{.width = 3, .height = 2, .bins = 2};

    REQUIRE(voxel_index(spec, 0, 0, 0) == 0);
    REQUIRE(voxel_index(spec, 0, 1, 2) == 5);
    REQUIRE(voxel_index(spec, 1, 0, 0) == 6);
    REQUIRE(voxel_index(spec, 1, 1, 2) == 11);
}

TEST_CASE("Voxel grid initializes values to zero", "[grid]") {
    VoxelGrid grid(VoxelGridSpec{2, 2, 3});

    REQUIRE(grid.data().size() == 12);

    for (const float value : grid.data()) {
        REQUIRE(value == 0.0f);
    }
}

TEST_CASE("Voxel grid accumulates contributions", "[grid]") {
    VoxelGrid grid(VoxelGridSpec{2, 2, 3});

    grid.accumulate(1, 0, 1, 0.75f);
    grid.accumulate(1, 0, 1, -0.25f);

    REQUIRE(grid.at(1, 0, 1) == 0.5f);
    REQUIRE(grid.at(0, 0, 1) == 0.0f);
}

TEST_CASE("Voxel grid rejects invalid coordinates", "[grid]") {
    VoxelGrid grid(VoxelGridSpec{2, 2, 3});

    REQUIRE_THROWS_AS(grid.at(3, 0, 0), std::out_of_range);

    REQUIRE_THROWS_AS(grid.accumulate(0, 2, 0, 1.0f), std::out_of_range);
}

TEST_CASE("Voxel grid rejects zero dimensions", "[grid]") {
    REQUIRE_THROWS_AS(VoxelGrid(VoxelGridSpec{0, 2, 3}), std::invalid_argument);
}
