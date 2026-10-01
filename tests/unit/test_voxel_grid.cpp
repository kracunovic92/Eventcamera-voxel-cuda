#include "event_voxel/voxel_grid.hpp"

#include <catch2/catch_test_macros.hpp>

using event_voxel::voxel_index;
using event_voxel::VoxelGridSpec;

TEST_CASE("voxel grid calculates element count", "[voxel-grid]") {
    const VoxelGridSpec grid{
        .width = 4,
        .height = 3,
        .bins = 2,
    };

    REQUIRE(grid.element_count() == 24);
}

TEST_CASE("voxel indexing uses bin-major row-major layout", "[voxel-grid]") {
    const VoxelGridSpec grid{
        .width = 4,
        .height = 3,
        .bins = 2,
    };

    REQUIRE(voxel_index(grid, 0, 0, 0) == 0);
    REQUIRE(voxel_index(grid, 0, 0, 1) == 1);

    REQUIRE(voxel_index(grid, 0, 1, 0) == 4);
    REQUIRE(voxel_index(grid, 0, 2, 3) == 11);

    REQUIRE(voxel_index(grid, 1, 0, 0) == 12);
    REQUIRE(voxel_index(grid, 1, 2, 3) == 23);
}
