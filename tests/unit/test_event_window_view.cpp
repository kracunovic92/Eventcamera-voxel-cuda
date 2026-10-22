#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <stdexcept>

#include "event_voxel/cpu/reference_voxelizer.hpp"

TEST_CASE("CPU view and owning batch agree") {
    using namespace event_voxel;
    EventBatch batch{{{0, 0, 10, 1}, {0, 0, 30, 0}}, 2, 2, 10, 110};
    CpuReferenceVoxelizer voxelizer;
    REQUIRE(voxelizer.voxelize(batch, {5}).data() ==
            voxelizer.voxelize(view_of(batch), {5}).data());
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
