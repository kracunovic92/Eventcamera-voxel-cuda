#include "event_voxel/cpu/reference_voxelizer.hpp"
#include "event_voxel/cuda/strategy.hpp"
#include "event_voxel/cuda/voxelizer.hpp"
#include "event_voxel/event.hpp"
#include "event_voxel/event_batch.hpp"
#include "event_voxel/voxel_grid.hpp"
#include "event_voxel/voxelization_spec.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using event_voxel::CpuReferenceVoxelizer;
using event_voxel::Event;
using event_voxel::EventBatch;
using event_voxel::VoxelGrid;
using event_voxel::VoxelizationSpec;

using event_voxel::cuda::CudaVoxelizer;
using event_voxel::cuda::Strategy;

// --------------------------------------------------
// Test configuration
// --------------------------------------------------

constexpr double ABSOLUTE_TOLERANCE = 1.0e-4;
constexpr double RELATIVE_TOLERANCE = 1.0e-5;

constexpr std::array CUDA_STRATEGIES{
    Strategy::EventScatter,
    Strategy::PixelGather,
    Strategy::SortReduce,
};

// --------------------------------------------------
// Helpers
// --------------------------------------------------

[[nodiscard]]
Event make_event(std::uint16_t x, std::uint16_t y, std::int64_t timestamp_us,
                 std::int8_t polarity) {
    Event event{};

    event.x = x;
    event.y = y;
    event.timestamp_us = timestamp_us;
    event.polarity = polarity;

    return event;
}

[[nodiscard]]
EventBatch make_batch(std::size_t width, std::size_t height, std::int64_t start_us,
                      std::int64_t end_us, std::vector<Event> events) {
    EventBatch batch{};

    batch.width = width;
    batch.height = height;
    batch.start_us = start_us;
    batch.end_us = end_us;
    batch.events = std::move(events);

    return batch;
}

[[nodiscard]]
VoxelizationSpec make_spec(std::size_t bins) {
    VoxelizationSpec spec{};
    spec.bins = bins;
    return spec;
}

// --------------------------------------------------
// Grid comparison
// --------------------------------------------------

void require_grids_close(const VoxelGrid& expected, const VoxelGrid& actual, Strategy strategy) {
    REQUIRE(actual.data().size() == expected.data().size());

    for (std::size_t i = 0; i < expected.data().size(); ++i) {

        const double expected_value = static_cast<double>(expected.data()[i]);

        const double actual_value = static_cast<double>(actual.data()[i]);

        const double difference = std::abs(expected_value - actual_value);

        const double scale = std::max(1.0, std::abs(expected_value));

        const double allowed_error = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * scale;

        INFO("strategy = " << event_voxel::cuda::to_string(strategy));

        INFO("voxel index = " << i);

        INFO("expected = " << expected_value);

        INFO("actual = " << actual_value);

        INFO("difference = " << difference);

        INFO("allowed error = " << allowed_error);

        REQUIRE(difference <= allowed_error);
    }
}

// --------------------------------------------------
// Execute all CUDA strategies against CPU reference
// --------------------------------------------------

void require_all_cuda_strategies_match(const EventBatch& batch, const VoxelizationSpec& spec) {
    if (!event_voxel::cuda::is_available()) {
        SKIP("CUDA device is not available");
    }

    CpuReferenceVoxelizer cpu;

    const VoxelGrid expected = cpu.voxelize(batch, spec);

    for (const Strategy strategy : CUDA_STRATEGIES) {

        INFO("Testing CUDA strategy: " << event_voxel::cuda::to_string(strategy));

        const CudaVoxelizer gpu{strategy};

        const VoxelGrid actual = gpu.voxelize(batch, spec);

        require_grids_close(expected, actual, strategy);
    }
}

} // namespace

// ==================================================
// Tests
// ==================================================

TEST_CASE("CUDA voxelizers match CPU for one event", "[cuda][correctness]") {
    const EventBatch batch = make_batch(4, 3, 1000, 2000,
                                        {
                                            make_event(1, 1, 1000, 1),
                                        });

    const VoxelizationSpec spec = make_spec(5);

    require_all_cuda_strategies_match(batch, spec);
}

TEST_CASE("CUDA voxelizers match CPU for temporal interpolation",
          "[cuda][correctness][interpolation]") {
    // Window:
    //
    // 1000 --------------------------- 2000
    //
    // bins = 5
    //
    // temporal positions:
    //
    // 0       1       2       3       4
    //
    // Timestamp 1125 maps to position 0.5.
    // Therefore its contribution is split evenly
    // between bins 0 and 1.

    const EventBatch batch = make_batch(4, 3, 1000, 2000,
                                        {
                                            make_event(1, 1, 1125, 1),

                                            make_event(2, 1, 1375, 1),

                                            make_event(3, 2, 1625, 0),
                                        });

    const VoxelizationSpec spec = make_spec(5);

    require_all_cuda_strategies_match(batch, spec);
}

TEST_CASE("CUDA voxelizers match CPU for positive and negative polarity",
          "[cuda][correctness][polarity]") {
    const EventBatch batch = make_batch(4, 3, 1000, 2000,
                                        {
                                            make_event(1, 1, 1250, 1),

                                            make_event(1, 1, 1250, 0),

                                            make_event(2, 2, 1500, 1),

                                            make_event(2, 2, 1500, 0),
                                        });

    const VoxelizationSpec spec = make_spec(5);

    require_all_cuda_strategies_match(batch, spec);
}

TEST_CASE("CUDA voxelizers match CPU for events across many pixels",
          "[cuda][correctness][spatial]") {
    std::vector<Event> events;

    constexpr std::size_t width = 16;
    constexpr std::size_t height = 12;

    constexpr std::int64_t start_us = 1000;

    constexpr std::int64_t end_us = 11000;

    for (std::size_t i = 0; i < 500; ++i) {

        const auto x = static_cast<std::uint16_t>((i * 7) % width);

        const auto y = static_cast<std::uint16_t>((i * 11) % height);

        const auto timestamp = start_us + static_cast<std::int64_t>((i * 37) % (end_us - start_us));

        const auto polarity = static_cast<std::int8_t>(i % 2);

        events.push_back(make_event(x, y, timestamp, polarity));
    }

    const EventBatch batch = make_batch(width, height, start_us, end_us, std::move(events));

    const VoxelizationSpec spec = make_spec(5);

    require_all_cuda_strategies_match(batch, spec);
}

TEST_CASE("CUDA voxelizers match CPU under heavy voxel contention",
          "[cuda][correctness][contention]") {
    std::vector<Event> events;

    constexpr std::size_t event_count = 4096;

    constexpr std::int64_t start_us = 1000;

    constexpr std::int64_t end_us = 2000;

    events.reserve(event_count);

    for (std::size_t i = 0; i < event_count; ++i) {

        // Every event belongs to the same pixel.
        //
        // Timestamps stay in a narrow range so many
        // contributions also hit the same temporal bins.
        //
        // This deliberately stresses:
        //
        // EventScatter:
        //     global atomic contention
        //
        // PixelGather:
        //     one very busy pixel thread
        //
        // SortReduce:
        //     many identical voxel keys

        const auto timestamp = static_cast<std::int64_t>(1200 + (i % 50));

        const auto polarity = static_cast<std::int8_t>(i % 3 == 0 ? 0 : 1);

        events.push_back(make_event(3, 2, timestamp, polarity));
    }

    const EventBatch batch = make_batch(8, 6, start_us, end_us, std::move(events));

    const VoxelizationSpec spec = make_spec(5);

    require_all_cuda_strategies_match(batch, spec);
}

TEST_CASE("CUDA voxelizers match CPU near temporal boundaries", "[cuda][correctness][boundary]") {
    const EventBatch batch = make_batch(4, 4, 1000, 2000,
                                        {
                                            // Beginning of window.
                                            make_event(0, 0, 1000, 1),

                                            // Just after beginning.
                                            make_event(1, 0, 1001, 1),

                                            // Near end of window.
                                            make_event(2, 0, 1998, 1),

                                            // Last valid timestamp because
                                            // end_us itself is exclusive.
                                            make_event(3, 0, 1999, 0),
                                        });

    const VoxelizationSpec spec = make_spec(10);

    require_all_cuda_strategies_match(batch, spec);
}
