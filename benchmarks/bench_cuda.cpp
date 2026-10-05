#include "event_voxel/cuda/voxelizer.hpp"
#include "event_voxel/data/binary_event_loader.hpp"
#include "event_voxel/data/sample_event_batch.hpp"

#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

std::size_t parse_number(std::string_view text, const char* name, std::size_t minimum) {
    std::size_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value < minimum) {
        throw std::invalid_argument(std::string(name) +
                                    " must be an integer >= " + std::to_string(minimum));
    }
    return value;
}

event_voxel::cuda::Strategy parse_strategy(std::string_view value) {
    using event_voxel::cuda::Strategy;
    if (value == "event-scatter")
        return Strategy::EventScatter;
    if (value == "pixel-gather")
        return Strategy::PixelGather;
    if (value == "sort-reduce")
        return Strategy::SortReduce;
    throw std::invalid_argument("unknown strategy: " + std::string(value));
}

event_voxel::EventBatch make_batch(std::size_t event_count, std::string_view workload,
                                   std::string_view input) {
    if (workload == "real") {
        if (input.empty()) {
            throw std::invalid_argument("real workload requires an input events.bin file");
        }
        event_voxel::BinaryEventLoader loader{std::string(input)};
        return event_voxel::sample_evenly(loader.load(), event_count);
    }
    if (workload != "uniform" && workload != "collision-heavy") {
        throw std::invalid_argument("workload must be real, uniform, or collision-heavy");
    }

    event_voxel::EventBatch batch{};
    batch.width = 240;
    batch.height = 180;
    batch.start_us = 0;
    batch.end_us = 100000;
    batch.events.reserve(event_count);

    // Identical arguments produce identical input across runs and strategies.
    for (std::size_t i = 0; i < event_count; ++i) {
        event_voxel::Event event{};
        if (workload == "collision-heavy") {
            event.x = 120;
            event.y = 90;
            event.timestamp_us = static_cast<std::int64_t>(49000 + i % 200);
        } else {
            event.x = static_cast<std::uint16_t>((i * 101 + i / 431) % batch.width);
            event.y = static_cast<std::uint16_t>((i * 73 + i / 257) % batch.height);
            event.timestamp_us = static_cast<std::int64_t>((i * 7919ULL) % 100000ULL);
        }
        event.polarity = static_cast<std::uint8_t>((i / 3) % 2);
        batch.events.push_back(event);
    }
    return batch;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 6 || argc > 8) {
            throw std::invalid_argument(
                "usage: event_voxel_bench_cuda STRATEGY EVENT_COUNT BINS WARMUP MEASUREMENTS "
                "[real|uniform|collision-heavy] [events.bin]");
        }
        const auto strategy = parse_strategy(argv[1]);
        const auto event_count = parse_number(argv[2], "event count", 1);
        const auto bins = parse_number(argv[3], "bins", 2);
        const auto warmups = parse_number(argv[4], "warmups", 0);
        const auto iterations = parse_number(argv[5], "measurements", 1);
        const std::string_view workload = argc >= 7 ? argv[6] : "uniform";
        const std::string_view input = argc == 8 ? argv[7] : "";
        if ((workload == "real") != (argc == 8)) {
            throw std::invalid_argument("real workload requires exactly one events.bin path");
        }
        if (event_count > std::numeric_limits<std::size_t>::max() / sizeof(event_voxel::Event)) {
            throw std::overflow_error("event allocation size overflow");
        }
        if (!event_voxel::cuda::is_available()) {
            throw std::runtime_error("no CUDA device is available");
        }

        const auto batch = make_batch(event_count, workload, input);
        const event_voxel::VoxelizationSpec spec{.bins = bins};
        const event_voxel::cuda::CudaVoxelizer voxelizer{strategy};
        for (std::size_t i = 0; i < warmups; ++i) {
            (void)voxelizer.voxelize(batch, spec);
        }

        std::cout << "strategy,workload,event_count,bins,iteration,end_to_end_ms\n";
        std::cout << std::fixed << std::setprecision(9);
        for (std::size_t i = 0; i < iterations; ++i) {
            const auto start = std::chrono::steady_clock::now();
            auto grid = voxelizer.voxelize(batch, spec);
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();
            std::cout << event_voxel::cuda::to_string(strategy) << ',' << workload << ','
                      << event_count << ',' << bins << ',' << i << ',' << elapsed << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "benchmark error: " << error.what() << '\n';
        return 1;
    }
}
