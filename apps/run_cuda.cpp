#include "event_voxel/cuda/voxelizer.hpp"
#include "event_voxel/data/binary_event_loader.hpp"
#include "event_voxel/data/sample_event_batch.hpp"
#include "event_voxel/io/npy_writer.hpp"
#include "event_voxel/io/run_record.hpp"

#include <cuda_runtime.h>

#include <charconv>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

std::size_t parse_positive(std::string_view text, const char* name, std::size_t minimum) {
    std::size_t result = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), result);
    if (error != std::errc{} || end != text.data() + text.size() || result < minimum) {
        throw std::invalid_argument(std::string(name) +
                                    " must be an integer >= " + std::to_string(minimum));
    }
    return result;
}

event_voxel::cuda::Strategy parse_strategy(std::string_view name) {
    using event_voxel::cuda::Strategy;
    if (name == "event-scatter")
        return Strategy::EventScatter;
    if (name == "pixel-gather")
        return Strategy::PixelGather;
    if (name == "sort-reduce")
        return Strategy::SortReduce;
    throw std::invalid_argument("Unknown CUDA strategy: " + std::string(name));
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <events.bin> <bins> <output.npy> --strategy"
                     " event-scatter|pixel-gather|sort-reduce"
                     " [--count N] [--metrics run.json]\n";
        return 2;
    }
    try {
        using Clock = std::chrono::steady_clock;
        const auto milliseconds = [](Clock::time_point start, Clock::time_point end) {
            return std::chrono::duration<double, std::milli>(end - start).count();
        };
        const std::filesystem::path input_path{argv[1]};
        const std::size_t bins = parse_positive(argv[2], "bins", 2);
        const std::filesystem::path output_path{argv[3]};
        std::filesystem::path metrics_path;
        std::size_t selected_count = 0;
        std::string strategy_name;
        for (int index = 4; index < argc; ++index) {
            const std::string_view argument{argv[index]};
            if (argument == "--strategy" && index + 1 < argc) {
                strategy_name = argv[++index];
            } else if (argument == "--count" && index + 1 < argc) {
                selected_count = parse_positive(argv[++index], "count", 1);
            } else if (argument == "--metrics" && index + 1 < argc) {
                metrics_path = argv[++index];
            } else {
                throw std::invalid_argument("Unexpected or incomplete argument: " +
                                            std::string(argument));
            }
        }
        const auto strategy = parse_strategy(strategy_name);
        if (!event_voxel::cuda::is_available()) {
            throw std::runtime_error("No CUDA device is available");
        }

        const auto load_start = Clock::now();
        event_voxel::BinaryEventLoader loader{input_path};
        event_voxel::EventBatch batch = loader.load();
        const std::size_t source_count = batch.events.size();
        if (selected_count != 0) {
            batch = event_voxel::sample_evenly(std::move(batch), selected_count);
        }
        const auto load_end = Clock::now();

        std::size_t free_before = 0;
        std::size_t total_bytes = 0;
        check_cuda(cudaMemGetInfo(&free_before, &total_bytes), "cudaMemGetInfo before run");
        const event_voxel::VoxelizationSpec specification{.bins = bins};
        const event_voxel::cuda::CudaVoxelizer voxelizer{strategy};
        // voxelize() synchronizes the device and returns a host-resident grid.
        // This CPU clock measures the complete call, not kernel-only GPU time.
        const auto voxelize_start = Clock::now();
        const auto grid = voxelizer.voxelize(batch, specification);
        const auto voxelize_end = Clock::now();
        std::size_t free_after = 0;
        std::size_t total_after = 0;
        check_cuda(cudaMemGetInfo(&free_after, &total_after), "cudaMemGetInfo after run");

        const auto export_start = Clock::now();
        event_voxel::io::write_npy(grid, output_path);
        const double export_ms = milliseconds(export_start, Clock::now());
        if (!metrics_path.empty()) {
            event_voxel::io::write_run_record(
                metrics_path, {
                                  .backend = "cuda",
                                  .strategy = event_voxel::cuda::to_string(strategy),
                                  .input = input_path,
                                  .output = output_path,
                                  .source_events = source_count,
                                  .selected_events = batch.events.size(),
                                  .bins = bins,
                                  .width = batch.width,
                                  .height = batch.height,
                                  .output_bytes = grid.spec().memory_bytes(),
                                  .start_us = batch.start_us,
                                  .end_us = batch.end_us,
                                  .load_ms = milliseconds(load_start, load_end),
                                  .voxelize_ms = milliseconds(voxelize_start, voxelize_end),
                                  .export_ms = export_ms,
                                  .gpu_total_bytes = total_bytes,
                                  .gpu_free_before_bytes = free_before,
                                  .gpu_free_after_bytes = free_after,
                              });
        }
        std::cout << "CUDA " << event_voxel::cuda::to_string(strategy) << ": "
                  << batch.events.size() << " events, " << bins << " bins, output " << output_path
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CUDA run error: " << error.what() << '\n';
        return 1;
    }
}
