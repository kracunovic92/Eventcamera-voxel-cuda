
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "event_voxel/cpu/reference_voxelizer.hpp"
#include "event_voxel/data/binary_event_loader.hpp"
#include "event_voxel/data/sample_event_batch.hpp"
#include "event_voxel/io/npy_writer.hpp"
#include "event_voxel/io/run_record.hpp"

namespace {

using namespace event_voxel;

// Parse the number of temporal bins without accepting
// partially valid strings such as "5abc".
[[nodiscard]]
std::size_t parse_bins(std::string_view argument) {
    std::size_t bins = 0;

    const auto [ptr, error] =
        std::from_chars(argument.data(), argument.data() + argument.size(), bins);

    if (error != std::errc{} || ptr != argument.data() + argument.size() || bins < 2) {
        throw std::invalid_argument("The number of temporal bins must be an integer >= 2");
    }

    return bins;
}

[[nodiscard]]
std::size_t parse_count(std::string_view argument) {
    std::size_t count = 0;
    const auto [end, error] =
        std::from_chars(argument.data(), argument.data() + argument.size(), count);
    if (error != std::errc{} || end != argument.data() + argument.size() || count == 0) {
        throw std::invalid_argument("Event count must be an integer >= 1");
    }
    return count;
}

// Check conservation independently at each spatial pixel.
//
// Temporal interpolation should distribute each signed
// event contribution across bins without changing its sum.
[[nodiscard]]
double max_spatial_conservation_error(const EventBatch& batch, const VoxelGrid& grid) {
    const std::size_t width = batch.width;
    const std::size_t height = batch.height;

    const VoxelGridSpec& spec = grid.spec();

    if (spec.width != width || spec.height != height) {
        throw std::runtime_error("Voxel grid dimensions do not match the recording");
    }

    // Calculate expected signed event contributions.
    std::vector<std::int64_t> expected(width * height, 0);

    for (const Event& event : batch.events) {
        const std::size_t index = static_cast<std::size_t>(event.y) * width + event.x;

        expected[index] += event.polarity == 1 ? 1 : -1;
    }

    double maximum_error = 0.0;

    // Independently sum the actual voxel values
    // across every temporal bin at each pixel.
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            double actual = 0.0;

            for (std::size_t bin = 0; bin < spec.bins; ++bin) {
                const float value = grid.at(bin, y, x);

                if (!std::isfinite(value)) {
                    throw std::runtime_error("Voxel grid contains non-finite values");
                }

                actual += static_cast<double>(value);
            }

            const std::size_t index = y * width + x;

            const double error = std::abs(actual - static_cast<double>(expected[index]));

            maximum_error = std::max(maximum_error, error);
        }
    }

    return maximum_error;
}

void print_usage(const char* executable) {
    std::cerr << "Usage: " << executable
              << " <events.bin> <bins> [output.npy] [--count N] [--metrics run.json]\n"
              << "\n"
              << "Arguments:\n"
              << "  events.bin   Prepared binary recording\n"
              << "  bins         Number of temporal bins (>= 2)\n"
              << "  output.npy   Optional NumPy output path\n"
              << "  --count N    Evenly sample N events from the input\n"
              << "  --metrics    Save one-run timing and memory information\n";
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        const std::filesystem::path input_path{argv[1]};

        const std::size_t bins = parse_bins(argv[2]);
        std::filesystem::path output_path;
        std::filesystem::path metrics_path;
        std::size_t selected_count = 0;
        for (int index = 3; index < argc; ++index) {
            const std::string_view argument{argv[index]};
            if (argument == "--count" && index + 1 < argc) {
                selected_count = parse_count(argv[++index]);
            } else if (argument == "--metrics" && index + 1 < argc) {
                metrics_path = argv[++index];
            } else if (!argument.starts_with("--") && output_path.empty()) {
                output_path = argv[index];
            } else {
                throw std::invalid_argument("Unexpected or incomplete argument: " +
                                            std::string(argument));
            }
        }
        using Clock = std::chrono::steady_clock;
        const auto milliseconds = [](Clock::time_point start, Clock::time_point end) {
            return std::chrono::duration<double, std::milli>(end - start).count();
        };

        // ------------------------------------------
        // 1. Load the prepared recording
        // ------------------------------------------

        const auto load_start = Clock::now();
        BinaryEventLoader loader(input_path);
        EventBatch batch = loader.load();
        const std::size_t source_count = batch.events.size();
        if (selected_count != 0) {
            batch = sample_evenly(std::move(batch), selected_count);
        }
        const auto load_end = Clock::now();

        std::cout << "Recording:\n"
                  << "  Events: " << batch.events.size() << '\n'
                  << "  Resolution: " << batch.width << " x " << batch.height << '\n'
                  << "  Time window: [" << batch.start_us << ", " << batch.end_us << ") us\n";

        // ------------------------------------------
        // 2. Execute the CPU reference algorithm
        // ------------------------------------------

        CpuReferenceVoxelizer voxelizer;

        const VoxelizationSpec configuration{.bins = bins};

        const auto voxelize_start = Clock::now();
        const VoxelGrid grid = voxelizer.voxelize(batch, configuration);
        const auto voxelize_end = Clock::now();

        // ------------------------------------------
        // 3. Report the resulting tensor
        // ------------------------------------------

        const VoxelGridSpec& spec = grid.spec();

        std::cout << "\nVoxel grid:\n"
                  << "  Shape (B, H, W): " << spec.bins << " x " << spec.height << " x "
                  << spec.width << '\n'
                  << "  Elements: " << spec.element_count() << '\n'
                  << "  Memory: " << spec.memory_bytes() << " bytes\n";

        // ------------------------------------------
        // 4. Validate spatial conservation
        // ------------------------------------------

        const double maximum_error = max_spatial_conservation_error(batch, grid);

        std::cout << "\nValidation:\n"
                  << "  Maximum spatial conservation error: " << maximum_error << '\n';

        // Initial numerical threshold for our CPU
        // reference experiments.
        constexpr double kTolerance = 0.01;

        if (maximum_error > kTolerance) {
            std::cerr << "FAILED: Spatial conservation error "
                      << "exceeds the tolerance\n";

            return 1;
        }

        std::cout << "  PASSED: Spatial conservation\n";

        // ------------------------------------------
        // 5. Optional NumPy export
        // ------------------------------------------

        double export_ms = 0.0;
        if (!output_path.empty()) {
            const auto export_start = Clock::now();
            io::write_npy(grid, output_path);
            export_ms = milliseconds(export_start, Clock::now());

            std::cout << "\nExport:\n"
                      << "  File: " << output_path << '\n';
        }

        if (!metrics_path.empty()) {
            io::write_run_record(metrics_path,
                                 {
                                     .backend = "cpu",
                                     .strategy = "reference",
                                     .input = input_path,
                                     .output = output_path,
                                     .source_events = source_count,
                                     .selected_events = batch.events.size(),
                                     .bins = bins,
                                     .width = batch.width,
                                     .height = batch.height,
                                     .output_bytes = spec.memory_bytes(),
                                     .start_us = batch.start_us,
                                     .end_us = batch.end_us,
                                     .load_ms = milliseconds(load_start, load_end),
                                     .voxelize_ms = milliseconds(voxelize_start, voxelize_end),
                                     .export_ms = export_ms,
                                 });
        }

        return 0;

    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';

        return 1;
    }
}
