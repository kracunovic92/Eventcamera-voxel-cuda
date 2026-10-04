
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <vector>

#include "event_voxel/cpu/reference_voxelizer.hpp"
#include "event_voxel/data/binary_event_loader.hpp"
#include "event_voxel/io/npy_writer.hpp"

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
    std::cerr << "Usage: " << executable << " <events.bin> <bins> [output.npy]\n"
              << "\n"
              << "Arguments:\n"
              << "  events.bin   Prepared binary recording\n"
              << "  bins         Number of temporal bins (>= 2)\n"
              << "  output.npy   Optional NumPy output path\n";
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc != 3 && argc != 4) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        const std::filesystem::path input_path{argv[1]};

        const std::size_t bins = parse_bins(argv[2]);

        // ------------------------------------------
        // 1. Load the prepared recording
        // ------------------------------------------

        BinaryEventLoader loader(input_path);

        const EventBatch batch = loader.load();

        std::cout << "Recording:\n"
                  << "  Events: " << batch.events.size() << '\n'
                  << "  Resolution: " << batch.width << " x " << batch.height << '\n'
                  << "  Time window: [" << batch.start_us << ", " << batch.end_us << ") us\n";

        // ------------------------------------------
        // 2. Execute the CPU reference algorithm
        // ------------------------------------------

        CpuReferenceVoxelizer voxelizer;

        const VoxelizationSpec configuration{.bins = bins};

        const VoxelGrid grid = voxelizer.voxelize(batch, configuration);

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

        if (argc == 4) {
            const std::filesystem::path output_path{argv[3]};

            io::write_npy(grid, output_path);

            std::cout << "\nExport:\n"
                      << "  File: " << output_path << '\n';
        }

        return 0;

    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';

        return 1;
    }
}
