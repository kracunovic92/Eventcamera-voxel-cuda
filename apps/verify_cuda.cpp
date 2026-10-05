#include "event_voxel/cpu/reference_voxelizer.hpp"
#include "event_voxel/cuda/voxelizer.hpp"
#include "event_voxel/data/binary_event_loader.hpp"
#include "event_voxel/data/sample_event_batch.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

std::size_t parse_count(std::string_view value) {
    std::size_t result = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size() || result == 0) {
        throw std::invalid_argument("count and bins must be positive integers");
    }
    return result;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            throw std::invalid_argument("usage: event_voxel_verify_cuda events.bin COUNT BINS");
        }
        const auto count = parse_count(argv[2]);
        const auto bins = parse_count(argv[3]);
        if (bins < 2) {
            throw std::invalid_argument("bins must be at least 2");
        }
        if (!event_voxel::cuda::is_available()) {
            throw std::runtime_error("no CUDA device is available");
        }
        event_voxel::BinaryEventLoader loader{argv[1]};
        const auto batch = event_voxel::sample_evenly(loader.load(), count);
        const event_voxel::VoxelizationSpec spec{.bins = bins};
        event_voxel::CpuReferenceVoxelizer reference;
        const auto expected = reference.voxelize(batch, spec);
        constexpr std::array strategies{
            event_voxel::cuda::Strategy::EventScatter,
            event_voxel::cuda::Strategy::PixelGather,
            event_voxel::cuda::Strategy::SortReduce,
        };
        for (const auto strategy : strategies) {
            const event_voxel::cuda::CudaVoxelizer voxelizer{strategy};
            const auto actual = voxelizer.voxelize(batch, spec);
            if (actual.data().size() != expected.data().size()) {
                throw std::runtime_error("GPU output size differs from CPU output");
            }
            for (std::size_t i = 0; i < expected.data().size(); ++i) {
                const double cpu = expected.data()[i];
                const double gpu = actual.data()[i];
                const double tolerance = 1.0e-4 + 1.0e-5 * std::max(1.0, std::abs(cpu));
                if (!std::isfinite(gpu) || std::abs(cpu - gpu) > tolerance) {
                    std::cerr << "mismatch: strategy=" << event_voxel::cuda::to_string(strategy)
                              << " voxel=" << i << " cpu=" << cpu << " gpu=" << gpu
                              << " tolerance=" << tolerance << '\n';
                    return 1;
                }
            }
            std::cout << "PASS " << event_voxel::cuda::to_string(strategy) << " events=" << count
                      << " bins=" << bins << " file=" << argv[1] << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "verification error: " << error.what() << '\n';
        return 1;
    }
}
