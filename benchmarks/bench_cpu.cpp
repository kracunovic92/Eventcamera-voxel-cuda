#include <benchmark/benchmark.h>

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "event_voxel/cpu/reference_voxelizer.hpp"
#include "event_voxel/data/binary_event_loader.hpp"

namespace {

const event_voxel::EventBatch& benchmark_batch() {
    static const event_voxel::EventBatch batch = [] {
        const char* path = std::getenv("EVENT_VOXEL_BENCH_INPUT");
        if (path == nullptr || *path == '\0') {
            throw std::runtime_error("Set EVENT_VOXEL_BENCH_INPUT to events.bin");
        }
        event_voxel::BinaryEventLoader loader{path};
        return loader.load();
    }();
    return batch;
}

std::vector<std::size_t> configured_bins() {
    const char* configured = std::getenv("EVENT_VOXEL_BENCH_BINS");
    const std::string_view text = configured == nullptr ? "2,5,10" : configured;
    std::vector<std::size_t> bins;
    std::size_t start = 0;
    while (start < text.size()) {
        const std::size_t end = text.find(',', start);
        const auto field = text.substr(start, end - start);
        std::size_t value = 0;
        const auto [next, error] =
            std::from_chars(field.data(), field.data() + field.size(), value);
        if (error != std::errc{} || next != field.data() + field.size() || value < 2) {
            throw std::invalid_argument(
                "EVENT_VOXEL_BENCH_BINS requires comma-separated integers >= 2");
        }
        bins.push_back(value);
        if (end == std::string_view::npos) {
            break;
        }
        start = end + 1;
    }
    if (bins.empty() || text.back() == ',') {
        throw std::invalid_argument("EVENT_VOXEL_BENCH_BINS cannot be empty");
    }
    return bins;
}

void cpu_voxelization(benchmark::State& state, std::size_t bins) {
    const auto& batch = benchmark_batch();
    const event_voxel::VoxelizationSpec spec{.bins = bins};
    event_voxel::CpuReferenceVoxelizer voxelizer;
    for (auto _ : state) {
        auto grid = voxelizer.voxelize(batch, spec);
        benchmark::DoNotOptimize(grid);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) *
                            static_cast<std::int64_t>(batch.events.size()));
}

const bool registered = [] {
    for (const auto bins : configured_bins()) {
        benchmark::RegisterBenchmark(
            "BM_CpuVoxelization/" + std::to_string(bins),
            [bins](benchmark::State& state) { cpu_voxelization(state, bins); })
            ->Unit(benchmark::kMillisecond)
            ->UseRealTime();
    }
    return true;
}();

} // namespace
