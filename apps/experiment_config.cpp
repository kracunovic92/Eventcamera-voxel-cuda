#include "experiment.hpp"

#include <charconv>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace event_voxel::experiment {
namespace {

template <typename T> T parse_integer(std::string_view text, const char* option_name, T minimum) {
    T value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value < minimum)
        throw std::invalid_argument(std::string(option_name) + " has an invalid value");
    return value;
}

} // namespace

Config parse_config(int argc, char** argv) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        const std::string_view key(argv[i]);
        if (key == "--help") {
            std::cout
                << "Usage: event_voxel_experiment --input recording.hdf5 [options]\n"
                   "  --window-us N       Time-slice duration (default 100000 us = 100 ms)\n"
                   "  --bins N            Temporal voxel bins (default 5)\n"
#ifdef EVENT_VOXEL_HAS_CUDA
                   "  --strategy NAME     event-scatter (default), pixel-gather, sort-reduce\n"
                   "  --block-size N      CUDA block size: 128, 256 (default), 512\n"
#endif
                   "  --results-root DIR  Output directory (default results)\n"
                   "\nEvery time slice in the recording is processed once.\n";
            std::exit(0);
        }
        if (i + 1 == argc)
            throw std::invalid_argument("Missing value for " + std::string(key));
        const std::string value = argv[++i];
        if (key == "--input")
            config.input = value;
        else if (key == "--results-root")
            config.results_root = value;
        else if (key == "--window-us")
            config.window_us = parse_integer<std::int64_t>(value, "window-us", 1);
        else if (key == "--bins")
            config.bins = parse_integer<std::size_t>(value, "bins", 2);
#ifdef EVENT_VOXEL_HAS_CUDA
        else if (key == "--strategy") {
            if (value == "event-scatter")
                config.strategy = cuda::Strategy::EventScatter;
            else if (value == "pixel-gather")
                config.strategy = cuda::Strategy::PixelGather;
            else if (value == "sort-reduce")
                config.strategy = cuda::Strategy::SortReduce;
            else
                throw std::invalid_argument("Unknown CUDA strategy: " + value);
        } else if (key == "--block-size")
            config.block_size = parse_integer<unsigned int>(value, "block-size", 1);
#endif
        else
            throw std::invalid_argument("Unknown option: " + std::string(key));
    }
#ifdef EVENT_VOXEL_HAS_CUDA
    if (config.block_size != 128 && config.block_size != 256 && config.block_size != 512)
        throw std::invalid_argument("CUDA block size must be 128, 256 or 512");
#endif
    if (config.input.empty())
        throw std::invalid_argument("--input recording.hdf5 is required");
    return config;
}

} // namespace event_voxel::experiment
