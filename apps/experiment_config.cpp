#include "experiment.hpp"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace event_voxel::experiment {
namespace {

template <typename T> T number(std::string_view s, const char* name, T minimum) {
    T value{};
    const auto [end, error] = std::from_chars(s.data(), s.data() + s.size(), value);
    if (error != std::errc{} || end != s.data() + s.size() || value < minimum)
        throw std::invalid_argument(std::string(name) + " has an invalid value");
    return value;
}

} // namespace

Config parse(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        const std::string_view key(argv[i]);
        if (key == "--help") {
            std::cout
                << "Usage: event_voxel_experiment --input recording.hdf5 [options]\n"
                   "\nCore options:\n"
                   "  --window-us N       Event window duration (default 100000 us = 100 ms)\n"
                   "  --bins N            Temporal voxel bins (default 5)\n"
                   "  --strategy NAME     event-scatter (default), pixel-gather, sort-reduce\n"
                   "\nMeasurements:\n"
                   "  --mode NAME         What to measure: pipeline (default), kernel,\n"
                   "                      strategy, transfer, replay, profile, validate, export\n"
                   "  --scenario NAME     Processing path/output: auto (default), cpu-host,\n"
                   "                      cpu-to-device, gpu-device, gpu-to-host, all\n"
                   "                      auto runs all paths with a GPU, CPU host otherwise\n"
                   "  --max-windows N     Maximum windows to process (default 100)\n"
                   "  --warmup N          Warmup repetitions per window (default 3)\n"
                   "  --repeat N          Measured repetitions per window (default 10)\n"
                   "\nAdvanced options:\n"
                   "  --max-window-events N  Per-window event cap (default 1000000)\n"
                   "  --block-size N         CUDA block size: 128, 256 (default), 512\n"
                   "  --profile-index N      Window to profile (default 0)\n"
                   "  --export-index N       Window to validate/export (default 0)\n"
                   "  --results-root DIR     Output directory (default results)\n";
            std::exit(0);
        }
        if (i + 1 == argc)
            throw std::invalid_argument("Missing value for " + std::string(key));
        const std::string value = argv[++i];
        if (key == "--input")
            c.input = value;
        else if (key == "--mode")
            c.mode = value;
        else if (key == "--scenario")
            c.scenario = value;
        else if (key == "--strategy")
            c.strategy = value;
        else if (key == "--results-root")
            c.results_root = value;
        else if (key == "--window-us")
            c.window_us = number<std::int64_t>(value, "window-us", 1);
        else if (key == "--bins")
            c.bins = number<std::size_t>(value, "bins", 2);
        else if (key == "--warmup")
            c.warmup = number<std::size_t>(value, "warmup", 0);
        else if (key == "--repeat")
            c.repeat = number<std::size_t>(value, "repeat", 1);
        else if (key == "--max-windows")
            c.max_windows = number<std::size_t>(value, "max-windows", 1);
        else if (key == "--max-window-events")
            c.max_window_events = number<std::size_t>(value, "max-window-events", 1);
        else if (key == "--export-index")
            c.export_index = number<std::size_t>(value, "export-index", 0);
        else if (key == "--profile-index")
            c.profile_index = number<std::size_t>(value, "profile-index", 0);
        else if (key == "--block-size")
            c.block_size = number<unsigned int>(value, "block-size", 1);
        else
            throw std::invalid_argument("Unknown option: " + std::string(key));
    }
    const auto one_of = [](const std::string& value,
                           std::initializer_list<std::string_view> choices) {
        return std::find(choices.begin(), choices.end(), value) != choices.end();
    };
    if (!one_of(c.mode, {"pipeline", "kernel", "strategy", "transfer", "replay", "profile",
                         "validate", "export"}) ||
        !one_of(c.scenario,
                {"auto", "cpu-host", "cpu-to-device", "gpu-device", "gpu-to-host", "all"}) ||
        !one_of(c.strategy, {"event-scatter", "pixel-gather", "sort-reduce"}) ||
        (c.block_size != 128 && c.block_size != 256 && c.block_size != 512)) {
        throw std::invalid_argument("Invalid mode, scenario, strategy or block size");
    }
    if (c.input.empty())
        throw std::invalid_argument("--input recording.hdf5 is required");
    if (c.mode == "kernel" || c.mode == "strategy" || c.mode == "transfer")
        c.scenario = "gpu-device";
    if (c.mode == "profile") {
        c.warmup = 1;
        c.repeat = 1;
    }
    return c;
}

} // namespace event_voxel::experiment
