#include "event_voxel/cpu/reference_voxelizer.hpp"
#include "event_voxel/data/event_window_source.hpp"
#include "event_voxel/data/synthetic_generator.hpp"
#include "event_voxel/io/npy_writer.hpp"
#include "event_voxel/voxel_grid_spec.hpp"
#ifdef EVENT_VOXEL_HAS_METAVISION
#include "event_voxel/data/metavision_hdf5.hpp"
#endif
#ifdef EVENT_VOXEL_HAS_CUDA
#include "event_voxel/cuda/execution_session.hpp"
#include "event_voxel/cuda/voxelizer.hpp"
#endif
#ifdef EVENT_VOXEL_NVTX
#include <nvtx3/nvtx3.hpp>
#endif

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {
using namespace event_voxel;
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

struct Config {
    std::string source = "hdf5", mode = "pipeline", scenario = "cpu-host";
    std::string strategy = "event-scatter";
    fs::path input, results_root = "results";
    std::size_t bins = 5, warmup = 3, repeat = 10, max_windows = 100;
    std::size_t max_chunk_events = 1'000'000, max_window_events = 1'000'000;
    std::size_t synthetic_events = 100'000, export_index = 0, profile_index = 0;
    std::int64_t window_us = 100'000, chunk_us = 10'000'000;
    unsigned int block_size = 256;
};

template <typename T> T number(std::string_view s, const char* name, T minimum) {
    T value{};
    const auto [end, error] = std::from_chars(s.data(), s.data() + s.size(), value);
    if (error != std::errc{} || end != s.data() + s.size() || value < minimum)
        throw std::invalid_argument(std::string(name) + " has an invalid value");
    return value;
}

Config parse(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        const std::string_view key(argv[i]);
        if (key == "--help") {
            std::cout
                << "Usage: event_voxel_experiment --source hdf5 --input recording.hdf5 "
                   "[--mode pipeline|kernel|strategy|transfer|replay|profile|validate|export] "
                   "[--scenario cpu-host|cpu-to-device|gpu-device|gpu-to-host|all] "
                   "[--window-us N] [--bins N] [--chunk-us N] "
                   "[--max-chunk-events N] [--max-window-events N] "
                   "[--max-windows N] [--warmup N] [--repeat N] "
                   "[--strategy event-scatter|pixel-gather|sort-reduce] "
                   "[--block-size 128|256|512] [--results-root DIR] "
                   "[--profile-index N] [--export-index N]\n"
                   "Synthetic smoke input: --source synthetic [--synthetic-events N]\n";
            std::exit(0);
        }
        if (i + 1 == argc)
            throw std::invalid_argument("Missing value for " + std::string(key));
        const std::string value = argv[++i];
        if (key == "--source")
            c.source = value;
        else if (key == "--input")
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
        else if (key == "--chunk-us")
            c.chunk_us = number<std::int64_t>(value, "chunk-us", 1);
        else if (key == "--bins")
            c.bins = number<std::size_t>(value, "bins", 2);
        else if (key == "--warmup")
            c.warmup = number<std::size_t>(value, "warmup", 0);
        else if (key == "--repeat")
            c.repeat = number<std::size_t>(value, "repeat", 1);
        else if (key == "--max-windows")
            c.max_windows = number<std::size_t>(value, "max-windows", 1);
        else if (key == "--max-chunk-events")
            c.max_chunk_events = number<std::size_t>(value, "max-chunk-events", 1);
        else if (key == "--max-window-events")
            c.max_window_events = number<std::size_t>(value, "max-window-events", 1);
        else if (key == "--synthetic-events")
            c.synthetic_events = number<std::size_t>(value, "synthetic-events", 1);
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
    if (!one_of(c.source, {"hdf5", "synthetic"}) ||
        !one_of(c.mode, {"pipeline", "kernel", "strategy", "transfer", "replay", "profile",
                         "validate", "export"}) ||
        !one_of(c.scenario, {"cpu-host", "cpu-to-device", "gpu-device", "gpu-to-host", "all"}) ||
        !one_of(c.strategy, {"event-scatter", "pixel-gather", "sort-reduce"}) ||
        (c.block_size != 128 && c.block_size != 256 && c.block_size != 512)) {
        throw std::invalid_argument("Invalid source, mode, scenario, strategy or block size");
    }
    if (c.source == "hdf5" && c.input.empty())
        throw std::invalid_argument("--input is required for hdf5");
    if (c.mode == "kernel" || c.mode == "strategy" || c.mode == "transfer")
        c.scenario = "gpu-device";
    if (c.mode == "profile") {
        c.warmup = 1;
        c.repeat = 1;
    }
    return c;
}

class SyntheticReader final : public data::EventReader {
  public:
    explicit SyntheticReader(std::size_t count, std::int64_t duration)
        : events_(generate_events({.width = 64,
                                   .height = 48,
                                   .event_count = count,
                                   .duration_us = duration,
                                   .seed = 42})) {}
    bool next(Event& event) override {
        if (position_ == events_.size())
            return false;
        event = events_[position_++];
        return true;
    }
    std::uint32_t width() const override {
        return 64;
    }
    std::uint32_t height() const override {
        return 48;
    }

  private:
    std::vector<Event> events_;
    std::size_t position_ = 0;
};

std::unique_ptr<data::EventReader> open_reader(const Config& c) {
    if (c.source == "synthetic") {
        if (c.window_us > std::numeric_limits<std::int64_t>::max() / 4)
            throw std::invalid_argument("Synthetic duration overflow");
        return std::make_unique<SyntheticReader>(c.synthetic_events, c.window_us * 4);
    }
#ifdef EVENT_VOXEL_HAS_METAVISION
    return data::open_metavision_hdf5(c.input, c.max_chunk_events);
#else
    throw std::runtime_error(
        "Direct HDF5 requires EVENT_VOXEL_ENABLE_METAVISION=ON and Metavision SDK Stream");
#endif
}

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::string json_quote(std::string_view s) {
    std::string out = "\"";
    for (const char c : s) {
        if (c == '\\' || c == '"') {
            out += '\\';
            out += c;
        } else if (c == '\n')
            out += "\\n";
        else if (static_cast<unsigned char>(c) < 32)
            out += '?';
        else
            out += c;
    }
    return out + '"';
}

std::string cpu_model() {
    std::ifstream file("/proc/cpuinfo");
    std::string line;
    while (std::getline(file, line)) {
        if (line.rfind("model name", 0) == 0) {
            const auto colon = line.find(':');
            if (colon != std::string::npos)
                return line.substr(colon + 2);
        }
    }
    return "unknown";
}

double quantile(std::vector<double> values, double q) {
    if (values.empty())
        return 0;
    std::sort(values.begin(), values.end());
    return values[std::min(values.size() - 1,
                           static_cast<std::size_t>(std::ceil(q * values.size()) - 1))];
}

std::vector<std::string> scenarios(const Config& c) {
    if (c.scenario == "all")
        return {"cpu-host", "cpu-to-device", "gpu-device", "gpu-to-host"};
    return {c.scenario};
}

#ifdef EVENT_VOXEL_HAS_CUDA
cuda::Strategy selected_strategy(const std::string& s) {
    if (s == "event-scatter")
        return cuda::Strategy::EventScatter;
    if (s == "pixel-gather")
        return cuda::Strategy::PixelGather;
    return cuda::Strategy::SortReduce;
}
#endif

void validate_cpu_conservation(const EventWindowView& window, const VoxelGrid& grid) {
    std::vector<std::int64_t> expected(static_cast<std::size_t>(window.width) * window.height);
    for (const auto& event : window.events)
        expected[static_cast<std::size_t>(event.y) * window.width + event.x] +=
            event.polarity ? 1 : -1;
    for (std::size_t pixel = 0; pixel < expected.size(); ++pixel) {
        double actual = 0;
        for (std::size_t bin = 0; bin < grid.spec().bins; ++bin)
            actual += grid.data()[bin * expected.size() + pixel];
        if (!std::isfinite(actual) ||
            std::abs(actual - expected[pixel]) > 0.01 + 1e-5 * std::abs(double(expected[pixel])))
            throw std::runtime_error("CPU signed contribution conservation failed");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Config c = parse(argc, argv);
        auto reader = open_reader(c);
        data::EventWindowSource source(
            std::move(reader), {c.window_us, c.chunk_us, c.max_chunk_events, c.max_window_events});
        const auto run_stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::system_clock::now().time_since_epoch())
                                   .count();
        const auto uniqueness = Clock::now().time_since_epoch().count();
        const fs::path run_dir = c.results_root / ("run-" + std::to_string(run_stamp) + "-" +
                                                   std::to_string(uniqueness));
        if (!fs::create_directories(run_dir))
            throw std::runtime_error("Results directory already exists");
        std::ofstream windows(run_dir / "windows.csv"), samples_file(run_dir / "samples.csv");
        if (!windows || !samples_file)
            throw std::runtime_error("Cannot create result files");
        windows << "sequence,start_us,end_us,event_count,rate_per_second,partial_final,input_"
                   "preparation_ms\n";
        samples_file << "sequence,iteration,scenario,stage,wall_ms,cuda_event_ms,bytes\n";
        CpuReferenceVoxelizer cpu;
        const VoxelizationSpec spec{c.bins};
        std::map<std::string, std::vector<double>> durations;
        std::vector<double> event_counts, input_preparation_times;
        std::size_t input_preparation_over_interval = 0;
        std::size_t window_count = 0, windows_seen = 0, crossover_gpu_wins = 0,
                    crossover_compared = 0;
        bool selected_found = false, profile_found = false;
        std::size_t crossover_first_gpu_win_events = 0;
        std::size_t max_input_capacity = 0, max_output_capacity = 0;
#ifdef EVENT_VOXEL_HAS_CUDA
        std::map<std::string, std::unique_ptr<cuda::CudaExecutionSession>> gpu_sessions;
#endif
        std::map<std::string, double> replay_finish_ms, max_backlog_ms, total_idle_ms;
        std::map<std::string, std::size_t> deadline_misses;
        std::ofstream crossovers(run_dir / "crossovers.csv");
        crossovers << "sequence,window_us,event_count,cpu_to_device_median_ms,gpu_device_median_ms,"
                      "gpu_faster\n";
        std::int64_t origin_us = 0;
        std::int64_t final_end_us = 0;
        std::uint32_t sensor_width = 0, sensor_height = 0;
        auto add = [&](const EventWindowView& w, std::size_t rep, const std::string& scenario,
                       const std::string& stage, double wall, double gpu, std::size_t bytes) {
            samples_file << w.sequence << ',' << rep << ',' << scenario << ',' << stage << ','
                         << std::setprecision(12) << wall << ',' << gpu << ',' << bytes << '\n';
            durations[scenario + "/" + stage].push_back(wall);
        };
        EventWindowView w;
        for (; window_count < c.max_windows; ++window_count) {
            const auto preparation_start = Clock::now();
            bool has_window = false;
            {
#ifdef EVENT_VOXEL_NVTX
                nvtx3::scoped_range input_marker("input decode and window preparation");
#endif
                has_window = source.next(w);
            }
            if (!has_window)
                break;
            ++windows_seen;
            const double prep_ms = elapsed_ms(preparation_start);
            event_counts.push_back(static_cast<double>(w.events.size()));
            input_preparation_times.push_back(prep_ms);
            if (prep_ms > double(c.window_us) / 1000.0)
                ++input_preparation_over_interval;
            if (windows_seen == 1)
                origin_us = w.start_us;
            final_end_us = w.end_us;
            sensor_width = w.width;
            sensor_height = w.height;
            const auto grid_spec = VoxelGridSpec{w.width, w.height, c.bins};
            const auto grid_bytes = grid_spec.memory_bytes();
            windows << w.sequence << ',' << w.start_us << ',' << w.end_us << ',' << w.events.size()
                    << ',' << (1'000'000.0 * w.events.size() / (w.end_us - w.start_us)) << ','
                    << w.partial_final << ',' << prep_ms << '\n';
            if (c.mode == "profile" && w.sequence != c.profile_index)
                continue;
            if (c.mode == "profile")
                profile_found = true;
            if (c.mode == "export" || c.mode == "validate") {
                if (w.sequence != c.export_index)
                    continue;
                selected_found = true;
                const auto reference = cpu.voxelize(w, spec);
                if (c.mode == "validate")
                    validate_cpu_conservation(w, reference);
#ifdef EVENT_VOXEL_HAS_CUDA
                if (c.mode == "validate" && cuda::is_available()) {
                    for (const auto strategy :
                         {cuda::Strategy::EventScatter, cuda::Strategy::PixelGather,
                          cuda::Strategy::SortReduce}) {
                        cuda::CudaExecutionSession gpu(strategy, c.block_size);
                        gpu.reserve(w.events.size(), w.width, w.height, c.bins);
                        gpu.set_window(w, spec);
                        gpu.upload_events();
                        gpu.run_strategy();
                        std::vector<float> actual(reference.data().size());
                        gpu.download_grid(actual);
                        for (std::size_t i = 0; i < actual.size(); ++i) {
                            const double tolerance =
                                1e-4 + 1e-5 * std::max(1.0, std::abs(double(reference.data()[i])));
                            if (std::abs(double(actual[i]) - reference.data()[i]) > tolerance)
                                throw std::runtime_error("GPU validation failed at voxel " +
                                                         std::to_string(i));
                        }
                    }
                }
#endif
                if (c.mode == "export") {
                    fs::create_directories(run_dir / "outputs");
                    io::write_npy(reference, run_dir / "outputs" / "cpu.npy");
#ifdef EVENT_VOXEL_HAS_CUDA
                    if (cuda::is_available()) {
                        cuda::CudaExecutionSession gpu(selected_strategy(c.strategy), c.block_size);
                        gpu.reserve(w.events.size(), w.width, w.height, c.bins);
                        gpu.set_window(w, spec);
                        gpu.upload_events();
                        gpu.run_strategy();
                        VoxelGrid gpu_output(reference.spec());
                        gpu.download_grid(gpu_output.data());
                        io::write_npy(gpu_output,
                                      run_dir / "outputs" / ("gpu-" + c.strategy + ".npy"));
                    }
#endif
                }
                std::cout << c.mode << " completed for window " << w.sequence << '\n';
                break;
            }
            std::map<std::string, double> window_medians;
            for (const auto& scenario : scenarios(c)) {
                const bool cpu_path = scenario == "cpu-host" || scenario == "cpu-to-device";
                const bool needs_cuda = !cpu_path || scenario == "cpu-to-device" ||
                                        c.mode == "kernel" || c.mode == "strategy" ||
                                        c.mode == "transfer";
#ifndef EVENT_VOXEL_HAS_CUDA
                if (needs_cuda)
                    throw std::runtime_error("This scenario requires a CUDA build");
#else
                if (needs_cuda && !cuda::is_available())
                    throw std::runtime_error("CUDA device unavailable");
                cuda::CudaExecutionSession* gpu = nullptr;
                if (needs_cuda) {
                    auto& session = gpu_sessions[scenario];
                    if (!session)
                        session = std::make_unique<cuda::CudaExecutionSession>(
                            selected_strategy(c.strategy), c.block_size);
                    gpu = session.get();
                    gpu->reserve(w.events.size(), w.width, w.height, c.bins);
                    gpu->set_window(w, spec);
                    max_input_capacity = std::max(max_input_capacity, gpu->input_capacity());
                    max_output_capacity = std::max(max_output_capacity, gpu->output_capacity());
                }
#endif
                std::vector<float> host_grid(grid_spec.element_count());
                auto iteration = [&](std::size_t rep, bool record) {
                    const auto pipeline_start = Clock::now();
                    if (c.mode == "kernel" || c.mode == "strategy" || c.mode == "transfer") {
#ifdef EVENT_VOXEL_HAS_CUDA
                        if (c.mode == "transfer") {
                            const auto h2d = gpu->upload_events();
                            if (record)
                                add(w, rep, scenario, "h2d-events", h2d.wall_ms, h2d.cuda_ms,
                                    w.events.size_bytes());
                            const auto reference = cpu.voxelize_prevalidated(
                                w, spec); // outside any transfer stage timer
                            const auto vox_h2d = gpu->upload_host_grid(reference.data());
                            if (record)
                                add(w, rep, scenario, "h2d-cpu-grid", vox_h2d.wall_ms,
                                    vox_h2d.cuda_ms, grid_bytes);
                            const auto d2h = gpu->download_grid(host_grid);
                            if (record)
                                add(w, rep, scenario, "d2h-grid", d2h.wall_ms, d2h.cuda_ms,
                                    grid_bytes);
                        } else if (c.mode == "strategy") {
                            const auto time = gpu->run_strategy();
                            if (record)
                                add(w, rep, scenario, "strategy", time.wall_ms, time.cuda_ms, 0);
                        } else {
                            gpu->clear();
                            gpu->prepare_kernel();
                            const auto time = gpu->run_kernel();
                            if (record)
                                add(w, rep, scenario, "kernel", time.wall_ms, time.cuda_ms, 0);
                        }
#endif
                        return elapsed_ms(pipeline_start);
                    }
                    if (cpu_path) {
                        const auto start = Clock::now();
                        auto output = cpu.voxelize_prevalidated(w, spec);
                        if (record)
                            add(w, rep, scenario, "cpu-voxelize", elapsed_ms(start), 0, 0);
#ifdef EVENT_VOXEL_HAS_CUDA
                        if (scenario == "cpu-to-device") {
                            const auto time = gpu->upload_host_grid(output.data());
                            if (record)
                                add(w, rep, scenario, "h2d-cpu-grid", time.wall_ms, time.cuda_ms,
                                    grid_bytes);
                        }
#endif
                    } else {
#ifdef EVENT_VOXEL_HAS_CUDA
                        const auto h2d = gpu->upload_events();
                        if (record)
                            add(w, rep, scenario, "h2d-events", h2d.wall_ms, h2d.cuda_ms,
                                w.events.size_bytes());
                        const auto strategy_time = gpu->run_strategy();
                        if (record)
                            add(w, rep, scenario, "strategy", strategy_time.wall_ms,
                                strategy_time.cuda_ms, 0);
                        if (scenario == "gpu-to-host") {
                            const auto d2h = gpu->download_grid(host_grid);
                            if (record)
                                add(w, rep, scenario, "d2h-grid", d2h.wall_ms, d2h.cuda_ms,
                                    grid_bytes);
                        }
#endif
                    }
                    const double wall_ms = elapsed_ms(pipeline_start);
                    if (record)
                        add(w, rep, scenario, "pipeline", wall_ms, 0, 0);
                    return wall_ms;
                };
#ifdef EVENT_VOXEL_HAS_CUDA
                if (c.mode == "kernel" || c.mode == "strategy")
                    gpu->upload_events();
#endif
                for (std::size_t i = 0; i < c.warmup; ++i)
                    iteration(i, false);
                std::vector<double> replay_samples;
                for (std::size_t i = 0; i < c.repeat; ++i) {
                    const auto duration = iteration(i, true);
                    replay_samples.push_back(duration);
                }
                if (c.mode == "pipeline" || c.mode == "replay" || c.mode == "profile")
                    window_medians[scenario] = quantile(replay_samples, 0.5);
                if (c.mode == "replay") {
                    const double release_ms = double(w.end_us - origin_us) / 1000.0;
                    const double work_ms = quantile(replay_samples, 0.5);
                    const double start_ms = std::max(release_ms, replay_finish_ms[scenario]);
                    total_idle_ms[scenario] +=
                        std::max(0.0, release_ms - replay_finish_ms[scenario]);
                    max_backlog_ms[scenario] =
                        std::max(max_backlog_ms[scenario], start_ms - release_ms);
                    replay_finish_ms[scenario] = start_ms + work_ms;
                    if (replay_finish_ms[scenario] > release_ms + double(c.window_us) / 1000.0)
                        ++deadline_misses[scenario];
                    add(w, 0, scenario, "replay-backlog", start_ms - release_ms, 0, 0);
                    add(w, 0, scenario, "replay-end-to-result",
                        replay_finish_ms[scenario] - release_ms, 0, 0);
                }
            }
            if (window_medians.contains("cpu-to-device") && window_medians.contains("gpu-device")) {
                const bool gpu_faster =
                    window_medians["gpu-device"] < window_medians["cpu-to-device"];
                crossovers << w.sequence << ',' << c.window_us << ',' << w.events.size() << ','
                           << window_medians["cpu-to-device"] << ',' << window_medians["gpu-device"]
                           << ',' << gpu_faster << '\n';
                ++crossover_compared;
                if (gpu_faster) {
                    ++crossover_gpu_wins;
                    if (crossover_first_gpu_win_events == 0)
                        crossover_first_gpu_win_events = w.events.size();
                }
            }
            if (c.mode == "profile")
                break;
        }
        if (windows_seen == 0)
            throw std::runtime_error("Input contained no CD windows");
        if ((c.mode == "validate" || c.mode == "export") && !selected_found)
            throw std::runtime_error("Selected export/validation window was not found");
        if (c.mode == "profile" && !profile_found)
            throw std::runtime_error("Selected profile window was not found");
        std::ofstream metadata(run_dir / "metadata.json");
        metadata
            << "{\n  \"schema_version\": 1,\n  \"source\": " << json_quote(c.source)
            << ",\n  \"recording\": " << json_quote(c.input.string())
            << ",\n  \"mode\": " << json_quote(c.mode)
            << ",\n  \"scenario\": " << json_quote(c.scenario)
            << ",\n  \"strategy\": " << json_quote(c.strategy)
            << ",\n  \"cpu_model\": " << json_quote(cpu_model())
            << ",\n  \"cpu_threads\": " << std::thread::hardware_concurrency()
            << ",\n  \"build_type\": " << json_quote(EVENT_VOXEL_BUILD_TYPE)
            << ",\n  \"git_commit\": " << json_quote(EVENT_VOXEL_GIT_COMMIT)
            << ",\n  \"event_size_bytes\": " << sizeof(Event)
            << ",\n  \"sensor_width\": " << sensor_width
            << ",\n  \"sensor_height\": " << sensor_height
            << ",\n  \"recording_start_us\": " << origin_us
            << ",\n  \"last_window_end_us\": " << final_end_us
            << ",\n  \"peak_buffered_events\": " << source.peak_buffered_events()
            << ",\n  \"device_input_capacity_events\": " << max_input_capacity
            << ",\n  \"device_output_capacity_voxels\": " << max_output_capacity
            << ",\n  \"bins\": " << c.bins << ",\n  \"window_us\": " << c.window_us
            << ",\n  \"chunk_us\": " << c.chunk_us
            << ",\n  \"max_chunk_events\": " << c.max_chunk_events
            << ",\n  \"max_window_events\": " << c.max_window_events
            << ",\n  \"warmup\": " << c.warmup << ",\n  \"repeat\": " << c.repeat
            << ",\n  \"block_size\": " << c.block_size
            << ",\n  \"profile_index\": " << c.profile_index << ",\n  \"host_memory\": \"pageable\""
            << ",\n  \"cuda_streams\": " << (max_input_capacity ? 1 : 0)
            << ",\n  \"stream_semantics\": \"nonblocking; synchronized CUDA stop event per stage\""
            << ",\n  \"thrust_internal_allocations_possible\": "
            << (max_input_capacity && c.strategy != "event-scatter" ? "true" : "false")
            << ",\n  \"window_origin\": \"first_cd_event_timestamp\""
            << ",\n  \"final_window_policy\": "
               "\"emit_partial_through_last_event_plus_one_microsecond\""
            << ",\n  \"release_time\": \"window_end\""
            << ",\n  \"input_read_in_timing\": false,\n  \"validation_in_timing\": false,\n  "
               "\"serialization_in_timing\": false";
#ifdef EVENT_VOXEL_HAS_METAVISION
        metadata << ",\n  \"metavision_sdk_version\": "
                 << json_quote(EVENT_VOXEL_METAVISION_VERSION);
#endif
#ifdef EVENT_VOXEL_HAS_CUDA
        metadata << ",\n  \"cuda_toolkit_version\": "
                 << json_quote(EVENT_VOXEL_CUDA_TOOLKIT_VERSION);
#endif
        if (!c.input.empty())
            metadata << ",\n  \"recording_size_bytes\": " << fs::file_size(c.input);
#ifdef EVENT_VOXEL_HAS_CUDA
        if (cuda::is_available()) {
            const auto info = cuda::gpu_info();
            metadata << ",\n  \"gpu_name\": " << json_quote(info.name)
                     << ",\n  \"gpu_compute_capability\": "
                     << json_quote(std::to_string(info.compute_major) + "." +
                                   std::to_string(info.compute_minor))
                     << ",\n  \"cuda_driver_version\": " << info.driver_version
                     << ",\n  \"cuda_runtime_version\": " << info.runtime_version;
        }
#endif
        metadata << "\n}\n";
        std::ofstream summary(run_dir / "summary.json");
        summary << "{\n  \"schema_version\": 1,\n  \"windows\": " << windows_seen
                << ",\n  \"event_count_median\": " << quantile(event_counts, 0.5)
                << ",\n  \"event_count_p95\": " << quantile(event_counts, 0.95)
                << ",\n  \"input_preparation_median_ms\": "
                << quantile(input_preparation_times, 0.5)
                << ",\n  \"input_preparation_p95_ms\": " << quantile(input_preparation_times, 0.95)
                << ",\n  \"input_preparation_over_interval_windows\": "
                << input_preparation_over_interval
                << ",\n  \"crossover_compared_windows\": " << crossover_compared
                << ",\n  \"crossover_gpu_faster_windows\": " << crossover_gpu_wins
                << ",\n  \"first_observed_gpu_faster_event_count\": "
                << crossover_first_gpu_win_events << ",\n  \"replay\": {";
        bool first_replay = true;
        for (const auto& [scenario, misses] : deadline_misses) {
            if (!first_replay)
                summary << ',';
            first_replay = false;
            summary << "\n    " << json_quote(scenario) << ": {\"deadline_misses\": " << misses
                    << ", \"max_backlog_ms\": " << max_backlog_ms[scenario]
                    << ", \"total_idle_ms\": " << total_idle_ms[scenario] << '}';
        }
        summary << "\n  },\n  \"stages\": {\n";
        bool first = true;
        for (const auto& [key, values] : durations) {
            const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
            double variance = 0;
            for (double value : values)
                variance += (value - mean) * (value - mean);
            variance /= values.size();
            if (!first)
                summary << ",\n";
            first = false;
            summary << "    " << json_quote(key) << ": {\"count\": " << values.size()
                    << ", \"mean_ms\": " << mean << ", \"median_ms\": " << quantile(values, 0.5)
                    << ", \"p95_ms\": " << quantile(values, 0.95)
                    << ", \"std_ms\": " << std::sqrt(variance) << '}';
        }
        summary << "\n  }\n}\n";
        std::cout << "Wrote " << run_dir << " (" << windows_seen << " windows)\n";
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
