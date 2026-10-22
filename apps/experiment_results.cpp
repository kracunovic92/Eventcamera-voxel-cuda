#include "experiment.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <thread>

namespace event_voxel::experiment {
namespace {
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

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

fs::path make_run_directory(const fs::path& root) {
    const auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
    const auto uniqueness = Clock::now().time_since_epoch().count();
    const fs::path dir = root / ("run-" + std::to_string(stamp) + "-" + std::to_string(uniqueness));
    if (!fs::create_directories(dir))
        throw std::runtime_error("Results directory already exists");
    return dir;
}

} // namespace

double quantile(std::vector<double> values, double q) {
    if (values.empty())
        return 0;
    std::sort(values.begin(), values.end());
    return values[std::min(values.size() - 1,
                           static_cast<std::size_t>(std::ceil(q * values.size()) - 1))];
}

// Open all result files together so every run starts with the same schema.
ResultFiles::ResultFiles(const fs::path& root)
    : dir(make_run_directory(root)), windows(dir / "windows.csv"), samples(dir / "samples.csv"),
      crossovers(dir / "crossovers.csv") {
    if (!windows || !samples || !crossovers)
        throw std::runtime_error("Cannot create result files");
    windows << "sequence,start_us,end_us,event_count,rate_per_second,partial_final,input_"
               "preparation_ms\n";
    samples << "sequence,iteration,scenario,stage,wall_ms,cuda_event_ms,bytes\n";
    crossovers << "sequence,window_us,event_count,cpu_to_device_median_ms,gpu_device_median_ms,"
                  "gpu_faster\n";
}

void Experiment::write_metadata() const {
    const auto& c = config_;
    const auto& s = stats_;
    std::ofstream metadata(files_.dir / "metadata.json");
    if (!metadata)
        throw std::runtime_error("Cannot create metadata.json");
    metadata << "{\n  \"schema_version\": 2,\n  \"source\": \"hdf5\""
             << ",\n  \"recording\": " << json_quote(c.input.string())
             << ",\n  \"mode\": " << json_quote(c.mode)
             << ",\n  \"scenario\": " << json_quote(c.scenario)
             << ",\n  \"strategy\": " << json_quote(c.strategy)
             << ",\n  \"cpu_model\": " << json_quote(cpu_model())
             << ",\n  \"cpu_threads\": " << std::thread::hardware_concurrency()
             << ",\n  \"build_type\": " << json_quote(EVENT_VOXEL_BUILD_TYPE)
             << ",\n  \"git_commit\": " << json_quote(EVENT_VOXEL_GIT_COMMIT)
             << ",\n  \"event_size_bytes\": " << sizeof(Event)
             << ",\n  \"sensor_width\": " << s.sensor_width
             << ",\n  \"sensor_height\": " << s.sensor_height
             << ",\n  \"recording_start_us\": " << s.origin_us
             << ",\n  \"last_window_end_us\": " << s.final_end_us
             << ",\n  \"peak_window_events\": " << source_->peak_window_events()
             << ",\n  \"device_input_capacity_events\": " << s.max_input_capacity
             << ",\n  \"device_output_capacity_voxels\": " << s.max_output_capacity
             << ",\n  \"bins\": " << c.bins << ",\n  \"window_us\": " << c.window_us
             << ",\n  \"max_window_events\": " << c.max_window_events
             << ",\n  \"sdk_slice_queue_capacity\": 1"
             << ",\n  \"offline_real_time_playback\": false"
             << ",\n  \"timestamp_shift\": false"
             << ",\n  \"warmup\": " << c.warmup << ",\n  \"repeat\": " << c.repeat
             << ",\n  \"block_size\": " << c.block_size
             << ",\n  \"profile_index\": " << c.profile_index
             << ",\n  \"host_memory\": \"pageable\""
             << ",\n  \"cuda_streams\": " << (s.max_input_capacity ? 1 : 0)
             << ",\n  \"stream_semantics\": \"nonblocking; synchronized CUDA stop event per stage\""
             << ",\n  \"thrust_internal_allocations_possible\": "
             << (s.max_input_capacity && c.strategy != "event-scatter" ? "true" : "false")
             << ",\n  \"window_origin\": \"sdk_first_cd_slice\""
             << ",\n  \"final_window_policy\": "
                "\"emit_partial_through_last_event_plus_one_microsecond\""
             << ",\n  \"release_time\": \"window_end\""
             << ",\n  \"input_read_in_timing\": false,\n  \"validation_in_timing\": false,\n  "
                "\"serialization_in_timing\": false";
    metadata << ",\n  \"metavision_sdk_version\": " << json_quote(EVENT_VOXEL_METAVISION_VERSION);
#ifdef EVENT_VOXEL_HAS_CUDA
    metadata << ",\n  \"cuda_toolkit_version\": " << json_quote(EVENT_VOXEL_CUDA_TOOLKIT_VERSION);
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
    if (!metadata)
        throw std::runtime_error("Failed to write metadata.json");
}

void Experiment::write_summary() const {
    const auto& s = stats_;
    std::ofstream summary(files_.dir / "summary.json");
    if (!summary)
        throw std::runtime_error("Cannot create summary.json");
    summary << "{\n  \"schema_version\": 2,\n  \"windows\": " << s.windows_seen
            << ",\n  \"event_count_median\": " << quantile(s.event_counts, 0.5)
            << ",\n  \"event_count_p95\": " << quantile(s.event_counts, 0.95)
            << ",\n  \"input_preparation_median_ms\": " << quantile(s.input_preparation_times, 0.5)
            << ",\n  \"input_preparation_p95_ms\": " << quantile(s.input_preparation_times, 0.95)
            << ",\n  \"input_preparation_over_interval_windows\": "
            << s.input_preparation_over_interval
            << ",\n  \"crossover_compared_windows\": " << s.crossover_compared
            << ",\n  \"crossover_gpu_faster_windows\": " << s.crossover_gpu_wins
            << ",\n  \"first_observed_gpu_faster_event_count\": "
            << s.first_gpu_win_events.value_or(0) << ",\n  \"replay\": {";
    bool first_replay = true;
    for (const auto& [scenario, misses] : s.deadline_misses) {
        if (!first_replay)
            summary << ',';
        first_replay = false;
        summary << "\n    " << json_quote(scenario) << ": {\"deadline_misses\": " << misses
                << ", \"max_backlog_ms\": " << s.max_backlog_ms.at(scenario)
                << ", \"total_idle_ms\": " << s.total_idle_ms.at(scenario) << '}';
    }
    summary << "\n  },\n  \"stages\": {\n";
    bool first = true;
    for (const auto& [key, values] : s.stage_durations) {
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
    if (!summary)
        throw std::runtime_error("Failed to write summary.json");
}

} // namespace event_voxel::experiment
