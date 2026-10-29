#include "experiment.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>

namespace event_voxel::experiment {
namespace {
using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;
namespace fs = std::filesystem;

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

std::string run_label(const Config& config) {
#ifdef EVENT_VOXEL_HAS_CUDA
    return "cuda-" + std::string(cuda::to_string(config.strategy)) + "-w" +
           std::to_string(config.window_us) + "us-b" + std::to_string(config.bins) + "-t" +
           std::to_string(config.block_size);
#else
    return "cpu-reference-w" + std::to_string(config.window_us) + "us-b" +
           std::to_string(config.bins);
#endif
}

fs::path make_run_directory(const fs::path& root, std::string_view label) {
    const auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
    const auto uniqueness = Clock::now().time_since_epoch().count();
    const fs::path dir = root / ("run-" + std::string(label) + "-" + std::to_string(stamp) + "-" +
                                 std::to_string(uniqueness));
    if (!fs::create_directories(dir))
        throw std::runtime_error("Results directory already exists");
    return dir;
}

void write_json(const fs::path& path, const Json& document) {
    std::ofstream output(path);
    if (!output)
        throw std::runtime_error("Cannot create " + path.string());
    output << document.dump(2) << '\n';
    output.flush();
    if (!output)
        throw std::runtime_error("Failed to write " + path.string());
}

Json describe_stage(const std::vector<double>& values) {
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double squared_difference = 0;
    for (const double value : values)
        squared_difference += (value - mean) * (value - mean);
    return {{"count", values.size()},
            {"mean_ms", mean},
            {"median_ms", nearest_rank_quantile(values, 0.5)},
            {"p95_ms", nearest_rank_quantile(values, 0.95)},
            {"std_ms", std::sqrt(squared_difference / values.size())}};
}

} // namespace

double nearest_rank_quantile(std::vector<double> values, double probability) {
    if (values.empty())
        return 0;
    std::sort(values.begin(), values.end());
    return values[std::min(values.size() - 1,
                           static_cast<std::size_t>(std::ceil(probability * values.size()) - 1))];
}

ResultFiles::ResultFiles(const Config& config)
    : dir(make_run_directory(config.results_root, run_label(config))), windows(dir / "windows.csv"),
      samples(dir / "samples.csv") {
    if (!windows || !samples)
        throw std::runtime_error("Cannot create result files");
    windows << "sequence,start_us,end_us,event_count,rate_per_second,partial_final,"
               "input_preparation_ms\n";
    samples << "sequence,backend,stage,wall_ms,cuda_event_ms,bytes\n";
}

void Experiment::write_metadata() const {
    const auto& config = config_;
    const auto& stats = stats_;
    Json metadata = {
        {"schema_version", 4},
        {"run_id", files_.dir.filename().string()},
        {"source", "hdf5"},
        {"recording", config.input.string()},
        {"recording_size_bytes", fs::file_size(config.input)},
        {"cpu_model", cpu_model()},
        {"cpu_threads", std::thread::hardware_concurrency()},
        {"build_type", EVENT_VOXEL_BUILD_TYPE},
        {"git_commit", EVENT_VOXEL_GIT_COMMIT},
        {"metavision_sdk_version", EVENT_VOXEL_METAVISION_VERSION},
        {"event_size_bytes", sizeof(Event)},
        {"sensor_width", stats.sensor_width},
        {"sensor_height", stats.sensor_height},
        {"recording_start_us", stats.first_window_start_us},
        {"last_window_end_us", stats.last_window_end_us},
        {"peak_window_events", source_->peak_window_events()},
        {"bins", config.bins},
        {"window_us", config.window_us},
        {"sdk_slice_queue_capacity", 1},
        {"offline_real_time_playback", false},
        {"timestamp_shift", false},
        {"host_memory", "pageable"},
        {"output_location", "host"},
        {"window_origin", "sdk_first_cd_slice"},
        {"final_window_policy", "emit_partial_through_last_event_plus_one_microsecond"},
        {"input_preparation_definition", "wall time waiting for and converting the next SDK slice"},
        {"run_wall_definition", "run() entry through EOF; includes processing and CSV, excludes "
                                "source opening and final JSON writing"},
    };
#ifdef EVENT_VOXEL_HAS_CUDA
    const auto gpu = cuda::gpu_info();
    metadata["backend"] = std::string(cuda::to_string(config.strategy));
    metadata["strategy"] = std::string(cuda::to_string(config.strategy));
    metadata["block_size"] = config.block_size;
    metadata["cuda_toolkit_version"] = EVENT_VOXEL_CUDA_TOOLKIT_VERSION;
    metadata["gpu_name"] = gpu.name;
    metadata["gpu_compute_capability"] =
        std::to_string(gpu.compute_major) + "." + std::to_string(gpu.compute_minor);
    metadata["cuda_driver_version"] = gpu.driver_version;
    metadata["cuda_runtime_version"] = gpu.runtime_version;
    metadata["device_input_capacity_events"] = stats.max_device_input_capacity_events;
    metadata["device_output_capacity_voxels"] = stats.max_device_output_capacity_voxels;
    metadata["cuda_streams"] = 1;
    metadata["stream_semantics"] = "nonblocking; synchronized CUDA stop event per stage";
    metadata["pipeline_definition"] =
        "H2D events, strategy, D2H grid; excludes input, buffer preparation and CSV";
    metadata["device_ready_definition"] =
        "H2D events and strategy; excludes input, buffer preparation and D2H";
    metadata["processing_total_definition"] =
        "buffer preparation, H2D events, strategy, D2H grid; excludes input and CSV";
#else
    metadata["backend"] = "cpu";
    metadata["pipeline_definition"] = "CPU voxelization to a host grid; excludes input and CSV";
    metadata["processing_total_definition"] = "same timed interval as CPU pipeline";
#endif
    write_json(files_.dir / "metadata.json", metadata);
}

void Experiment::write_summary() const {
    const auto& stats = stats_;
    const double seconds = stats.run_wall_ms / 1000.0;
    Json stages = Json::object();
    for (const auto& [stage, values] : stats.stage_wall_times_ms)
        stages[stage] = describe_stage(values);

    const Json summary = {
        {"schema_version", 4},
        {"windows", stats.windows},
        {"total_events", stats.total_events},
        {"run_wall_ms", stats.run_wall_ms},
        {"events_per_second", seconds > 0 ? stats.total_events / seconds : 0.0},
        {"windows_per_second", seconds > 0 ? stats.windows / seconds : 0.0},
        {"event_count_median", nearest_rank_quantile(stats.window_event_counts, 0.5)},
        {"event_count_p95", nearest_rank_quantile(stats.window_event_counts, 0.95)},
        {"input_preparation_median_ms",
         nearest_rank_quantile(stats.input_preparation_times_ms, 0.5)},
        {"input_preparation_p95_ms", nearest_rank_quantile(stats.input_preparation_times_ms, 0.95)},
        {"input_preparation_over_interval_windows", stats.input_preparation_over_interval_windows},
        {"stages", std::move(stages)},
    };
    write_json(files_.dir / "summary.json", summary);
}

} // namespace event_voxel::experiment
