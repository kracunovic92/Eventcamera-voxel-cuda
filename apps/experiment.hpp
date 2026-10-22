#pragma once

#include "event_voxel/cpu/reference_voxelizer.hpp"
#include "event_voxel/data/time_slice_source.hpp"
#include "event_voxel/voxelization_spec.hpp"
#ifdef EVENT_VOXEL_HAS_CUDA
#include "event_voxel/cuda/execution_session.hpp"
#endif

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace event_voxel::experiment {

struct Config {
    std::string mode = "pipeline", scenario = "auto";
    std::string strategy = "event-scatter";
    std::filesystem::path input, results_root = "results";
    std::size_t bins = 5, warmup = 3, repeat = 10, max_windows = 100;
    std::size_t max_window_events = 1'000'000;
    std::size_t export_index = 0, profile_index = 0;
    std::int64_t window_us = 100'000;
    unsigned int block_size = 256;
};

Config parse(int argc, char** argv);
std::unique_ptr<data::TimeSliceSource> open_source(const Config& config);
double quantile(std::vector<double> values, double q);

struct ResultFiles {
    explicit ResultFiles(const std::filesystem::path& root);

    std::filesystem::path dir;
    std::ofstream windows;
    std::ofstream samples;
    std::ofstream crossovers;
};

struct RunStatistics {
    std::map<std::string, std::vector<double>> stage_durations;
    std::vector<double> event_counts;
    std::vector<double> input_preparation_times;
    std::size_t input_preparation_over_interval = 0;
    std::size_t windows_seen = 0;
    std::size_t crossover_gpu_wins = 0;
    std::size_t crossover_compared = 0;
    std::optional<std::size_t> first_gpu_win_events;
    std::size_t max_input_capacity = 0;
    std::size_t max_output_capacity = 0;
    std::map<std::string, double> replay_finish_ms;
    std::map<std::string, double> max_backlog_ms;
    std::map<std::string, double> total_idle_ms;
    std::map<std::string, std::size_t> deadline_misses;
    std::int64_t origin_us = 0;
    std::int64_t final_end_us = 0;
    std::uint32_t sensor_width = 0;
    std::uint32_t sensor_height = 0;
    bool selected_found = false;
    bool profile_found = false;
};

struct ScenarioRun {
    std::string name;
    bool cpu_path;
    std::size_t grid_bytes;
    std::vector<float> host_grid;
#ifdef EVENT_VOXEL_HAS_CUDA
    cuda::CudaExecutionSession* gpu = nullptr;
#endif
};

class Experiment {
  public:
    explicit Experiment(Config config);
    void run();

  private:
    void record_window(const EventWindowView& window, double preparation_ms);
    void record_sample(const EventWindowView& window, std::size_t rep, const std::string& scenario,
                       const std::string& stage, double wall_ms, double cuda_ms = 0,
                       std::size_t bytes = 0);
    void run_selected_window(const EventWindowView& window);
#ifdef EVENT_VOXEL_HAS_CUDA
    cuda::CudaExecutionSession* session_for(const std::string& scenario,
                                            const EventWindowView& window);
#endif
    void measure_window(const EventWindowView& window);
    double measure_scenario(const EventWindowView& window, const std::string& scenario);
    double run_iteration(const EventWindowView& window, ScenarioRun& run, std::size_t rep,
                         bool record);
    void record_replay(const EventWindowView& window, const std::string& scenario, double work_ms);
    void record_crossover(const EventWindowView& window,
                          const std::map<std::string, double>& medians);
    void write_metadata() const;
    void write_summary() const;

    Config config_;
    std::unique_ptr<data::TimeSliceSource> source_;
    ResultFiles files_;
    CpuReferenceVoxelizer cpu_;
    VoxelizationSpec spec_;
    RunStatistics stats_;
#ifdef EVENT_VOXEL_HAS_CUDA
    std::map<std::string, std::unique_ptr<cuda::CudaExecutionSession>> gpu_sessions_;
#endif
};

} // namespace event_voxel::experiment
