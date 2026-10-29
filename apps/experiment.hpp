#pragma once

#include "event_voxel/data/time_slice_source.hpp"
#include "event_voxel/voxelization_spec.hpp"
#ifdef EVENT_VOXEL_HAS_CUDA
#include "event_voxel/cuda/execution_session.hpp"
#else
#include "event_voxel/cpu/reference_voxelizer.hpp"
#endif

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace event_voxel::experiment {

struct Config {
    std::filesystem::path input;
    std::filesystem::path results_root = "results";
    std::int64_t window_us = 100'000;
    std::size_t bins = 5;
#ifdef EVENT_VOXEL_HAS_CUDA
    cuda::Strategy strategy = cuda::Strategy::EventScatter;
    unsigned int block_size = 256;
#endif
};

Config parse_config(int argc, char** argv);
double nearest_rank_quantile(std::vector<double> values, double probability);

struct ResultFiles {
    explicit ResultFiles(const Config& config);

    std::filesystem::path dir;
    std::ofstream windows;
    std::ofstream samples;
};

struct RunStatistics {
    std::map<std::string, std::vector<double>> stage_wall_times_ms;
    std::vector<double> window_event_counts;
    std::vector<double> input_preparation_times_ms;
    std::size_t windows = 0;
    std::size_t total_events = 0;
    std::size_t input_preparation_over_interval_windows = 0;
    double run_wall_ms = 0;
    std::int64_t first_window_start_us = 0;
    std::int64_t last_window_end_us = 0;
    std::uint32_t sensor_width = 0;
    std::uint32_t sensor_height = 0;
#ifdef EVENT_VOXEL_HAS_CUDA
    std::size_t max_device_input_capacity_events = 0;
    std::size_t max_device_output_capacity_voxels = 0;
#endif
};

class Experiment {
  public:
    explicit Experiment(Config config);
    void run();

  private:
    bool read_next_window(EventWindowView& window);
    void record_input_window(const EventWindowView& window, double preparation_ms);
    void process_window(const EventWindowView& window);
    void record_sample(const EventWindowView& window, std::string_view backend,
                       std::string_view stage, double wall_ms,
                       std::optional<double> cuda_event_ms = std::nullopt, std::size_t bytes = 0);
    void write_metadata() const;
    void write_summary() const;

    Config config_;
    std::unique_ptr<data::TimeSliceSource> source_;
    ResultFiles files_;
    VoxelizationSpec spec_;
    RunStatistics stats_;
#ifdef EVENT_VOXEL_HAS_CUDA
    std::unique_ptr<cuda::CudaExecutionSession> gpu_session_;
    std::vector<float> host_gpu_grid_;
#else
    CpuReferenceVoxelizer cpu_;
#endif
};

} // namespace event_voxel::experiment
