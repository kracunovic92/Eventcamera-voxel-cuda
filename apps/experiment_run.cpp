#include "experiment.hpp"

#include "event_voxel/data/metavision_hdf5.hpp"
#ifdef EVENT_VOXEL_HAS_CUDA
#include "event_voxel/voxel_grid_spec.hpp"
#include <nvtx3/nvtx3.hpp>
#endif

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace event_voxel::experiment {
namespace {
using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
} // namespace

Experiment::Experiment(Config config)
    : config_(std::move(config)),
      source_(data::open_metavision_hdf5(config_.input, config_.window_us)), files_(config_),
      spec_{config_.bins} {}

void Experiment::run() {
    const auto run_start = Clock::now();
#ifdef EVENT_VOXEL_HAS_CUDA
    if (!cuda::is_available())
        throw std::runtime_error("CUDA build requires an available CUDA device");
    gpu_session_ =
        std::make_unique<cuda::CudaExecutionSession>(config_.strategy, config_.block_size);
#endif
    EventWindowView window{};
    while (true) {
        const auto preparation_start = Clock::now();
        if (!read_next_window(window))
            break;
        record_input_window(window, elapsed_ms(preparation_start));
        process_window(window);
    }
    if (stats_.windows == 0)
        throw std::runtime_error("Input contained no CD windows");
    stats_.run_wall_ms = elapsed_ms(run_start);
    write_metadata();
    write_summary();
    std::cout << "Wrote " << files_.dir << " (" << stats_.windows << " windows processed)\n";
}

bool Experiment::read_next_window(EventWindowView& window) {
#ifdef EVENT_VOXEL_HAS_CUDA
    nvtx3::scoped_range marker("read next CD window");
#endif
    return source_->next(window);
}

void Experiment::process_window(const EventWindowView& window) {
#ifdef EVENT_VOXEL_HAS_CUDA
    const auto backend = cuda::to_string(config_.strategy);
    const auto processing_start = Clock::now();
    gpu_session_->reserve(window.events.size(), window.width, window.height, spec_.bins);
    gpu_session_->set_window(window, spec_);
    host_gpu_grid_.resize(VoxelGridSpec{window.width, window.height, spec_.bins}.element_count());
    const double prepare_ms = elapsed_ms(processing_start);

    const auto pipeline_start = Clock::now();
    const auto upload = gpu_session_->upload_events();
    const auto strategy = gpu_session_->run_strategy();
    const double device_ready_ms = elapsed_ms(pipeline_start);
    const auto download = gpu_session_->download_grid(host_gpu_grid_);
    const double pipeline_ms = elapsed_ms(pipeline_start);
    const double processing_total_ms = elapsed_ms(processing_start);

    // CSV writes happen after the timed pipeline, so they cannot inflate its wall time.
    stats_.max_device_input_capacity_events =
        std::max(stats_.max_device_input_capacity_events, gpu_session_->input_capacity());
    stats_.max_device_output_capacity_voxels =
        std::max(stats_.max_device_output_capacity_voxels, gpu_session_->output_capacity());
    record_sample(window, backend, "prepare", prepare_ms);
    record_sample(window, backend, "h2d-events", upload.wall_ms, upload.cuda_ms,
                  window.events.size_bytes());
    record_sample(window, backend, "strategy", strategy.wall_ms, strategy.cuda_ms);
    record_sample(window, backend, "device-ready", device_ready_ms);
    record_sample(window, backend, "d2h-grid", download.wall_ms, download.cuda_ms,
                  host_gpu_grid_.size() * sizeof(float));
    record_sample(window, backend, "pipeline", pipeline_ms);
    record_sample(window, backend, "processing-total", processing_total_ms);
#else
    const auto start = Clock::now();
    [[maybe_unused]] auto grid = cpu_.voxelize_prevalidated(window, spec_);
    const double voxelize_ms = elapsed_ms(start);
    record_sample(window, "cpu", "voxelize", voxelize_ms);
    record_sample(window, "cpu", "pipeline", voxelize_ms);
    record_sample(window, "cpu", "processing-total", voxelize_ms);
#endif
}

} // namespace event_voxel::experiment
