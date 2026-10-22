#include "experiment.hpp"

#include "event_voxel/io/npy_writer.hpp"
#include "event_voxel/voxel_grid_spec.hpp"
#ifdef EVENT_VOXEL_NVTX
#include <nvtx3/nvtx3.hpp>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace event_voxel::experiment {
namespace {
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

bool is_gpu_stage_mode(std::string_view mode) {
    return mode == "kernel" || mode == "strategy" || mode == "transfer";
}

bool is_pipeline_mode(std::string_view mode) {
    return mode == "pipeline" || mode == "replay" || mode == "profile";
}

bool is_selected_window_mode(std::string_view mode) {
    return mode == "validate" || mode == "export";
}

std::vector<std::string> scenarios(const Config& c) {
    if (c.scenario == "all")
        return {"cpu-host", "cpu-to-device", "gpu-device", "gpu-to-host"};
    if (c.scenario == "auto") {
#ifdef EVENT_VOXEL_HAS_CUDA
        if (cuda::is_available()) {
            if (c.mode == "profile")
                return {"gpu-device"};
            return {"cpu-host", "cpu-to-device", "gpu-device", "gpu-to-host"};
        }
#endif
        return {"cpu-host"};
    }
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

Experiment::Experiment(Config config)
    : config_(std::move(config)), source_(open_source(config_)), files_(config_.results_root),
      spec_{config_.bins} {}

void Experiment::run() {
    EventWindowView window;
    for (std::size_t index = 0; index < config_.max_windows; ++index) {
        const auto preparation_start = Clock::now();
        bool has_window = false;
        {
#ifdef EVENT_VOXEL_NVTX
            nvtx3::scoped_range marker("input decode and window preparation");
#endif
            has_window = source_->next(window);
        }
        if (!has_window)
            break;
        record_window(window, elapsed_ms(preparation_start));

        if (config_.mode == "profile" && window.sequence != config_.profile_index)
            continue;
        if (config_.mode == "profile")
            stats_.profile_found = true;
        if (is_selected_window_mode(config_.mode)) {
            if (window.sequence != config_.export_index)
                continue;
            run_selected_window(window);
            break;
        }
        measure_window(window);
        if (config_.mode == "profile")
            break;
    }

    if (stats_.windows_seen == 0)
        throw std::runtime_error("Input contained no CD windows");
    if (is_selected_window_mode(config_.mode) && !stats_.selected_found)
        throw std::runtime_error("Selected export/validation window was not found");
    if (config_.mode == "profile" && !stats_.profile_found)
        throw std::runtime_error("Selected profile window was not found");
    write_metadata();
    write_summary();
    std::cout << "Wrote " << files_.dir << " (" << stats_.windows_seen << " windows)\n";
}

void Experiment::record_window(const EventWindowView& window, double preparation_ms) {
    ++stats_.windows_seen;
    stats_.event_counts.push_back(static_cast<double>(window.events.size()));
    stats_.input_preparation_times.push_back(preparation_ms);
    if (preparation_ms > double(config_.window_us) / 1000.0)
        ++stats_.input_preparation_over_interval;
    if (stats_.windows_seen == 1)
        stats_.origin_us = window.start_us;
    stats_.final_end_us = window.end_us;
    stats_.sensor_width = window.width;
    stats_.sensor_height = window.height;
    files_.windows << window.sequence << ',' << window.start_us << ',' << window.end_us << ','
                   << window.events.size() << ','
                   << (1'000'000.0 * window.events.size() / (window.end_us - window.start_us))
                   << ',' << window.partial_final << ',' << preparation_ms << '\n';
}

void Experiment::record_sample(const EventWindowView& window, std::size_t rep,
                               const std::string& scenario, const std::string& stage,
                               double wall_ms, double cuda_ms, std::size_t bytes) {
    files_.samples << window.sequence << ',' << rep << ',' << scenario << ',' << stage << ','
                   << std::setprecision(12) << wall_ms << ',' << cuda_ms << ',' << bytes << '\n';
    stats_.stage_durations[scenario + "/" + stage].push_back(wall_ms);
}

void Experiment::run_selected_window(const EventWindowView& window) {
    stats_.selected_found = true;
    const auto reference = cpu_.voxelize(window, spec_);
    if (config_.mode == "validate")
        validate_cpu_conservation(window, reference);
#ifdef EVENT_VOXEL_HAS_CUDA
    if (config_.mode == "validate" && cuda::is_available()) {
        for (const auto strategy : {cuda::Strategy::EventScatter, cuda::Strategy::PixelGather,
                                    cuda::Strategy::SortReduce}) {
            cuda::CudaExecutionSession gpu(strategy, config_.block_size);
            gpu.reserve(window.events.size(), window.width, window.height, config_.bins);
            gpu.set_window(window, spec_);
            gpu.upload_events();
            gpu.run_strategy();
            std::vector<float> actual(reference.data().size());
            gpu.download_grid(actual);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const double tolerance =
                    1e-4 + 1e-5 * std::max(1.0, std::abs(double(reference.data()[i])));
                if (std::abs(double(actual[i]) - reference.data()[i]) > tolerance)
                    throw std::runtime_error("GPU validation failed at voxel " + std::to_string(i));
            }
        }
    }
#endif
    if (config_.mode == "export") {
        fs::create_directories(files_.dir / "outputs");
        io::write_npy(reference, files_.dir / "outputs" / "cpu.npy");
#ifdef EVENT_VOXEL_HAS_CUDA
        if (cuda::is_available()) {
            cuda::CudaExecutionSession gpu(selected_strategy(config_.strategy), config_.block_size);
            gpu.reserve(window.events.size(), window.width, window.height, config_.bins);
            gpu.set_window(window, spec_);
            gpu.upload_events();
            gpu.run_strategy();
            VoxelGrid gpu_output(reference.spec());
            gpu.download_grid(gpu_output.data());
            io::write_npy(gpu_output,
                          files_.dir / "outputs" / ("gpu-" + config_.strategy + ".npy"));
        }
#endif
    }
    std::cout << config_.mode << " completed for window " << window.sequence << '\n';
}

#ifdef EVENT_VOXEL_HAS_CUDA
cuda::CudaExecutionSession* Experiment::session_for(const std::string& scenario,
                                                    const EventWindowView& window) {
    if (!cuda::is_available())
        throw std::runtime_error("CUDA device unavailable");
    auto& session = gpu_sessions_[scenario];
    if (!session)
        session = std::make_unique<cuda::CudaExecutionSession>(selected_strategy(config_.strategy),
                                                               config_.block_size);
    session->reserve(window.events.size(), window.width, window.height, config_.bins);
    session->set_window(window, spec_);
    stats_.max_input_capacity = std::max(stats_.max_input_capacity, session->input_capacity());
    stats_.max_output_capacity = std::max(stats_.max_output_capacity, session->output_capacity());
    return session.get();
}
#endif

void Experiment::measure_window(const EventWindowView& window) {
    std::map<std::string, double> window_medians;
    for (const auto& scenario : scenarios(config_)) {
        const auto median = measure_scenario(window, scenario);
        if (is_pipeline_mode(config_.mode))
            window_medians[scenario] = median;
        if (config_.mode == "replay")
            record_replay(window, scenario, median);
    }
    record_crossover(window, window_medians);
}

double Experiment::measure_scenario(const EventWindowView& window, const std::string& scenario) {
    const bool cpu_path = scenario == "cpu-host" || scenario == "cpu-to-device";
    const bool needs_cuda =
        !cpu_path || scenario == "cpu-to-device" || is_gpu_stage_mode(config_.mode);
#ifndef EVENT_VOXEL_HAS_CUDA
    if (needs_cuda)
        throw std::runtime_error("This scenario requires a CUDA build");
#endif
    const VoxelGridSpec grid_spec{window.width, window.height, config_.bins};
    // Reserve buffers and prepare the ready window before any timed repetition.
    ScenarioRun run{scenario, cpu_path, grid_spec.memory_bytes(), {}};
#ifdef EVENT_VOXEL_HAS_CUDA
    if (needs_cuda)
        run.gpu = session_for(scenario, window);
#endif
    run.host_grid.resize(grid_spec.element_count());
#ifdef EVENT_VOXEL_HAS_CUDA
    if (config_.mode == "kernel" || config_.mode == "strategy")
        run.gpu->upload_events();
#endif
    for (std::size_t i = 0; i < config_.warmup; ++i)
        run_iteration(window, run, i, false);
    std::vector<double> samples;
    samples.reserve(config_.repeat);
    for (std::size_t i = 0; i < config_.repeat; ++i)
        samples.push_back(run_iteration(window, run, i, true));
    return quantile(samples, 0.5);
}

double Experiment::run_iteration(const EventWindowView& window, ScenarioRun& run, std::size_t rep,
                                 bool record) {
    struct PendingSample {
        const char* stage;
        double wall_ms;
        double cuda_ms;
        std::size_t bytes;
    };
    // Each path emits at most three stage samples. Keep recording out of the timed interval.
    std::array<PendingSample, 3> pending{};
    std::size_t pending_count = 0;
    const auto add_sample = [&](const char* stage, double wall_ms, double cuda_ms = 0.0,
                                std::size_t bytes = 0) {
        if (record)
            pending[pending_count++] = {stage, wall_ms, cuda_ms, bytes};
    };
    const auto pipeline_start = Clock::now();
    const auto finish = [&](bool pipeline_mode) {
        const double wall_ms = elapsed_ms(pipeline_start);
        if (record) {
            for (std::size_t i = 0; i < pending_count; ++i) {
                const auto& sample = pending[i];
                record_sample(window, rep, run.name, sample.stage, sample.wall_ms, sample.cuda_ms,
                              sample.bytes);
            }
            if (pipeline_mode)
                record_sample(window, rep, run.name, "pipeline", wall_ms);
        }
        return wall_ms;
    };
    if (is_gpu_stage_mode(config_.mode)) {
#ifdef EVENT_VOXEL_HAS_CUDA
        if (config_.mode == "transfer") {
            const auto h2d = run.gpu->upload_events();
            add_sample("h2d-events", h2d.wall_ms, h2d.cuda_ms, window.events.size_bytes());
            const auto reference = cpu_.voxelize_prevalidated(window, spec_);
            const auto vox_h2d = run.gpu->upload_host_grid(reference.data());
            add_sample("h2d-cpu-grid", vox_h2d.wall_ms, vox_h2d.cuda_ms, run.grid_bytes);
            const auto d2h = run.gpu->download_grid(run.host_grid);
            add_sample("d2h-grid", d2h.wall_ms, d2h.cuda_ms, run.grid_bytes);
        } else if (config_.mode == "strategy") {
            const auto time = run.gpu->run_strategy();
            add_sample("strategy", time.wall_ms, time.cuda_ms);
        } else {
            run.gpu->clear();
            run.gpu->prepare_kernel();
            const auto time = run.gpu->run_kernel();
            add_sample("kernel", time.wall_ms, time.cuda_ms);
        }
#endif
        return finish(false);
    }
    if (run.cpu_path) {
        const auto start = Clock::now();
        auto output = cpu_.voxelize_prevalidated(window, spec_);
        add_sample("cpu-voxelize", elapsed_ms(start));
#ifdef EVENT_VOXEL_HAS_CUDA
        if (run.name == "cpu-to-device") {
            const auto time = run.gpu->upload_host_grid(output.data());
            add_sample("h2d-cpu-grid", time.wall_ms, time.cuda_ms, run.grid_bytes);
        }
#endif
    } else {
#ifdef EVENT_VOXEL_HAS_CUDA
        const auto h2d = run.gpu->upload_events();
        add_sample("h2d-events", h2d.wall_ms, h2d.cuda_ms, window.events.size_bytes());
        const auto strategy = run.gpu->run_strategy();
        add_sample("strategy", strategy.wall_ms, strategy.cuda_ms);
        if (run.name == "gpu-to-host") {
            const auto d2h = run.gpu->download_grid(run.host_grid);
            add_sample("d2h-grid", d2h.wall_ms, d2h.cuda_ms, run.grid_bytes);
        }
#endif
    }
    return finish(true);
}

void Experiment::record_replay(const EventWindowView& window, const std::string& scenario,
                               double work_ms) {
    // Simulate one sequential worker, releasing each window at its interval end.
    const double release_ms = double(window.end_us - stats_.origin_us) / 1000.0;
    const double start_ms = std::max(release_ms, stats_.replay_finish_ms[scenario]);
    stats_.total_idle_ms[scenario] += std::max(0.0, release_ms - stats_.replay_finish_ms[scenario]);
    stats_.max_backlog_ms[scenario] =
        std::max(stats_.max_backlog_ms[scenario], start_ms - release_ms);
    stats_.replay_finish_ms[scenario] = start_ms + work_ms;
    auto& misses = stats_.deadline_misses[scenario];
    if (stats_.replay_finish_ms[scenario] > release_ms + double(config_.window_us) / 1000.0)
        ++misses;
    record_sample(window, 0, scenario, "replay-backlog", start_ms - release_ms);
    record_sample(window, 0, scenario, "replay-end-to-result",
                  stats_.replay_finish_ms[scenario] - release_ms);
}

void Experiment::record_crossover(const EventWindowView& window,
                                  const std::map<std::string, double>& medians) {
    if (!medians.contains("cpu-to-device") || !medians.contains("gpu-device"))
        return;
    const bool gpu_faster = medians.at("gpu-device") < medians.at("cpu-to-device");
    files_.crossovers << window.sequence << ',' << config_.window_us << ',' << window.events.size()
                      << ',' << medians.at("cpu-to-device") << ',' << medians.at("gpu-device")
                      << ',' << gpu_faster << '\n';
    ++stats_.crossover_compared;
    if (gpu_faster) {
        ++stats_.crossover_gpu_wins;
        if (!stats_.first_gpu_win_events)
            stats_.first_gpu_win_events = window.events.size();
    }
}

} // namespace event_voxel::experiment
