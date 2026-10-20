#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>

#include "event_voxel/cuda/strategy.hpp"
#include "event_voxel/event_window.hpp"
#include "event_voxel/voxelization_spec.hpp"

namespace event_voxel::cuda {

struct StageTiming {
    double cuda_ms = 0.0;
    double wall_ms = 0.0;
};

struct GpuInfo {
    std::string name;
    int compute_major = 0, compute_minor = 0;
    int driver_version = 0, runtime_version = 0;
};

[[nodiscard]] GpuInfo gpu_info();

class CudaExecutionSession {
  public:
    explicit CudaExecutionSession(Strategy strategy, unsigned int block_size = 256);
    ~CudaExecutionSession();
    CudaExecutionSession(const CudaExecutionSession&) = delete;
    CudaExecutionSession& operator=(const CudaExecutionSession&) = delete;

    // Allocations and strategy workspaces are retained until destruction.
    void reserve(std::size_t events, std::size_t width, std::size_t height, std::size_t bins);
    void set_window(EventWindowView window, VoxelizationSpec spec);
    StageTiming upload_events();
    StageTiming upload_host_grid(std::span<const float> grid);
    StageTiming clear();
    StageTiming prepare_kernel();
    StageTiming run_kernel();
    StageTiming run_strategy(); // clear + all preprocessing + final kernel
    StageTiming download_grid(std::span<float> grid);
    [[nodiscard]] const float* device_grid() const noexcept;
    [[nodiscard]] std::size_t input_capacity() const noexcept;
    [[nodiscard]] std::size_t output_capacity() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace event_voxel::cuda
