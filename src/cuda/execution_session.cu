#include "event_voxel/cuda/execution_session.hpp"

#include "event_voxel/validation.hpp"
#include "event_voxel/voxel_grid_spec.hpp"
#include "shared/cuda_check.cuh"
#include "shared/device_buffer.cuh"
#include "shared/strategy_launchers.cuh"

#include <cuda_runtime.h>

#include <chrono>
#include <functional>
#include <stdexcept>
#ifdef EVENT_VOXEL_NVTX
#include <nvtx3/nvtx3.hpp>
#endif

namespace event_voxel::cuda {

GpuInfo gpu_info() {
    cudaDeviceProp properties{};
    shared::check_cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
    GpuInfo info;
    info.name = properties.name;
    info.compute_major = properties.major;
    info.compute_minor = properties.minor;
    shared::check_cuda(cudaDriverGetVersion(&info.driver_version), "cudaDriverGetVersion");
    shared::check_cuda(cudaRuntimeGetVersion(&info.runtime_version), "cudaRuntimeGetVersion");
    return info;
}

struct CudaExecutionSession::Impl {
    explicit Impl(Strategy selected, unsigned int threads)
        : strategy(selected), block_size(threads) {
        if (threads != 128 && threads != 256 && threads != 512) {
            throw std::invalid_argument("CUDA block size must be 128, 256 or 512");
        }
        shared::check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                           "cudaStreamCreate");
        try {
            shared::check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
            shared::check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
        } catch (...) {
            if (start)
                cudaEventDestroy(start);
            cudaStreamDestroy(stream);
            throw;
        }
    }
    ~Impl() {
        if (stop)
            cudaEventDestroy(stop);
        if (start)
            cudaEventDestroy(start);
        if (stream)
            cudaStreamDestroy(stream);
    }
    StageTiming measure(const std::function<void()>& action) {
        using Clock = std::chrono::steady_clock;
        const auto wall_start = Clock::now();
        shared::check_cuda(cudaEventRecord(start, stream), "cudaEventRecord(start)");
        action();
        shared::check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord(stop)");
        shared::check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
        const auto wall_end = Clock::now();
        float gpu_ms = 0;
        shared::check_cuda(cudaEventElapsedTime(&gpu_ms, start, stop), "cudaEventElapsedTime");
        return {gpu_ms, std::chrono::duration<double, std::milli>(wall_end - wall_start).count()};
    }
    void prepare() {
#ifdef EVENT_VOXEL_NVTX
        nvtx3::scoped_range marker("strategy preprocessing");
#endif
        switch (strategy) {
        case Strategy::EventScatter:
            break;
        case Strategy::PixelGather:
            shared::prepare_pixel_gather(events.data(), params, pixel, stream, block_size);
            break;
        case Strategy::SortReduce:
            shared::prepare_sort_reduce(events.data(), params, sort, stream, block_size);
            break;
        }
    }
    void kernel() {
#ifdef EVENT_VOXEL_NVTX
        nvtx3::scoped_range marker("strategy final kernel");
#endif
        switch (strategy) {
        case Strategy::EventScatter:
            shared::run_event_scatter(events.data(), voxels.data(), params, stream, block_size);
            break;
        case Strategy::PixelGather:
            shared::finish_pixel_gather(voxels.data(), params, pixel, stream, block_size);
            break;
        case Strategy::SortReduce:
            shared::finish_sort_reduce(voxels.data(), params, sort, stream, block_size);
            break;
        }
    }
    Strategy strategy;
    unsigned int block_size;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    shared::DeviceBuffer<Event> events;
    shared::DeviceBuffer<float> voxels;
    shared::PixelGatherWorkspace pixel;
    shared::SortReduceWorkspace sort;
    shared::VoxelizationParams params{};
    EventWindowView window{};
};

CudaExecutionSession::CudaExecutionSession(Strategy strategy, unsigned int block_size)
    : impl_(std::make_unique<Impl>(strategy, block_size)) {}
CudaExecutionSession::~CudaExecutionSession() = default;

void CudaExecutionSession::reserve(std::size_t count, std::size_t width, std::size_t height,
                                   std::size_t bins) {
    const VoxelGridSpec spec{width, height, bins};
    if (width == 0 || height == 0 || bins < 2)
        throw std::invalid_argument("Invalid grid shape");
    const auto voxels = spec.element_count();
    const auto checked_bytes = spec.memory_bytes();
    (void)checked_bytes;
    impl_->events.reserve(count, "cudaMalloc(events)");
    impl_->voxels.reserve(voxels, "cudaMalloc(voxels)");
    if (impl_->strategy == Strategy::PixelGather)
        impl_->pixel.reserve(count, width * height);
    if (impl_->strategy == Strategy::SortReduce)
        impl_->sort.reserve(count);
}

void CudaExecutionSession::set_window(EventWindowView window, VoxelizationSpec spec) {
    validate_window(window, spec);
    if (window.events.size() > impl_->events.size() ||
        VoxelGridSpec{window.width, window.height, spec.bins}.element_count() >
            impl_->voxels.size()) {
        throw std::runtime_error("Call reserve() for the window before set_window()");
    }
    impl_->window = window;
    impl_->params = {window.events.size(),
                     window.width,
                     window.height,
                     spec.bins,
                     VoxelGridSpec{window.width, window.height, spec.bins}.element_count(),
                     window.start_us,
                     static_cast<double>(spec.bins - 1) /
                         static_cast<double>(window.end_us - window.start_us)};
}

StageTiming CudaExecutionSession::upload_events() {
#ifdef EVENT_VOXEL_NVTX
    nvtx3::scoped_range marker("H2D events");
#endif
    return impl_->measure([this] {
        const auto bytes = impl_->window.events.size_bytes();
        if (bytes)
            shared::check_cuda(cudaMemcpyAsync(impl_->events.data(), impl_->window.events.data(),
                                               bytes, cudaMemcpyHostToDevice, impl_->stream),
                               "H2D(events)");
    });
}
StageTiming CudaExecutionSession::upload_host_grid(std::span<const float> grid) {
#ifdef EVENT_VOXEL_NVTX
    nvtx3::scoped_range marker("H2D CPU voxels");
#endif
    if (grid.size() != impl_->params.voxel_count)
        throw std::invalid_argument("Grid size mismatch");
    return impl_->measure([this, grid] {
        shared::check_cuda(cudaMemcpyAsync(impl_->voxels.data(), grid.data(), grid.size_bytes(),
                                           cudaMemcpyHostToDevice, impl_->stream),
                           "H2D(CPU grid)");
    });
}
StageTiming CudaExecutionSession::clear() {
#ifdef EVENT_VOXEL_NVTX
    nvtx3::scoped_range marker("output clear");
#endif
    return impl_->measure([this] {
        shared::check_cuda(cudaMemsetAsync(impl_->voxels.data(), 0,
                                           impl_->params.voxel_count * sizeof(float),
                                           impl_->stream),
                           "cudaMemset(voxels)");
    });
}
StageTiming CudaExecutionSession::prepare_kernel() {
    return impl_->measure([this] { impl_->prepare(); });
}
StageTiming CudaExecutionSession::run_kernel() {
    return impl_->measure([this] { impl_->kernel(); });
}
StageTiming CudaExecutionSession::run_strategy() {
    return impl_->measure([this] {
#ifdef EVENT_VOXEL_NVTX
        {
            nvtx3::scoped_range marker("output clear");
#endif
            shared::check_cuda(cudaMemsetAsync(impl_->voxels.data(), 0,
                                               impl_->params.voxel_count * sizeof(float),
                                               impl_->stream),
                               "cudaMemset(voxels)");
#ifdef EVENT_VOXEL_NVTX
        }
#endif
        impl_->prepare();
        impl_->kernel();
    });
}
StageTiming CudaExecutionSession::download_grid(std::span<float> grid) {
#ifdef EVENT_VOXEL_NVTX
    nvtx3::scoped_range marker("D2H voxels");
#endif
    if (grid.size() != impl_->params.voxel_count)
        throw std::invalid_argument("Grid size mismatch");
    return impl_->measure([this, grid] {
        shared::check_cuda(cudaMemcpyAsync(grid.data(), impl_->voxels.data(), grid.size_bytes(),
                                           cudaMemcpyDeviceToHost, impl_->stream),
                           "D2H(grid)");
    });
}
const float* CudaExecutionSession::device_grid() const noexcept {
    return impl_->voxels.data();
}
std::size_t CudaExecutionSession::input_capacity() const noexcept {
    return impl_->events.size();
}
std::size_t CudaExecutionSession::output_capacity() const noexcept {
    return impl_->voxels.size();
}

} // namespace event_voxel::cuda
