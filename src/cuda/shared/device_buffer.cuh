#pragma once

#include "cuda_check.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

namespace event_voxel::cuda::shared {

template <typename T> class DeviceBuffer {
  public:
    DeviceBuffer() = default;

    DeviceBuffer(std::size_t count, const char* allocation_name) : count_(count) {
        if (count_ == 0) {
            return;
        }

        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), bytes()), allocation_name);
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;

    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), count_(std::exchange(other.count_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this == &other) {
            return *this;
        }

        if (data_ != nullptr) {
            cudaFree(data_);
        }

        data_ = std::exchange(other.data_, nullptr);

        count_ = std::exchange(other.count_, 0);

        return *this;
    }

    [[nodiscard]]
    T* data() noexcept {
        return data_;
    }

    [[nodiscard]]
    const T* data() const noexcept {
        return data_;
    }

    [[nodiscard]]
    std::size_t size() const noexcept {
        return count_;
    }

    [[nodiscard]]
    std::size_t bytes() const noexcept {
        return count_ * sizeof(T);
    }

  private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

} // namespace event_voxel::cuda::shared
