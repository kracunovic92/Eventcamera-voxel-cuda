#pragma once

#include "cuda_check.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace event_voxel::cuda::shared {

template <typename T> class DeviceBuffer {
  public:
    DeviceBuffer() = default;

    DeviceBuffer(std::size_t count, const char* allocation_name) {
        reserve(count, allocation_name);
    }

    void reserve(std::size_t count, const char* allocation_name) {
        if (count <= count_)
            return;
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::overflow_error("Device allocation byte count overflow");
        }
        T* replacement = nullptr;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&replacement), count * sizeof(T)),
                   allocation_name);
        if (data_ != nullptr)
            cudaFree(data_);
        data_ = replacement;
        count_ = count;
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
    std::size_t bytes() const {
        if (count_ > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::overflow_error("Device allocation byte count overflow");
        }
        return count_ * sizeof(T);
    }

  private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

} // namespace event_voxel::cuda::shared
