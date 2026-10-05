#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <string_view>

namespace event_voxel::cuda::shared {

inline void check_cuda(cudaError_t status, std::string_view operation) {
    if (status == cudaSuccess) {
        return;
    }

    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

} // namespace event_voxel::cuda::shared
