#include "event_voxel/cuda/voxelizer.hpp"

#include <iostream>

int main() {

    if (!event_voxel::cuda::is_available()) {
        std::cerr << "CUDA device is not available \n";
        return 1;
    }

    std::cout << "CUDA device is available \n";
    return 0;
}
