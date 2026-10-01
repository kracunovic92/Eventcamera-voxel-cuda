cmake_minimum_required(VERSION 3.24)

project(
    event_voxel_cuda
    VERSION 0.1.0
    DESCRIPTION "GPU optimization of event-to-voxel conversion"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(
    EVENT_VOXEL_ENABLE_CUDA
    "Build CUDA implementations"
    OFF
)

# --------------------------------------------------
# Main CPU library
# --------------------------------------------------

add_library(
    event_voxel_cpu
    src/cpu/voxelize.cpp
)

target_include_directories(
    event_voxel_cpu
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(
    event_voxel_cpu
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

# --------------------------------------------------
# CUDA -- optional
# --------------------------------------------------

if(EVENT_VOXEL_ENABLE_CUDA)
    enable_language(CUDA)

    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    find_package(CUDAToolkit REQUIRED)

    # CUDA targets come later.
endif()

# --------------------------------------------------
# Tests
# --------------------------------------------------

include(CTest)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

# --------------------------------------------------
# Formatting
# --------------------------------------------------

add_custom_target(
    format
    COMMAND ${PROJECT_SOURCE_DIR}/scripts/format.sh
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

add_custom_target(
    format-check
    COMMAND ${PROJECT_SOURCE_DIR}/scripts/check-format.sh
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
