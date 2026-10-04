
cmake_minimum_required(VERSION 3.24)

project(
    event_voxel_cuda
    VERSION 0.1.0
    DESCRIPTION "GPU optimization of event-to-voxel conversion"
    LANGUAGES CXX
)

# --------------------------------------------------
# Project configuration
# --------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(
    EVENT_VOXEL_BUILD_BENCHMARKS
    "Build performance benchmarks"
    OFF
)

# --------------------------------------------------
# Main CPU library
# --------------------------------------------------

add_library(
    event_voxel_cpu

    src/cpu/reference_voxelizer.cpp

    src/data/synthetic_generator.cpp
    src/data/binary_event_loader.cpp

    src/io/npy_writer.cpp
)

target_include_directories(
    event_voxel_cpu
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
        event_voxel_cpu
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )
elseif(MSVC)
    target_compile_options(
        event_voxel_cpu
        PRIVATE
            /W4
    )
endif()

# --------------------------------------------------
# CPU processing application
# --------------------------------------------------

add_executable(
    event_voxel_run_cpu
    apps/run_cpu.cpp
)

target_link_libraries(
    event_voxel_run_cpu
    PRIVATE
        event_voxel_cpu
)

# --------------------------------------------------
# Correctness tests
# --------------------------------------------------

include(CTest)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

# --------------------------------------------------
# Google Benchmark
# --------------------------------------------------

if(EVENT_VOXEL_BUILD_BENCHMARKS)
    include(FetchContent)

    # We need Google Benchmark itself, not its own
    # unit tests or installation targets.
    set(
        BENCHMARK_ENABLE_TESTING
        OFF CACHE BOOL "" FORCE
    )

    set(
        BENCHMARK_ENABLE_GTEST_TESTS
        OFF CACHE BOOL "" FORCE
    )

    set(
        BENCHMARK_ENABLE_INSTALL
        OFF CACHE BOOL "" FORCE
    )

    # Pinned to the locally verified v1.9.5 source revision.
    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY
            https://github.com/google/benchmark.git
        GIT_TAG
            192ef10025eb2c4cdd392bc502f0c852196baa48
    )

    FetchContent_MakeAvailable(googlebenchmark)

    # A separate executable for performance experiments.
    add_executable(
        event_voxel_bench_cpu
        benchmarks/bench_cpu.cpp
    )

    target_link_libraries(
        event_voxel_bench_cpu
        PRIVATE
            event_voxel_cpu
            benchmark::benchmark_main
    )
endif()

add_custom_target(
    format-check
    COMMAND bash "${PROJECT_SOURCE_DIR}/scripts/check-format.sh"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    VERBATIM
)
