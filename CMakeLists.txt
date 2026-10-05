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
    EVENT_VOXEL_ENABLE_CUDA
    "Build CUDA implementations"
    OFF
)

option(
    EVENT_VOXEL_BUILD_BENCHMARKS
    "Build benchmark executables"
    OFF
)

# --------------------------------------------------
# Common library
# --------------------------------------------------

add_library(
    event_voxel_common
    src/common/validation.cpp
    src/data/binary_event_loader.cpp
    src/data/sample_event_batch.cpp
    src/data/synthetic_generator.cpp
    src/io/npy_writer.cpp
)

target_include_directories(
    event_voxel_common
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

# --------------------------------------------------
# CPU implementation
# --------------------------------------------------

add_library(
    event_voxel_cpu
    src/cpu/reference_voxelizer.cpp
)

target_include_directories(
    event_voxel_cpu
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(
    event_voxel_cpu
    PUBLIC
        event_voxel_common
)

# --------------------------------------------------
# Compiler warnings
# --------------------------------------------------

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
        event_voxel_common
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

    target_compile_options(
        event_voxel_cpu
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

elseif(MSVC)
    target_compile_options(
        event_voxel_common
        PRIVATE
            /W4
    )

    target_compile_options(
        event_voxel_cpu
        PRIVATE
            /W4
    )
endif()

# --------------------------------------------------
# CPU application
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
# CPU benchmarks
# --------------------------------------------------

if(EVENT_VOXEL_BUILD_BENCHMARKS)
    include(FetchContent)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG 192ef10025eb2c4cdd392bc502f0c852196baa48
    )
    FetchContent_MakeAvailable(googlebenchmark)

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

# --------------------------------------------------
# CUDA -- optional
# --------------------------------------------------

if(EVENT_VOXEL_ENABLE_CUDA)
    include(CheckLanguage)

    check_language(CUDA)

    if(NOT CMAKE_CUDA_COMPILER)
        message(
            FATAL_ERROR
            "EVENT_VOXEL_ENABLE_CUDA=ON, but no CUDA compiler was found"
        )
    endif()

    enable_language(CUDA)

    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    find_package(CUDAToolkit REQUIRED)

    message(
        STATUS
        "CUDA backend enabled: ${CMAKE_CUDA_COMPILER}"
    )

    # --------------------------------------------------
    # CUDA library
    # --------------------------------------------------

    add_library(
        event_voxel_cuda
        src/cuda/voxelizer.cu
        src/cuda/event_scatter.cu
        src/cuda/pixel_gather.cu
        src/cuda/sort_reduce.cu
    )

    target_include_directories(
        event_voxel_cuda
        PUBLIC
            ${PROJECT_SOURCE_DIR}/include
    )

    target_link_libraries(
        event_voxel_cuda
        PUBLIC
            event_voxel_common
        PRIVATE
            CUDA::cudart
    )

    set_target_properties(
        event_voxel_cuda
        PROPERTIES
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
            CUDA_EXTENSIONS OFF
    )

    # --------------------------------------------------
    # CUDA application
    # --------------------------------------------------

    add_executable(
        event_voxel_run_cuda
        apps/run_cuda.cpp
    )

    target_link_libraries(
        event_voxel_run_cuda
        PRIVATE
            event_voxel_cuda
    )

    add_executable(event_voxel_verify_cuda apps/verify_cuda.cpp)
    target_link_libraries(event_voxel_verify_cuda PRIVATE event_voxel_cpu event_voxel_cuda)

    if(EVENT_VOXEL_BUILD_BENCHMARKS)
        add_executable(event_voxel_bench_cuda benchmarks/bench_cuda.cpp)
        target_link_libraries(event_voxel_bench_cuda PRIVATE event_voxel_cuda)
    endif()
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

set(
    FORMAT_SCRIPT
    "${PROJECT_SOURCE_DIR}/scripts/format.sh"
)

set(
    CHECK_FORMAT_SCRIPT
    "${PROJECT_SOURCE_DIR}/scripts/check-format.sh"
)

if(EXISTS "${FORMAT_SCRIPT}")
    add_custom_target(
        format
        COMMAND bash "${FORMAT_SCRIPT}"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        COMMENT "Formatting source files"
        VERBATIM
    )
endif()

if(EXISTS "${CHECK_FORMAT_SCRIPT}")
    add_custom_target(
        format-check
        COMMAND bash "${CHECK_FORMAT_SCRIPT}"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        COMMENT "Checking source formatting"
        VERBATIM
    )
endif()
