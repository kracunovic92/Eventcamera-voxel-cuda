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

option(EVENT_VOXEL_ENABLE_METAVISION "Build direct Metavision C++ HDF5 adapter" OFF)

# --------------------------------------------------
# Common library
# --------------------------------------------------

add_library(
    event_voxel_common
    src/common/validation.cpp
    src/data/binary_event_loader.cpp
    src/data/sample_event_batch.cpp
    src/data/synthetic_generator.cpp
    src/data/event_window_source.cpp
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

if(EVENT_VOXEL_ENABLE_METAVISION)
    find_package(MetavisionSDK COMPONENTS stream QUIET)
    if(NOT TARGET MetavisionSDK::stream)
        message(FATAL_ERROR
            "Direct HDF5 was requested but Metavision SDK Stream was not found. "
            "Install Metavision SDK/OpenEB with the HDF5 ECF codec, then set "
            "MetavisionSDK_DIR to its generated CMake package directory. "
            "See docs/streaming-experiments.md.")
    endif()
    add_library(event_voxel_metavision src/data/metavision/hdf5_reader.cpp)
    target_include_directories(event_voxel_metavision PUBLIC ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(event_voxel_metavision PUBLIC event_voxel_common PRIVATE MetavisionSDK::stream)
    message(STATUS "Direct Metavision HDF5 reader enabled")
else()
    message(STATUS "Direct HDF5 reader disabled; enable EVENT_VOXEL_ENABLE_METAVISION with Metavision SDK Stream installed")
endif()

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

add_executable(event_voxel_experiment apps/experiment.cpp)
target_link_libraries(event_voxel_experiment PRIVATE event_voxel_cpu)
execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                OUTPUT_VARIABLE EVENT_VOXEL_GIT_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
target_compile_definitions(event_voxel_experiment PRIVATE
    EVENT_VOXEL_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    EVENT_VOXEL_GIT_COMMIT="${EVENT_VOXEL_GIT_COMMIT}")
if(TARGET event_voxel_metavision)
    target_link_libraries(event_voxel_experiment PRIVATE event_voxel_metavision)
    target_compile_definitions(event_voxel_experiment PRIVATE EVENT_VOXEL_HAS_METAVISION)
    target_compile_definitions(event_voxel_experiment PRIVATE
        EVENT_VOXEL_METAVISION_VERSION="${MetavisionSDK_VERSION}")
endif()

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
        src/cuda/execution_session.cu
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

    # NVTX3 is header-only; locations vary between CUDA distributions.
    file(GLOB _nvtx_candidates "${CUDAToolkit_ROOT}/nsight-compute*/host/target-linux-x64/nvtx/include")
    find_path(EVENT_VOXEL_NVTX_INCLUDE_DIR nvtx3/nvtx3.hpp
              HINTS ${_nvtx_candidates} ${CUDAToolkit_INCLUDE_DIRS})
    if(EVENT_VOXEL_NVTX_INCLUDE_DIR)
        target_include_directories(event_voxel_cuda PRIVATE ${EVENT_VOXEL_NVTX_INCLUDE_DIR})
        target_compile_definitions(event_voxel_cuda PRIVATE EVENT_VOXEL_NVTX)
        target_include_directories(event_voxel_experiment PRIVATE ${EVENT_VOXEL_NVTX_INCLUDE_DIR})
        target_compile_definitions(event_voxel_experiment PRIVATE EVENT_VOXEL_NVTX)
        if(TARGET event_voxel_metavision)
            target_include_directories(event_voxel_metavision PRIVATE ${EVENT_VOXEL_NVTX_INCLUDE_DIR})
            target_compile_definitions(event_voxel_metavision PRIVATE EVENT_VOXEL_NVTX)
        endif()
    else()
        message(STATUS "NVTX3 headers not found; profiling markers disabled")
    endif()

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
            CUDA::cudart
    )

    target_link_libraries(event_voxel_experiment PRIVATE event_voxel_cuda)
    target_compile_definitions(event_voxel_experiment PRIVATE EVENT_VOXEL_HAS_CUDA)
    target_compile_definitions(event_voxel_experiment PRIVATE
        EVENT_VOXEL_CUDA_TOOLKIT_VERSION="${CUDAToolkit_VERSION}")

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
