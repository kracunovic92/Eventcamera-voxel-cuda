cmake_minimum_required(VERSION 3.30)

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

set(EVENT_VOXEL_HDF5_TEST_INPUT "${PROJECT_SOURCE_DIR}/data/raw/sparklers.hdf5"
    CACHE FILEPATH "Real HDF5 recording for the optional streaming integration test")

# HDF5 recordings are the only experiment input, so every build needs Stream.
if(EXISTS "${PROJECT_SOURCE_DIR}/.deps/openeb/share/cmake/MetavisionSDK/MetavisionSDKConfig.cmake")
    set(EVENT_VOXEL_LOCAL_OPENEB TRUE)
    list(PREPEND CMAKE_PREFIX_PATH
        "${PROJECT_SOURCE_DIR}/.deps/openeb"
        "${PROJECT_SOURCE_DIR}/.deps/openeb-build-env")
endif()
find_package(MetavisionSDK COMPONENTS stream REQUIRED)
if(NOT TARGET MetavisionSDK::stream)
    message(FATAL_ERROR "Metavision SDK Stream did not provide MetavisionSDK::stream")
endif()

# --------------------------------------------------
# Common library
# --------------------------------------------------

add_library(
    event_voxel_common
    src/common/validation.cpp
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

add_library(event_voxel_metavision src/data/metavision/hdf5_time_slice_source.cpp)
target_include_directories(event_voxel_metavision PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(event_voxel_metavision PUBLIC event_voxel_common PRIVATE MetavisionSDK::stream)
if(EVENT_VOXEL_LOCAL_OPENEB AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # System OpenCV and the local SDK use different HDF5/CURL builds.
    target_link_options(event_voxel_metavision INTERFACE
        "LINKER:-rpath-link,/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}")
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
# Experiment application
# --------------------------------------------------

find_package(nlohmann_json 3.11 CONFIG REQUIRED)

add_executable(event_voxel_experiment
    apps/experiment.cpp
    apps/experiment_config.cpp
    apps/experiment_metrics.cpp
    apps/experiment_results.cpp
    apps/experiment_run.cpp)
target_link_libraries(event_voxel_experiment PRIVATE
    event_voxel_metavision nlohmann_json::nlohmann_json)
execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                OUTPUT_VARIABLE EVENT_VOXEL_GIT_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
target_compile_definitions(event_voxel_experiment PRIVATE
    EVENT_VOXEL_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    EVENT_VOXEL_GIT_COMMIT="${EVENT_VOXEL_GIT_COMMIT}")
target_compile_definitions(event_voxel_experiment PRIVATE
    EVENT_VOXEL_METAVISION_VERSION="${MetavisionSDK_VERSION}")

# --------------------------------------------------
# CUDA -- optional
# --------------------------------------------------

if(EVENT_VOXEL_ENABLE_CUDA)
    include(CheckLanguage)

    check_language(CUDA)

    if(NOT CMAKE_CUDA_COMPILER)
        message(
            FATAL_ERROR
            "EVENT_VOXEL_ENABLE_CUDA=ON, but no CUDA compiler was found. "
            "Select a preset with a valid CMAKE_CUDA_COMPILER, put nvcc on PATH, "
            "or set CUDACXX."
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

    # CUDA builds always include NVTX3 markers; header locations vary by toolkit.
    file(GLOB _nvtx_candidates "${CUDAToolkit_ROOT}/nsight-compute*/host/target-linux-x64/nvtx/include")
    find_path(EVENT_VOXEL_NVTX3_INCLUDE_DIR nvtx3/nvtx3.hpp
              HINTS ${_nvtx_candidates} ${CUDAToolkit_INCLUDE_DIRS})
    if(NOT EXISTS "${EVENT_VOXEL_NVTX3_INCLUDE_DIR}/nvtx3/nvtx3.hpp")
        message(FATAL_ERROR "CUDA builds require NVTX3 headers (nvtx3/nvtx3.hpp). "
                            "Set EVENT_VOXEL_NVTX3_INCLUDE_DIR to their include directory.")
    endif()
    target_include_directories(event_voxel_cuda PRIVATE ${EVENT_VOXEL_NVTX3_INCLUDE_DIR})
    target_include_directories(event_voxel_experiment PRIVATE ${EVENT_VOXEL_NVTX3_INCLUDE_DIR})
    target_include_directories(event_voxel_metavision PRIVATE ${EVENT_VOXEL_NVTX3_INCLUDE_DIR})
    target_compile_definitions(event_voxel_metavision PRIVATE EVENT_VOXEL_HAS_CUDA)

    set_target_properties(
        event_voxel_cuda
        PROPERTIES
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
            CUDA_EXTENSIONS OFF
    )

    # --------------------------------------------------
    # CUDA experiment support
    # --------------------------------------------------

    target_link_libraries(event_voxel_experiment PRIVATE event_voxel_cuda)
    target_compile_definitions(event_voxel_experiment PRIVATE EVENT_VOXEL_HAS_CUDA)
    target_compile_definitions(event_voxel_experiment PRIVATE
        EVENT_VOXEL_CUDA_TOOLKIT_VERSION="${CUDAToolkit_VERSION}")

endif()

if(NOT EVENT_VOXEL_ENABLE_CUDA)
    target_link_libraries(event_voxel_experiment PRIVATE event_voxel_cpu)
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
