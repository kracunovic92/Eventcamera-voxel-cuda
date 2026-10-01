include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.15.3
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(Catch2)

add_executable(
    event_voxel_tests
    unit/test_voxel_grid.cpp
)

target_link_libraries(
    event_voxel_tests
    PRIVATE
        event_voxel_cpu
        Catch2::Catch2WithMain
)

list(
    APPEND CMAKE_MODULE_PATH
    ${catch2_SOURCE_DIR}/extras
)

include(Catch)

catch_discover_tests(
    event_voxel_tests
)
