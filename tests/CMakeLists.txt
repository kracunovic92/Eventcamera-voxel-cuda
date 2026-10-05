
include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG 8b08d4d79514f45f7e4ce2a607ac9c94e920d1bb
)

FetchContent_MakeAvailable(Catch2)

# Automatically discover unit test source files.
file(
    GLOB_RECURSE TEST_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/unit/test_*.cpp"
)

if(NOT TARGET event_voxel_cuda)
    list(REMOVE_ITEM TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/unit/test_cuda_voxelizer.cpp")
endif()

add_executable(
    event_voxel_tests
    ${TEST_SOURCES}
)

target_link_libraries(
    event_voxel_tests
    PRIVATE
        event_voxel_cpu
        Catch2::Catch2WithMain
)

# Allow tests to locate fixture files.
target_compile_definitions(
    event_voxel_tests
    PRIVATE
        TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
)

# Register individual Catch2 test cases with CTest.
list(
    APPEND CMAKE_MODULE_PATH
    ${catch2_SOURCE_DIR}/extras
)


if(TARGET event_voxel_cuda)
    target_link_libraries(
        event_voxel_tests
        PRIVATE
            event_voxel_cuda
    )
endif()
include(Catch)

catch_discover_tests(event_voxel_tests)
