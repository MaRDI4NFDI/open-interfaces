cmake_minimum_required(VERSION 3.18)

project(oif-toy-example LANGUAGES C CXX)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Disable optimizations for Debug build type")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
endif()

add_library(oif_dispatch SHARED dispatch.c)
target_compile_features(oif_dispatch PUBLIC c_std_11)
target_include_directories(oif_dispatch PUBLIC .)

add_subdirectory(src/oif)



# -----------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_executable(test_qeq tests/oif/backend_c/test_qeq.cpp)
target_link_libraries(test_qeq GTest::gtest_main oif_backend_c)
target_include_directories(test_qeq PUBLIC include)
target_include_directories(test_qeq PUBLIC ${CMAKE_SOURCE_DIR})
target_compile_features(test_qeq PUBLIC cxx_std_11)
set_target_properties(test_qeq PROPERTIES CXX_EXTENSIONS OFF)

include(GoogleTest)
gtest_discover_tests(test_qeq)

include(CTest)
add_test(NAME test_qeq COMMAND test_qeq)
