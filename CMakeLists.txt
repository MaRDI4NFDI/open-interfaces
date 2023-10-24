cmake_minimum_required(VERSION 3.18)

project(oif-toy-example LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Disable optimizations for Debug build type")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
endif()

# DOWNLOAD_EXTRACT_TIMESTAMP = TRUE
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(oif)
file(COPY oif_impl DESTINATION ${CMAKE_BINARY_DIR})
add_subdirectory(oif_impl)

add_subdirectory(examples)



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

 add_executable(test_qeq tests/lang_c/test_qeq.cpp)
 target_link_libraries(test_qeq GTest::gtest_main oif_c)
 target_include_directories(test_qeq PUBLIC ${CMAKE_SOURCE_DIR}/oif/include)
 target_include_directories(test_qeq PUBLIC ${CMAKE_SOURCE_DIR}/oif/interfaces/c/include)
 target_compile_features(test_qeq PUBLIC cxx_std_11)
 set_target_properties(test_qeq PROPERTIES CXX_EXTENSIONS OFF)

 add_executable(test_linsolve tests/lang_c/test_linsolve.cpp)
 target_link_libraries(test_linsolve GTest::gtest_main oif_c)
 target_include_directories(test_linsolve PUBLIC ${CMAKE_SOURCE_DIR}/oif/include)
 target_include_directories(test_linsolve PUBLIC ${CMAKE_SOURCE_DIR}/oif/interfaces/c/include)
 target_compile_features(test_linsolve PUBLIC cxx_std_11)
 set_target_properties(test_linsolve PROPERTIES CXX_EXTENSIONS OFF)
 
 include(GoogleTest)
 gtest_discover_tests(test_qeq)
 gtest_discover_tests(test_linsolve)
 
 include(CTest)
 add_test(NAME test_qeq COMMAND test_qeq)
