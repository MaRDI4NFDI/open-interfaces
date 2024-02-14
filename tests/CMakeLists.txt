include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(test_qeq lang_c/test_qeq.cpp)
target_link_libraries(test_qeq GTest::gtest_main oif_c)
target_include_directories(test_qeq PUBLIC ${CMAKE_SOURCE_DIR}/oif/include)
target_include_directories(test_qeq
                           PUBLIC ${CMAKE_SOURCE_DIR}/oif/interfaces/c/include)
target_compile_features(test_qeq PUBLIC cxx_std_11)
set_target_properties(test_qeq PROPERTIES CXX_EXTENSIONS OFF)

add_executable(test_linsolve lang_c/test_linsolve.cpp)
target_link_libraries(test_linsolve GTest::gtest_main oif_c)
target_include_directories(test_linsolve PUBLIC ${CMAKE_SOURCE_DIR}/oif/include)
target_include_directories(test_linsolve
                           PUBLIC ${CMAKE_SOURCE_DIR}/oif/interfaces/c/include)
target_compile_features(test_linsolve PUBLIC cxx_std_11)
set_target_properties(test_linsolve PROPERTIES CXX_EXTENSIONS OFF)

add_executable(test_ivp lang_c/test_ivp.cpp)
target_link_libraries(test_ivp GTest::gtest_main oif_c)
target_include_directories(test_ivp PUBLIC ${CMAKE_SOURCE_DIR}/oif/include)
target_include_directories(test_ivp
                           PUBLIC ${CMAKE_SOURCE_DIR}/oif/interfaces/c/include)
target_include_directories(test_ivp PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_compile_features(test_ivp PUBLIC cxx_std_11)
set_target_properties(test_ivp PROPERTIES CXX_EXTENSIONS OFF)

include(GoogleTest)
gtest_discover_tests(test_qeq)
gtest_discover_tests(test_linsolve)
gtest_discover_tests(test_ivp)
