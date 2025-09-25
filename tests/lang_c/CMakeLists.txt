add_executable(test_qeq test_qeq.cpp)
target_link_libraries(test_qeq GTest::gtest_main oif_c)
target_include_directories(test_qeq PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_qeq
                           PUBLIC ${CMAKE_SOURCE_DIR}/oif/interfaces/c/include)

add_executable(test_linsolve test_linsolve.cpp)
target_link_libraries(test_linsolve GTest::gtest_main oif_c)
target_include_directories(test_linsolve PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_linsolve
                           PUBLIC ${CMAKE_SOURCE_DIR}/oif/interfaces/c/include)

add_executable(test_ivp test_ivp.cpp)
target_link_libraries(test_ivp GTest::gtest_main oif_c)
target_include_directories(test_ivp PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_ivp
                           PUBLIC ${CMAKE_SOURCE_DIR}/oif/interfaces/c/include)
target_include_directories(test_ivp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(test_ivp PUBLIC cxx_std_11)
set_target_properties(test_ivp PROPERTIES CXX_EXTENSIONS OFF)

add_executable(test_config_dict test_config_dict.cpp)
target_link_libraries(test_config_dict GTest::gtest_main oif_c)
target_include_directories(test_config_dict PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(test_examples test_examples.cpp)
target_link_libraries(test_examples GTest::gtest_main)
target_include_directories(test_examples PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(test_output test_output.cpp)
target_link_libraries(test_output GTest::gtest_main)
target_include_directories(test_output PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(run_aux_tests_config_dict run_aux_tests_config_dict.cpp)
target_link_libraries(run_aux_tests_config_dict GTest::gtest_main oif_c)
target_include_directories(run_aux_tests_config_dict
                           PUBLIC ${CMAKE_SOURCE_DIR}/include)

gtest_discover_tests(test_qeq)
gtest_discover_tests(test_linsolve)
gtest_discover_tests(test_ivp)
gtest_discover_tests(test_config_dict)
gtest_discover_tests(test_examples WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
gtest_discover_tests(test_output WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
