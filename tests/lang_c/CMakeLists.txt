add_executable(test_qeq test_qeq.cpp)
target_link_libraries(test_qeq GTest::gtest_main oif_c)
target_include_directories(test_qeq PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_qeq
                           PUBLIC ${CMAKE_SOURCE_DIR}/oif/interfaces/c/include)
target_compile_features(test_qeq PUBLIC cxx_std_11)
set_target_properties(test_qeq PROPERTIES CXX_EXTENSIONS OFF)

add_executable(test_linsolve test_linsolve.cpp)
target_link_libraries(test_linsolve GTest::gtest_main oif_c)
target_include_directories(test_linsolve PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_linsolve
                           PUBLIC ${CMAKE_SOURCE_DIR}/oif/interfaces/c/include)
target_compile_features(test_linsolve PUBLIC cxx_std_11)
set_target_properties(test_linsolve PROPERTIES CXX_EXTENSIONS OFF)

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
target_compile_features(test_config_dict PUBLIC cxx_std_11)
set_target_properties(test_config_dict PROPERTIES CXX_EXTENSIONS OFF)

gtest_discover_tests(test_qeq)
gtest_discover_tests(test_linsolve)
gtest_discover_tests(test_ivp)
gtest_discover_tests(test_config_dict)
