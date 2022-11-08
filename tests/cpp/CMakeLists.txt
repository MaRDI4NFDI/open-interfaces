add_executable(test_all_cpp test_all.cc catch2/catch_amalgamated.cpp)
oif_add_test(NAME all_cpp COMMAND test_all_cpp)
oif_enable_compile_warnings(test_all_cpp)
target_include_directories(
  test_all_cpp PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
                      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
target_link_libraries(test_all_cpp $<BUILD_INTERFACE:oif_connector>)

add_dependencies(test_all_cpp all_lang)
