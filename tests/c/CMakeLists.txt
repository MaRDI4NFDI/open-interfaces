add_executable(test_all_c test_all.c)
oif_add_test(NAME all_c COMMAND test_all_c)
oif_enable_compile_warnings(test_all_c)
target_include_directories(
  test_all_c PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
                    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
target_link_libraries(test_all_c $<BUILD_INTERFACE:oif_connector>)

target_include_directories(test_all_c PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/..)
add_dependencies(test_all_c all_lang)
set_tests_properties(all_c PROPERTIES DEPENDS build_all_lang)
