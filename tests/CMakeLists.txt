enable_testing()
add_test(
  NAME pytest
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${OIF_PYTEST} -c ${PROJECT_SOURCE_DIR}/setup.cfg
          ${PROJECT_SOURCE_DIR}/tests/python)

add_test(
  NAME requirements
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${OIF_VENV_PATH}/bin/check_reqs
          ${PROJECT_SOURCE_DIR}/requirements.txt)

add_test(NAME build_all_lang
         COMMAND ${CMAKE_COMMAND} --build ${PROJECT_BINARY_DIR} --target
                 all_lang --config $<CONFIGURATION>)
set_tests_properties(pytest PROPERTIES DEPENDS build_all_lang)

add_executable(test_all_cpp cpp/test_all.cc)
add_test(NAME all_cpp COMMAND test_all_cpp)
oif_enable_compile_warnings(test_all_cpp)
target_include_directories(
  test_all_cpp PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
                      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
target_link_libraries(test_all_cpp $<BUILD_INTERFACE:oif_connector>)

add_dependencies(all_lang test_all_cpp)

add_executable(test_all_c c/test_all.c)
add_test(NAME all_c COMMAND test_all_c)
oif_enable_compile_warnings(test_all_c)
target_include_directories(
  test_all_c PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
                    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
target_link_libraries(test_all_c $<BUILD_INTERFACE:oif_connector>)

target_include_directories(test_all_c PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/..)
add_dependencies(all_lang test_all_c)
