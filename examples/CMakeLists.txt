add_custom_target(examples)

add_executable(example_c c/main_c.c)
oif_enable_compile_warnings(example_c)
target_include_directories(
  example_c PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
                   $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
target_link_libraries(example_c $<BUILD_INTERFACE:oif_connector>)
target_include_directories(example_c PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/..)
add_dependencies(examples example_c)

add_executable(example_cpp c/main_c.c)
oif_enable_compile_warnings(example_cpp)
target_include_directories(
  example_cpp PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
                     $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
target_link_libraries(example_cpp $<BUILD_INTERFACE:oif_connector>)
target_include_directories(example_cpp PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/..)
add_dependencies(examples example_cpp)
