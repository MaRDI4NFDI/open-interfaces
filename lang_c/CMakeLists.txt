add_executable(oif_c oif_c.c)

target_link_libraries(oif_c PUBLIC oif_connector)
oif_enable_compile_warnings(oif_c)

target_include_directories(oif_c PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/..)
install(TARGETS oif_c DESTINATION bin)
