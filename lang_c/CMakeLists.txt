add_executable(oif_main oif_c.c)

target_link_libraries(oif_main PUBLIC oif_connector)
oif_enable_compile_warnings(oif_main)

target_include_directories(oif_main PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/..)
install(TARGETS oif_main DESTINATION bin)
