add_executable(oif_main oif_main.c)

target_link_libraries(oif_main PUBLIC oif_connector)

target_include_directories(oif_main PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/..)
install(TARGETS oif_main DESTINATION bin)
