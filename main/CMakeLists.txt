add_executable(oif_main oif_main.c)
if(OIF_USE_JULIA)
  target_link_libraries(oif_main PUBLIC oif_julia)
endif()
target_include_directories(oif_main
                           PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/..
                           )
