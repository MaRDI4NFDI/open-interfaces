# Frontend C
cmake_minimum_required(VERSION 3.18)

add_library(oif_frontend_c SHARED frontend_c.c qeq.c)
target_include_directories(oif_frontend_c PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(oif_frontend_c PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(oif_frontend_c PUBLIC ${CMAKE_SOURCE_DIR}/include/oif/oif_frontend_c)
target_link_libraries(oif_frontend_c PRIVATE oif_dispatch)
