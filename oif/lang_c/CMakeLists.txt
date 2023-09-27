# Frontend C
add_library(
    oif_frontend_c SHARED c_bindings.c
    ${CMAKE_SOURCE_DIR}/src/oif/frontend_c/qeq.c
    ${CMAKE_SOURCE_DIR}/src/oif/frontend_c/linsolve.c
)
target_include_directories(oif_frontend_c PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(oif_frontend_c PUBLIC ${CMAKE_SOURCE_DIR}/oif/include)
target_link_libraries(oif_frontend_c PRIVATE oif_dispatch)
