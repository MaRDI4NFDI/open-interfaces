add_library(oif_linsolve_c_lapack SHARED linsolve_impl.c)
target_link_libraries(oif_linsolve_c_lapack PRIVATE lapack)
target_include_directories(oif_linsolve_c_lapack PRIVATE ${CMAKE_SOURCE_DIR}/oif/include)
target_include_directories(oif_linsolve_c_lapack PUBLIC ${CMAKE_SOURCE_DIR}/oif_impl/c/include)

