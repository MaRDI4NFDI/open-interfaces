set(BLA_VENDOR Intel10_64_dyn)
find_package(LAPACK)

if(NOT LAPACK_FOUND)
  message("LAPACK not found. Skipping `linsolve/c_lapack` implementation.")
  return()
endif()

add_library(oif_linsolve_c_lapack SHARED linsolve_impl.c)
target_include_directories(oif_linsolve_c_lapack
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(oif_linsolve_c_lapack
                           PUBLIC ${CMAKE_SOURCE_DIR}/oif_impl/lang_c/include)
target_link_libraries(oif_linsolve_c_lapack PRIVATE LAPACK::LAPACK)
