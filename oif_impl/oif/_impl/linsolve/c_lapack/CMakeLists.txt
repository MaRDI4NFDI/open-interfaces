set(BLA_VENDOR Intel10_64_dyn)
find_package(LAPACK)

if(NOT LAPACK_FOUND)
  message("LAPACK not found. Skipping `linsolve/c_lapack` implementation.")
  return()
endif()

# We need to look for the include directory for LAPACK, as it is somehow not
# provided by `find_package`. It is particularly a problem on macOS (where
# Homebrew is a common package manager). Note that we need lapackE.h, this is
# not a typo.
find_path(
  LAPACKE_INCLUDE_DIR
  NAMES lapacke.h
  HINTS /opt/homebrew/opt/openblas/include
  PATH_SUFFIXES lapack lapack/include)

add_library(oif_linsolve_c_lapack SHARED linsolve_impl.c)
target_include_directories(oif_linsolve_c_lapack
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(oif_linsolve_c_lapack
                           PUBLIC ${CMAKE_SOURCE_DIR}/oif_impl/lang_c/include)
target_include_directories(oif_linsolve_c_lapack ${LAPACKE_INCLUDE_DIR})
target_link_libraries(oif_linsolve_c_lapack PRIVATE LAPACK::LAPACK)
