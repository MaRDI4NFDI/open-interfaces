find_package(PkgConfig REQUIRED)
pkg_check_modules(FFI REQUIRED IMPORTED_TARGET libffi)

add_library(oif_dispatch_c SHARED dispatch_c.c)
target_include_directories(
  oif_dispatch_c
  PRIVATE ${CMAKE_SOURCE_DIR}/oif/include
  PRIVATE ${FFI_INCLUDE_DIRS})

target_link_libraries(oif_dispatch_c PRIVATE oif_c_data_structures)
target_link_libraries(oif_dispatch_c PRIVATE oif_c_util)
target_link_libraries(oif_dispatch_c PRIVATE PkgConfig::FFI)
