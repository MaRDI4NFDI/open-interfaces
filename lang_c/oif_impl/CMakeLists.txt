find_package(PkgConfig REQUIRED)
pkg_check_modules(FFI REQUIRED IMPORTED_TARGET libffi)

add_library(oif_bridge_c SHARED bridge_c.c allocation_tracker.c)
target_include_directories(
  oif_bridge_c
  PRIVATE ${CMAKE_SOURCE_DIR}/include
  ${FFI_INCLUDE_DIRS})

target_link_libraries(oif_bridge_c PRIVATE oif_common_data_structures)
target_link_libraries(oif_bridge_c PRIVATE oif_common_util)
target_link_libraries(oif_bridge_c PRIVATE PkgConfig::FFI)

add_subdirectory(_impl)
