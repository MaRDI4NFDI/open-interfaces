find_package(R REQUIRED)
add_library(oif_r SHARED oif_r.c)

# what target itself needs
target_include_directories(
  oif_r
  PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
         $<BUILD_INTERFACE:${R_INCLUDE_DIRS}>
         $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
target_link_libraries(oif_r $<BUILD_INTERFACE:${R_LIBRARY}>)
add_dependencies(all_lang oif_r)
oif_enable_compile_warnings(oif_r)

# what consumers of this lib need
target_include_directories(oif_r INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS oif_r DESTINATION lib)
