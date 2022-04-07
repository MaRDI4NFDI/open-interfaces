find_package(Julia REQUIRED)
add_library(oif_julia SHARED oif_julia.c)

# what target itself needs
target_include_directories(
  oif_julia
  PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
         $<BUILD_INTERFACE:${Julia_INCLUDE_DIRS}>
         $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
target_link_libraries(oif_julia $<BUILD_INTERFACE:${Julia_LIBRARY}>)

oif_enable_compile_warnings(oif_julia)
add_dependencies(all_lang oif_julia)

# what consumers of this lib need
target_include_directories(oif_julia INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS oif_julia DESTINATION lib)
install(PROGRAMS main_julia.jl DESTINATION bin)
