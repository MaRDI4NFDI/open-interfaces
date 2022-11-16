find_library(flexiblas REQUIRED NAMES flexiblas)
find_path(flexiblas_INCLUDE_DIR flexiblas/cblas.h REQUIRED)

add_library(oif_cpp SHARED oif_cpp.cc)

# what target itself needs
target_include_directories(
  oif_cpp
  PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
         $<BUILD_INTERFACE:${Julia_INCLUDE_DIRS}>
         $<BUILD_INTERFACE:${flexiblas_INCLUDE_DIR}>
         $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)

oif_enable_compile_warnings(oif_cpp)
target_link_libraries(oif_cpp $<BUILD_INTERFACE:flexiblas>)
add_dependencies(all_lang oif_cpp)

install(TARGETS oif_cpp DESTINATION lib)
