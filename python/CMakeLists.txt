find_package(Python3 REQUIRED COMPONENTS Development)
add_library(oif_python SHARED oif_python.c)

# what target itself needs
target_include_directories(
  oif_python
  PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
         $<BUILD_INTERFACE:${Python3_INCLUDE_DIRS}>
         $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>)
target_link_libraries(oif_python $<BUILD_INTERFACE:${Python3_LIBRARIES}>)

oif_enable_compile_warnings(oif_python)

# what consumers of this lib need
target_include_directories(oif_python INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS oif_python DESTINATION lib)
install(FILES oif_python.h DESTINATION include)
