find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

add_custom_target(
  run_lang_python
  ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/main_python.py
  ${CMAKE_CURRENT_BINARY_DIR}/../connector/liboif_connector.so
  DEPENDS oif_connector oif_julia)

add_library(oif_python SHARED oif_python.c ../oif_connector/oif_interface.h)

# what target itself needs
target_include_directories(
  oif_python
  PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
         $<BUILD_INTERFACE:${Python3_INCLUDE_DIRS}>
         $<BUILD_INTERFACE:${Python3_NumPy_INCLUDE_DIRS}>
         $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
target_link_libraries(oif_python $<BUILD_INTERFACE:${Python3_LIBRARIES}>)

oif_enable_compile_warnings(oif_python)
add_dependencies(all_lang oif_python)

install(TARGETS oif_python DESTINATION lib)
