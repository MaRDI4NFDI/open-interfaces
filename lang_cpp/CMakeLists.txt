add_library(oif_cpp SHARED oif_cpp.cc)

# what target itself needs
target_include_directories(
  oif_cpp
  PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
         $<BUILD_INTERFACE:${Julia_INCLUDE_DIRS}>
         $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)

oif_enable_compile_warnings(oif_cpp)
target_link_libraries(oif_cpp $<BUILD_INTERFACE:flexiblas>)
add_dependencies(all_lang oif_cpp)

add_executable(main_cpp main_cpp.cc)
oif_enable_compile_warnings(main_cpp)
target_include_directories(
  main_cpp PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
                  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
target_link_libraries(main_cpp $<BUILD_INTERFACE:oif_connector>)

target_include_directories(main_cpp PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/..)
install(TARGETS main_cpp DESTINATION bin)
add_dependencies(all_lang main_cpp)
