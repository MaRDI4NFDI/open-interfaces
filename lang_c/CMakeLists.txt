add_library(oif_c SHARED oif_c.c)

# what target itself needs
target_include_directories(
  oif_c
  PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
         $<BUILD_INTERFACE:${Julia_INCLUDE_DIRS}>
         $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)

oif_enable_compile_warnings(oif_c)
add_dependencies(all_lang oif_c)

add_executable(main_c main_c.c)
oif_enable_compile_warnings(main_c)
target_include_directories(
  main_c PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
                $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
target_link_libraries(main_c $<BUILD_INTERFACE:oif_connector>)

target_include_directories(main_c PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/..)
install(TARGETS main_c DESTINATION bin)
add_dependencies(all_lang main_c)
