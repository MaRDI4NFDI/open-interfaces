add_library(oif_connector SHARED oif_connector.c oif_interface.h)

# what target itself needs
target_include_directories(
  oif_connector
  PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
         $<BUILD_INTERFACE:${R_INCLUDE_DIRS}>
         $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>)
target_link_libraries(oif_connector $<BUILD_INTERFACE:dl>
                      $<BUILD_INTERFACE:${R_LIBRARY}>)
oif_enable_compile_warnings(oif_connector)

# what consumers of this lib need
target_include_directories(oif_connector INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/)
# TODO This feature is missing for clang
target_compile_features(oif_connector
                        PUBLIC $<$<OR:$<CXX_COMPILER_ID:GNU>>:c_std_11>)

install(TARGETS oif_connector DESTINATION lib)
install(FILES oif_connector.h oif_interface.h
        DESTINATION include/oif_connector/)
