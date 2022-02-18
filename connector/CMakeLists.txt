add_library(oif_connector oif_connector.c)

# what target itself needs
target_include_directories(
  oif_connector PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
                       $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>)
target_link_libraries(oif_connector $<BUILD_INTERFACE:dl>)

# what consumers of this lib need
target_include_directories(oif_connector INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/)

install(TARGETS oif_connector DESTINATION lib)
install(FILES oif_connector.h DESTINATION include)
