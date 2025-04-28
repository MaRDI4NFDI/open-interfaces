# -----------------------------------------------------------------------------
# Library `oif_dispatch`.
add_library(oif_dispatch SHARED dispatch.c)
target_include_directories(oif_dispatch PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(oif_dispatch PRIVATE oif_common_util)
target_link_libraries(oif_dispatch PRIVATE dl)
target_link_libraries(oif_dispatch PRIVATE HashMap::HashMap)
install(TARGETS oif_dispatch DESTINATION lib)
