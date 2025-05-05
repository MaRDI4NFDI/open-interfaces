# Common components that are used by C codes: utility functions and data
# structures.

# -----------------------------------------------------------------------------
# Library oif_common_util
add_library(oif_common_util SHARED util.c)
target_include_directories(oif_common_util PUBLIC ${CMAKE_SOURCE_DIR}/include)
install(TARGETS oif_common_util DESTINATION lib)

# -----------------------------------------------------------------------------
# Library oif_common_data_structures
get_property(cwpack_SOURCE_DIR GLOBAL PROPERTY cwpack_SOURCE_DIR)

add_library(oif_common_data_structures SHARED oif_config_dict.c
                                              ${cwpack_SOURCE_DIR}/src/cwpack.c)
target_include_directories(oif_common_data_structures
                           PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(oif_common_data_structures
                           PRIVATE ${cwpack_SOURCE_DIR}/src)
target_link_libraries(oif_common_data_structures PRIVATE oif_common_util)
target_link_libraries(oif_common_data_structures PRIVATE HashMap::HashMap)
install(TARGETS oif_common_data_structures DESTINATION lib)
