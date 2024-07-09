get_property(cwpack_SOURCE_DIR GLOBAL PROPERTY cwpack_SOURCE_DIR)

# User-facing C library that allows to interact with Open Interfaces.
add_library(
  oif_c SHARED
  c_bindings.c ${CMAKE_SOURCE_DIR}/oif/interfaces/c/src/qeq.c
  ${CMAKE_SOURCE_DIR}/oif/interfaces/c/src/linsolve.c
  ${CMAKE_SOURCE_DIR}/oif/interfaces/c/src/ivp.c)
target_include_directories(oif_c PUBLIC ${CMAKE_SOURCE_DIR}/oif/include)
target_include_directories(oif_c
                           PUBLIC ${CMAKE_SOURCE_DIR}/oif/interfaces/c/include/)
target_link_libraries(oif_c PUBLIC oif_c_data_structures)
target_link_libraries(oif_c PUBLIC oif_util)
target_link_libraries(oif_c PRIVATE oif_dispatch)

add_library(oif_c_data_structures SHARED oif_config_dict.c allocation_tracker.c
                                         ${cwpack_SOURCE_DIR}/src/cwpack.c)
target_include_directories(oif_c_data_structures
                           PUBLIC ${CMAKE_SOURCE_DIR}/oif/include)
target_link_libraries(oif_c_data_structures PRIVATE HashMap::HashMap)
target_include_directories(oif_c_data_structures
                           PRIVATE ${cwpack_SOURCE_DIR}/src)

add_library(oif_util SHARED oif_util.c)
target_include_directories(oif_util PUBLIC ${CMAKE_SOURCE_DIR}/oif/include)
