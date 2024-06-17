# Frontend C
add_library(
  oif_c SHARED
  c_bindings.c
  oif_config_dict.c
  ${CMAKE_SOURCE_DIR}/oif/interfaces/c/src/qeq.c
  ${CMAKE_SOURCE_DIR}/oif/interfaces/c/src/linsolve.c
  ${CMAKE_SOURCE_DIR}/oif/interfaces/c/src/ivp.c)
target_include_directories(oif_c PUBLIC ${CMAKE_SOURCE_DIR}/oif/include)
target_include_directories(oif_c
                           PUBLIC ${CMAKE_SOURCE_DIR}/oif/interfaces/c/include/)
target_link_libraries(oif_c PRIVATE oif_dispatch)
target_link_libraries(oif_c PRIVATE HashMap::HashMap)
