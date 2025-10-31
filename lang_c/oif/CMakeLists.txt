# User-facing C library that allows to interact with Open Interfaces.
add_library(
  oif_c SHARED
  c_bindings.c ${PROJECT_SOURCE_DIR}/lang_c/oif_interfaces/src/qeq.c
  ${PROJECT_SOURCE_DIR}/lang_c/oif_interfaces/src/linsolve.c
  ${PROJECT_SOURCE_DIR}/lang_c/oif_interfaces/src/ivp.c)
target_include_directories(oif_c PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(oif_c
    PUBLIC ${PROJECT_SOURCE_DIR}/lang_c/oif_interfaces/include/)
target_link_libraries(oif_c PUBLIC oif_common_data_structures)
target_link_libraries(oif_c PUBLIC oif_common_util)
target_link_libraries(oif_c PRIVATE oif_dispatch)
