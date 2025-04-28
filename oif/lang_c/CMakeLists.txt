# User-facing C library that allows to interact with Open Interfaces.
add_library(
  oif_c SHARED
  c_bindings.c ${CMAKE_SOURCE_DIR}/oif/interfaces/c/src/qeq.c
  ${CMAKE_SOURCE_DIR}/oif/interfaces/c/src/linsolve.c
  ${CMAKE_SOURCE_DIR}/oif/interfaces/c/src/ivp.c)
target_include_directories(oif_c PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(oif_c
                           PUBLIC ${CMAKE_SOURCE_DIR}/oif/interfaces/c/include/)
target_link_libraries(oif_c PUBLIC oif_common_data_structures)
target_link_libraries(oif_c PRIVATE oif_common_util)
target_link_libraries(oif_c PRIVATE oif_dispatch)
