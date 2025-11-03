add_library(oif_ivp_dopri5c SHARED dopri5c.c)
target_include_directories(oif_ivp_dopri5c
                           PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(
  oif_ivp_dopri5c PRIVATE ${PROJECT_SOURCE_DIR}/lang_c/oif_impl/include)
target_link_libraries(oif_ivp_dopri5c PRIVATE m)
target_link_libraries(oif_ivp_dopri5c PRIVATE oif_common_data_structures)
target_link_libraries(oif_ivp_dopri5c PRIVATE oif_common_util)
target_link_libraries(oif_ivp_dopri5c PRIVATE oif_c)
