add_library(oif_ivp_dopri5c SHARED dopri5c.c)
target_link_libraries(oif_ivp_dopri5c PRIVATE m)
target_include_directories(oif_ivp_dopri5c PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(oif_ivp_dopri5c
                           PRIVATE ${CMAKE_SOURCE_DIR}/oif_impl/lang_c/include)
target_link_libraries(oif_ivp_dopri5c PRIVATE oif_common_data_structures)
target_link_libraries(oif_ivp_dopri5c PRIVATE oif_common_util)
target_link_libraries(oif_ivp_dopri5c PRIVATE oif_c)
