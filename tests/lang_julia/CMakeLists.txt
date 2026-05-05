add_library(oif_aux_funcs SHARED aux_funcs.c)
target_include_directories(oif_aux_funcs PRIVATE ${CMAKE_SOURCE_DIR}/include)
