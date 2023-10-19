add_library(oif_qeq_c_qeq_solver SHARED qeq.c)
target_link_libraries(oif_qeq_c_qeq_solver PRIVATE m)
target_include_directories(oif_qeq_c_qeq_solver PRIVATE ${CMAKE_SOURCE_DIR}/oif/include)
target_include_directories(oif_qeq_c_qeq_solver PRIVATE ${CMAKE_SOURCE_DIR}/oif_impl/c/include)
