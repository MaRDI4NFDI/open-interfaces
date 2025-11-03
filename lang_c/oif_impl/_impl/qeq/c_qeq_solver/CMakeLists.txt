add_library(oif_qeq_c_qeq_solver SHARED qeq.c)
target_link_libraries(oif_qeq_c_qeq_solver PRIVATE m)
target_include_directories(oif_qeq_c_qeq_solver
    PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(oif_qeq_c_qeq_solver
    PRIVATE ${PROJECT_SOURCE_DIR}/lang_c/oif_impl/include)
