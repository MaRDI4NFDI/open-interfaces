cmake_minimum_required(VERSION 3.18)

add_executable(call_qeq_from_c call_qeq_from_c.c)
target_include_directories(call_qeq_from_c PRIVATE ${CMAKE_SOURCE_DIR}/oif/include)
target_link_libraries(call_qeq_from_c PRIVATE oif_frontend_c)

add_executable(call_linsolve_from_c call_linsolve_from_c.c)
target_include_directories(call_linsolve_from_c PRIVATE ${CMAKE_SOURCE_DIR}/oif/include)
target_link_libraries(call_linsolve_from_c PRIVATE oif_frontend_c)
