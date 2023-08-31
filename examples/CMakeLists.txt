cmake_minimum_required(VERSION 3.18)

add_executable(qeq_from_c qeq_c_c.c)
target_include_directories(qeq_from_c PRIVATE include)
target_link_libraries(qeq_from_c PRIVATE oif_frontend_c)
