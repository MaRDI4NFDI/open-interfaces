# Backend C
cmake_minimum_required(VERSION 3.18)
 
project(qeq LANGUAGES C)

add_library(oif_backend_c SHARED backend_c.c qeq_impl.c qeq.c)
target_include_directories(oif_backend_c PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(oif_backend_c PUBLIC ${CMAKE_SOURCE_DIR}/include/oif/backend_c)
target_link_libraries(oif_backend_c PUBLIC ffi)

