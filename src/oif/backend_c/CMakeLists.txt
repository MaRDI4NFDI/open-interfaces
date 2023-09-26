# Backend C
cmake_minimum_required(VERSION 3.18)
 
project(qeq LANGUAGES C)

add_library(oif_backend_c SHARED backend_c.c)
target_include_directories(oif_backend_c PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(oif_backend_c PRIVATE ${CMAKE_SOURCE_DIR}/oif/include)
target_link_libraries(oif_backend_c PUBLIC ffi)

add_library(oif_backend_c_qeq SHARED qeq_impl.c qeq.c)
target_link_libraries(oif_backend_c_qeq PRIVATE m)
target_include_directories(oif_backend_c_qeq PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(oif_backend_c_qeq PUBLIC ${CMAKE_SOURCE_DIR}/include/oif/backend_c)

add_library(oif_backend_c_linsolve SHARED linsolve_impl.c)
target_link_libraries(oif_backend_c_linsolve PRIVATE lapack)
target_include_directories(oif_backend_c_linsolve PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(oif_backend_c_linsolve PUBLIC ${CMAKE_SOURCE_DIR}/include/oif/backend_c)
