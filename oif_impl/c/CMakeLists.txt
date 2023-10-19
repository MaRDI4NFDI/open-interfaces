cmake_minimum_required(VERSION 3.18)
 
project(oif_dispatch_c LANGUAGES C)

add_library(oif_dispatch_c SHARED dispatch_c.c)
target_include_directories(oif_dispatch_c PRIVATE ${CMAKE_SOURCE_DIR}/oif/include)
target_link_libraries(oif_dispatch_c PUBLIC ffi)

# add_library(oif_backend_c_linsolve SHARED linsolve_impl.c)
# target_link_libraries(oif_backend_c_linsolve PRIVATE lapack)
# target_include_directories(oif_backend_c_linsolve PRIVATE ${CMAKE_SOURCE_DIR}/oif/include)
# target_include_directories(oif_backend_c_linsolve PUBLIC ${CMAKE_SOURCE_DIR}/include/oif/backend_c)
