cmake_minimum_required(VERSION 3.18)

project(oif-toy-example LANGUAGES C)

add_library(oif_dispatch SHARED dispatch.c)
target_include_directories(oif_dispatch PUBLIC .)
target_link_libraries(oif_dispatch PUBLIC ffi)
target_compile_features(oif_dispatch PUBLIC c_std_11)

add_subdirectory(src/oif)
