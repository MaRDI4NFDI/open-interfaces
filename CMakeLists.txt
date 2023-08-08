cmake_minimum_required(VERSION 3.18)

project(oif-toy-example LANGUAGES C)

add_library(oif_dispatch SHARED dispatch.c)
target_include_directories(oif_dispatch PUBLIC .)
target_compile_features(oif_dispatch PUBLIC c_std_11)
