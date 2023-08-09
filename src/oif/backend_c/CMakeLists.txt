cmake_minimum_required(VERSION 3.18)

project(qeq LANGUAGES C)

add_library(oif_backend_c SHARED qeq.c)
