cmake_minimum_required(VERSION 3.18)

# set the project name
project(julia_embed)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Julia REQUIRED)
add_subdirectory(julia)
