cmake_minimum_required(VERSION 3.18)

# set the project name
project(julia_embed)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

option(OIF_USE_JULIA "Use tutorial provided math implementation" ON)

if(OIF_USE_JULIA)
  add_subdirectory(julia)
endif()

configure_file(oif_config.h.in oif_config.h)

add_subdirectory(main)
