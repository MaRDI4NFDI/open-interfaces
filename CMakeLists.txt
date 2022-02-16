cmake_minimum_required(VERSION 3.18)

# set the project name
project(julia_embed)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# make sure installed binaries/libraries still find julia
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${Julia_LIBRARY_DIR}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

option(OIF_USE_JULIA "Use tutorial provided math implementation" ON)

if(OIF_USE_JULIA)
  add_subdirectory(julia)
endif()

configure_file(oif_config.h.in oif_config.h)

add_subdirectory(main)
