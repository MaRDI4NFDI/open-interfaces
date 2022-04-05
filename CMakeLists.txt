cmake_minimum_required(VERSION 3.18)

# set the project name
project(oif_experiments)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(CheckCXXCompilerFlag)
include(GNUInstallDirs)

# make sure installed binaries/libraries still find julia
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${Julia_LIBRARY_DIR}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

option(OIF_USE_JULIA "Build Julia bindings" ON)
option(OIF_USE_R "Build R bindings" ON)
option(OIF_USE_PYTHON "Build Python bindings" ON)

macro(oif_enable_compile_warnings _target)
  target_compile_options(
    ${_target}
    PRIVATE
      $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:-Wall
      -Werror
      -pedantic-errors
      -Wextra>
      $<$<CXX_COMPILER_ID:MSVC>:/W4>)
endmacro()

if(OIF_USE_JULIA)
  add_subdirectory(lang_julia)
endif()
if(OIF_USE_R)
  add_subdirectory(lang_r)
endif()
if(OIF_USE_PYTHON)
  add_subdirectory(lang_python)
endif()
add_subdirectory(connector)

configure_file(oif_config.h.in ${PROJECT_BINARY_DIR}/oif_config.h)

add_subdirectory(lang_c)
