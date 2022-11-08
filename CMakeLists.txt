cmake_minimum_required(VERSION 3.18)

# set the project name
project(oif_experiments)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# necessary for clang tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckCXXCompilerFlag)
include(GNUInstallDirs)
include(SetupPythonVirtualenv)

# make sure installed binaries/libraries still find julia
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

option(OIF_USE_JULIA "Build Julia bindings" ON)
option(OIF_USE_R "Build R bindings" ON)
option(OIF_USE_PYTHON "Build Python bindings" ON)
option(OIF_USE_CPP "Build Cpp bindings" ON)
option(OIF_USE_C "Build C bindings" ON)

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

add_custom_target(all_lang ALL)

if(OIF_USE_JULIA)
  add_subdirectory(lang_julia)
endif()

find_package(R)
if(OIF_USE_R)
  if(R_FOUND)
    add_subdirectory(lang_r)
  else()
    message(FATAL_ERROR "R bindings enabled, but R not found")
  endif()
endif()

if(OIF_USE_PYTHON)
  add_subdirectory(lang_python)
endif()

if(OIF_USE_CPP)
  add_subdirectory(lang_cpp)
endif()

if(OIF_USE_C)
  add_subdirectory(lang_c)
endif()

add_subdirectory(oif_connector)

set(M2_LIBRARY_PATH
    "${PROJECT_BINARY_DIR}/oif_connector:${PROJECT_BINARY_DIR}/lang_c:${PROJECT_BINARY_DIR}/lang_cpp:${PROJECT_BINARY_DIR}/lang_python:${PROJECT_BINARY_DIR}/lang_julia:${PROJECT_BINARY_DIR}/lang_r:${Julia_LIBRARY_DIR}"
)
configure_file(oif_config.h.in oif_config.h)
configure_file(oif_env.in oif.env)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/oif_config.h DESTINATION include)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/oif.env DESTINATION share)

add_subdirectory(docs)
add_subdirectory(examples)
# these cannot move to the subdir
include(CTest)
list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
list(APPEND CMAKE_CTEST_ARGUMENTS "-V")
#
add_subdirectory(tests)
