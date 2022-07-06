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

add_custom_target(all_lang)

if(OIF_USE_JULIA)
  add_subdirectory(lang_julia)
endif()

if(OIF_USE_R)
  find_package(R REQUIRED)
  add_subdirectory(lang_r)
endif()

if(OIF_USE_PYTHON)
  add_subdirectory(lang_python)
endif()

if(OIF_USE_CPP)
  add_subdirectory(lang_cpp)
endif()

add_subdirectory(lang_c)

add_subdirectory(oif_connector)

set(M2_LIBRARY_PATH
    "${Julia_LIBRARY_DIR}:${PROJECT_BINARY_DIR}/oif_connector:${PROJECT_BINARY_DIR}/lang_c:${PROJECT_BINARY_DIR}/lang_cpp:${PROJECT_BINARY_DIR}/lang_python:${PROJECT_BINARY_DIR}/lang_julia:${PROJECT_BINARY_DIR}/lang_r"
)
configure_file(oif_config.h.in oif_config.h)
configure_file(oif_env.in oif_env)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/oif_config.h DESTINATION include)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/oif_env DESTINATION share)

add_custom_target(
  test
  COMMAND
    ${CMAKE_COMMAND} -E env "M2_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}"
    "LD_LIBRARY_PATH=${M2_LIBRARY_PATH}"
    "R_LIBOIF_CONNECTOR=${PROJECT_BINARY_DIR}/oif_connector/liboif_connector.so"
    "ENV R_HOME=/usr/lib/R" ${OIF_PYTEST} -c ${PROJECT_SOURCE_DIR}/setup.cfg
    ${PROJECT_SOURCE_DIR}/tests
  DEPENDS all_lang)
