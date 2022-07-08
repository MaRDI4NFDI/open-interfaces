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

find_library(flexiblas REQUIRED NAMES flexiblas)

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
    "${PROJECT_BINARY_DIR}/oif_connector:${PROJECT_BINARY_DIR}/lang_c:${PROJECT_BINARY_DIR}/lang_cpp:${PROJECT_BINARY_DIR}/lang_python:${PROJECT_BINARY_DIR}/lang_julia:${PROJECT_BINARY_DIR}/lang_r:${Julia_LIBRARY_DIR}"
)
configure_file(oif_config.h.in oif_config.h)
configure_file(oif_env.in oif.env)
configure_file(oif_env_pytest.in oif_pytest.env)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/oif_config.h DESTINATION include)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/oif.env DESTINATION share)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/oif_pytest.env DESTINATION share)

enable_testing()
add_test(
  NAME pytest
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${OIF_PYTEST} -c ${PROJECT_SOURCE_DIR}/setup.cfg -x
          ${PROJECT_SOURCE_DIR}/tests)
set_tests_properties(pytest PROPERTIES DEPENDS "all_lang")
list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
list(APPEND CMAKE_CTEST_ARGUMENTS "-V")

add_subdirectory(docs)
