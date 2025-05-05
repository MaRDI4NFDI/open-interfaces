cmake_minimum_required(VERSION 3.18)

project(
  oif-toy-example
  LANGUAGES C CXX
  VERSION 0.5.5)

# Enforce using `-std=c11`, without any extensions like `gnu11`.
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Disable optimizations for Debug build type")
  string(PREPEND CMAKE_C_FLAGS_DEBUG "-O0 ")
endif()

# Incorporate additions by X/Open 7, that includes POSIX 17 additions and allow
# to use things like the `M_PI` constant in the code without warnings from
# static analyzers.
add_compile_definitions(_XOPEN_SOURCE=700)

# DOWNLOAD_EXTRACT_TIMESTAMP = TRUE
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

# Copy all built library targets to a common directory, so that it is easy to
# find and load them.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# when building, don't use the install RPATH already (but later on when
# installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

# Make sure that libraries can be found by the linker.
set(CMAKE_INSTALL_RPATH "$ORIGIN")

# Add additional cmake module to find packages.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

add_subdirectory(vendor)

add_subdirectory(common)
add_subdirectory(dispatch)

add_subdirectory(lang_python)

if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/oif")
  add_subdirectory(oif)
endif()
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/oif_impl")
  add_subdirectory(oif_impl)
endif()

if(EXISTS examples/CMakeLists.txt)
  add_subdirectory(examples)
endif()

enable_testing()
add_subdirectory(tests)
