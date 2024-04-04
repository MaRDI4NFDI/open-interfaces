cmake_minimum_required(VERSION 3.18)

project(oif-toy-example LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Disable optimizations for Debug build type")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
endif()

# DOWNLOAD_EXTRACT_TIMESTAMP = TRUE
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

# Copy all built library targets to a common directory, so that it is easy to
# find and load them.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Add additional cmake module to find packages.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

add_subdirectory(vendor)

add_subdirectory(oif)
add_subdirectory(oif_impl)

add_subdirectory(examples)

enable_testing()
add_subdirectory(tests)
