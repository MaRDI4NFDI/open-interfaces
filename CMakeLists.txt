cmake_minimum_required(VERSION 3.10)

# set the project name
project(julia_embed)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
find_package(Julia REQUIRED)

add_executable(julia_embed src/embed.c)

target_include_directories(julia_embed PUBLIC
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<BUILD_INTERFACE:${JLCXX_INCLUDE_DIR}>
  "$<BUILD_INTERFACE:${Julia_INCLUDE_DIRS}>"
)
target_link_libraries(julia_embed $<BUILD_INTERFACE:${Julia_LIBRARY}>)
