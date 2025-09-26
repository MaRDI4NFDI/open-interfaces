# Looking for Julia executable. For some reason, `find_package(Julia)`
# misbehaves when Julia is actually not installed, so we do a premature check,
# to avoid a crash.
find_program(Julia_EXECUTABLE julia DOC "Julia executable")
if(NOT Julia_EXECUTABLE)
  return()
endif()

find_package(Julia)

if(Julia_FOUND)
  add_library(oif_bridge_julia SHARED bridge_julia.c)
  target_include_directories(oif_bridge_julia
                             PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_include_directories(oif_bridge_julia
                             PUBLIC $<BUILD_INTERFACE:${Julia_INCLUDE_DIRS}>)
  target_link_libraries(oif_bridge_julia
                        PRIVATE $<BUILD_INTERFACE:${Julia_LIBRARY}>)
  target_link_libraries(oif_bridge_julia PRIVATE oif_common_data_structures)
else()
  message(STATUS "Julia not found")
endif()
