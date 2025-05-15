find_package(Julia REQUIRED)

add_library(oif_bridge_julia SHARED bridge_julia.c)
target_include_directories(oif_bridge_julia
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(oif_bridge_julia
                           PUBLIC $<BUILD_INTERFACE:${Julia_INCLUDE_DIRS}>)
target_link_libraries(oif_bridge_julia
                      PRIVATE $<BUILD_INTERFACE:${Julia_LIBRARY}>)
target_link_libraries(oif_bridge_julia PRIVATE oif_common_data_structures)
