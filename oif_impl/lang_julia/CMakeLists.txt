find_package(Julia REQUIRED)

add_library(oif_dispatch_julia SHARED dispatch_julia.c)
target_include_directories(oif_dispatch_julia
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(oif_dispatch_julia
                           PUBLIC $<BUILD_INTERFACE:${Julia_INCLUDE_DIRS}>)
target_link_libraries(oif_dispatch_julia
                      PRIVATE $<BUILD_INTERFACE:${Julia_LIBRARY}>)
target_link_libraries(oif_dispatch_julia PRIVATE oif_common_data_structures)
