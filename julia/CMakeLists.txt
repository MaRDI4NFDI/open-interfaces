
find_package(Julia REQUIRED)
add_library(oif_julia oif_julia.c)

target_include_directories(oif_julia PUBLIC
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<BUILD_INTERFACE:${JLCXX_INCLUDE_DIR}>
  "$<BUILD_INTERFACE:${Julia_INCLUDE_DIRS}>"
                           ${CMAKE_CURRENT_BINARY_DIR}/..
)
target_link_libraries(oif_julia $<BUILD_INTERFACE:${Julia_LIBRARY}>)

target_include_directories(oif_julia
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
                           )
install(TARGETS oif_julia DESTINATION lib)
install(FILES oif_julia.h DESTINATION include)
