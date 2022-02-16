

add_executable(julia_embed embed.c)

target_include_directories(julia_embed PUBLIC
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<BUILD_INTERFACE:${JLCXX_INCLUDE_DIR}>
  "$<BUILD_INTERFACE:${Julia_INCLUDE_DIRS}>"
)
target_link_libraries(julia_embed $<BUILD_INTERFACE:${Julia_LIBRARY}>)
