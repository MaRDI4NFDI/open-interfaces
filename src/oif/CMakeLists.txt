add_library(oif_dispatch SHARED dispatch.c)
target_include_directories(oif_dispatch PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(oif_dispatch PUBLIC ${CMAKE_SOURCE_DIR}/oif/include)

add_subdirectory(backend_c)
add_subdirectory(backend_python)

add_subdirectory(frontend_c)
