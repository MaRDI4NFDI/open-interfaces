add_library(oif_dispatch SHARED dispatch.c)
target_compile_features(oif_dispatch PUBLIC c_std_11)
target_include_directories(oif_dispatch PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_subdirectory(backend_c)
add_subdirectory(backend_python)

add_subdirectory(frontend_c)
