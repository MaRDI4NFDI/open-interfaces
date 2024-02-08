add_library(oif_dispatch SHARED dispatch.c)
target_include_directories(oif_dispatch PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(oif_dispatch PUBLIC ${CMAKE_SOURCE_DIR}/oif/include)
target_link_libraries(oif_dispatch PRIVATE HashMap::HashMap)

add_subdirectory(lang_c)
add_subdirectory(lang_python)
