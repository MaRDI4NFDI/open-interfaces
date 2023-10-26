add_library(oif_dispatch SHARED dispatch.c)
target_include_directories(oif_dispatch PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(oif_dispatch PUBLIC ${CMAKE_SOURCE_DIR}/oif/include)
target_include_directories(oif_dispatch PRIVATE ${CMAKE_SOURCE_DIR}/vendor/hashmap)
target_link_libraries(oif_dispatch PRIVATE HashMap::HashMap)

add_subdirectory(lang_c)


