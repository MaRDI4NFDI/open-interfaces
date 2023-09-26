# Backend Python
find_package(Python REQUIRED Interpreter Development NumPy)

add_library(oif_backend_python SHARED backend_python.c)
target_include_directories(oif_backend_python PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(oif_backend_python PRIVATE ${CMAKE_SOURCE_DIR}/oif/include)
target_include_directories(oif_backend_python PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(oif_backend_python PUBLIC Python::Python)
target_link_libraries(oif_backend_python PUBLIC Python::NumPy)
